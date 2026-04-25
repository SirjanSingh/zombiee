"""Inference helpers — JSON action parsing + LLM action functions.

Reused by:
    - training.train (reward function rolls out random actions after the
      model's first action; needs robust JSON parsing)
    - training.eval (drives the trained policy across full episodes)
    - training.simulator (single-episode pretty playthrough)
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Callable, Optional

logger = logging.getLogger("survivecity_v2.inference")


VALID_ACTION_TYPES = frozenset({
    "move_up", "move_down", "move_left", "move_right",
    "eat", "wait", "vote_lockout", "broadcast",
    "drink", "scan", "pickup", "drop", "give", "inject",
})

# Used for random rollouts in the GRPO reward function and as the
# random-baseline policy in eval / simulator.
RANDOM_NON_VOTE_ACTIONS = [
    "move_up", "move_down", "move_left", "move_right",
    "eat", "drink", "wait", "pickup",
]


def parse_action(text: str, agent_id: int) -> Optional[dict]:
    """Parse a JSON action from model output. Tolerates markdown fences and
    leading/trailing prose; returns None if no valid action_type can be
    extracted.
    """
    text = text.strip()
    # Markdown code-fence stripping
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()
    # Find a JSON object within the text
    for start in range(len(text)):
        if text[start] == "{":
            for end in range(len(text), start, -1):
                if text[end - 1] == "}":
                    try:
                        d = json.loads(text[start:end])
                        if not isinstance(d, dict):
                            continue
                    except (json.JSONDecodeError, TypeError):
                        continue
                    d.setdefault("agent_id", agent_id)
                    if d.get("action_type") in VALID_ACTION_TYPES:
                        return d
    return None


def random_action(agent_id: int, obs: dict, rng: Optional[random.Random] = None) -> dict:
    """Random baseline action. Casts a random vote at vote-phase steps."""
    rng = rng or random
    s = obs.get("step_count", 0)
    if s in (30, 60, 90):
        return {
            "agent_id": agent_id,
            "action_type": "vote_lockout",
            "vote_target": rng.choice([0, 1, 2, 3, 4]),
        }
    return {"agent_id": agent_id, "action_type": rng.choice(RANDOM_NON_VOTE_ACTIONS)}


def make_llm_action_fn(model: Any, tokenizer: Any, max_new_tokens: int = 96) -> Callable:
    """Build an action_fn that calls the LLM and parses output as JSON.

    On parse failure, falls back to random_action (so an episode never stalls).
    """
    import torch
    from survivecity_v2_env.prompts import build_system_prompt

    def llm_action(agent_id: int, obs: dict) -> dict:
        description = obs.get("description", "")
        prompt = build_system_prompt(agent_id, description)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "What is your next action? Respond with JSON only."},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.debug(f"llm generation error: {e}")
            return random_action(agent_id, obs)

        parsed = parse_action(response, agent_id=agent_id)
        if parsed is None:
            return random_action(agent_id, obs)

        # Sanitise: ensure broadcast message is short
        if parsed.get("action_type") == "broadcast":
            msg = parsed.get("message")
            if isinstance(msg, str):
                parsed["message"] = msg[:40]
            else:
                parsed["message"] = "alert"

        return parsed

    return llm_action
