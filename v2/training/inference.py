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
import re
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

# Regex fallbacks for lenient parsing (used when strict JSON fails).
# Pattern matches: action_type: "move_up" / action_type="move_up" /
# "action_type": move_up / action_type = "move_up" / etc.
_ACTION_TYPE_RE = re.compile(
    r"""action_type[\s"']*[:=]+[\s"']*(\w+)""", re.IGNORECASE
)
_VOTE_TARGET_RE   = re.compile(r"""vote_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_SCAN_TARGET_RE   = re.compile(r"""scan_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_INJECT_TARGET_RE = re.compile(r"""inject_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_GIFT_TARGET_RE   = re.compile(r"""gift_target[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_ITEM_SLOT_RE     = re.compile(r"""item_slot[\s"']*[:=]+[\s"']*(\d+)""", re.IGNORECASE)
_MESSAGE_RE       = re.compile(r"""message[\s"']*[:=]+[\s"']*["']([^"']{1,40})["']""", re.IGNORECASE)


def parse_action(text: str, agent_id: int) -> Optional[dict]:
    """Parse an action from model output.

    Tries three strategies in order:
      1. Strict JSON object extraction (any {...} substring that loads).
      2. Regex extraction of `action_type: <word>` and optional fields
         from prose like "I'll move_up" or `action_type=eat`.
      3. Last-resort: scan for any literal valid action_type word
         anywhere in the text (catches "I will move_up to the food cell").

    Returns None only if NONE of the 14 valid action_types appears
    anywhere in the completion. This is critical for GRPO — if every
    completion in a group returns None, all rewards floor at 0.01,
    `reward_std=0`, and the gradient signal is dead.
    """
    if not text:
        return None
    text = text.strip()

    # Strip markdown code fences if present
    fenced = text
    if fenced.startswith("```"):
        parts = fenced.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            fenced = inner.strip()

    # --- Strategy 1: strict JSON extraction ---
    for start in range(len(fenced)):
        if fenced[start] == "{":
            for end in range(len(fenced), start, -1):
                if fenced[end - 1] == "}":
                    try:
                        d = json.loads(fenced[start:end])
                        if not isinstance(d, dict):
                            continue
                    except (json.JSONDecodeError, TypeError):
                        continue
                    d.setdefault("agent_id", agent_id)
                    if d.get("action_type") in VALID_ACTION_TYPES:
                        return d

    # --- Strategy 2: regex-extract action_type + optional fields ---
    m = _ACTION_TYPE_RE.search(text)
    if m:
        atype = m.group(1).lower()
        if atype in VALID_ACTION_TYPES:
            result: dict[str, Any] = {"agent_id": agent_id, "action_type": atype}
            for field, regex in (
                ("vote_target",   _VOTE_TARGET_RE),
                ("scan_target",   _SCAN_TARGET_RE),
                ("inject_target", _INJECT_TARGET_RE),
                ("gift_target",   _GIFT_TARGET_RE),
                ("item_slot",     _ITEM_SLOT_RE),
            ):
                fm = regex.search(text)
                if fm:
                    try:
                        result[field] = int(fm.group(1))
                    except ValueError:
                        pass
            mm = _MESSAGE_RE.search(text)
            if mm:
                result["message"] = mm.group(1)[:40]
            return result

    # --- Strategy 3: last-resort word scan ---
    # Lowercase scan so "Move_up" / "MOVE_UP" still match. Sort by length
    # so "move_up" matches before "move" (no false-positive on "move").
    text_lower = text.lower()
    for atype in sorted(VALID_ACTION_TYPES, key=len, reverse=True):
        if atype in text_lower:
            return {"agent_id": agent_id, "action_type": atype}

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
