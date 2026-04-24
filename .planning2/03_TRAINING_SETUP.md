# Training Setup — SurviveCity

Qwen2.5-3B-Instruct + LoRA + GRPO via Unsloth and HuggingFace TRL.

## Why this stack

| Component | Choice | Reason |
|---|---|---|
| Base model | **Qwen2.5-3B-Instruct** | Small enough to fit + train fast; weak enough baseline to show big improvement curve |
| Adapter | **LoRA** (r=16, alpha=32) | Trains ~20M params (1% of model), ~40MB file, no full-model finetune needed |
| Loader | **Unsloth** (`FastLanguageModel`) | 2× faster training, 4-bit loading, excellent Qwen support |
| Algorithm | **GRPO** via `trl.GRPOTrainer` | No value network, group-relative advantage, DeepSeek-R1 proven recipe |
| Env API | OpenEnv HTTP client | Matches R1 pattern, validator-compliant |

## Why NOT other choices

- **7B model:** too slow per rollout; baseline is already decent → smaller visible delta hurts the 20% rubric
- **1.5B model:** can't reason well enough about multi-agent vote dynamics; may flatline
- **PPO instead of GRPO:** needs value network, harder to tune, slower convergence
- **Full finetune:** needs 60GB+ VRAM, no benefit over LoRA for this task
- **DPO:** needs preference pairs, not reward signal — wrong paradigm

## Hyperparameters

```python
# training/train.py
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    max_seq_length=4096,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

config = GRPOConfig(
    output_dir="./lora_v1",
    num_generations=8,           # 8 rollouts per prompt
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_steps=4000,
    save_steps=500,
    logging_steps=10,
    max_prompt_length=1024,
    max_completion_length=512,
    temperature=0.9,
    beta=0.04,                   # KL coefficient
    report_to="wandb",           # or "tensorboard"
)
```

## Reward function — hooking the env

```python
import requests
import json

ENV_URL = "http://localhost:7860"

def reward_fn(prompts, completions, **kwargs):
    """
    Called by GRPOTrainer per batch.
    Each (prompt, completion) pair represents one agent's action in one env step.
    For SurviveCity, we actually roll out full episodes where the model drives all 3 agents,
    and score each agent's trajectory separately.
    """
    rewards = []
    for completion in completions:
        # parse the completion into a structured action
        try:
            action = parse_completion_to_action(completion)
        except Exception:
            rewards.append(0.01)   # malformed output = min reward
            continue
        
        # Send action to env — note: you'll want to roll out whole episodes,
        # not individual steps, for meaningful GRPO signal.
        # See training/rollout.py for the wrapper.
        reward = rollout_and_score(action)
        rewards.append(reward)
    return rewards
```

## Multi-agent rollout strategy

GRPO expects (prompt, completion) pairs. For multi-agent, you have options:

**Option A — flatten every agent-step into a training example.**
- Each env step becomes one training example: `(prompt = agent's observation as prompt, completion = action)`.
- Reward = per-step `obs.reward`.
- Simpler. What TRL expects out of the box. **Start here.**

**Option B — one training example per episode, score = total episode reward.**
- Each episode = one example.
- Requires custom rollout machinery.
- Better credit assignment for long-horizon tasks but harder to set up.

Go with **Option A** for v1. Switch to B if results plateau.

## Post-mortem integration

Post-mortems are NOT part of the reward. They are a prompt-conditioning mechanism:

```python
# Between episodes (or at reset), prepend postmortems to the agent's system prompt
def build_system_prompt(agent_id: int, postmortem_buffer: dict):
    past = postmortem_buffer.get(agent_id, [])[-3:]
    past_block = "\n".join(past) if past else "(no past deaths yet)"
    return f"""You are agent A{agent_id} in a zombie-apocalypse survival simulation.

3 agents share a 10x10 city. 3 zombies chase you. You need food (F cells) and the safehouse (S cells heal you).
One of the 3 agents is SECRETLY INFECTED. At step 30 their infection reveals. At step 50 you all vote to lock one agent out of the safehouse.

PAST FAILURES (learn from these!):
{past_block}

Output your next action as JSON: {{"action_type": "...", ...}}
"""
```

`postmortem_buffer` is maintained outside the env, in the training loop, keyed by agent slot.

## Training wallclock budget

| Step | Machine | Time |
|---|---|---|
| Pre-training v1 (4000 steps × 8 rollouts) | DGX (single H100 or A100) | **~5 hours** |
| On-site v2 continuation (+2000 steps) | HF Spaces credits | **~6 hours** |

Start v1 pretraining the night of 2026-04-24 so it finishes before the 25th flight.

## Checkpointing strategy

- Save every 500 steps to `./lora_v1/checkpoint-{step}/`
- Also save final adapter to `./lora_v1/`
- Keep at least 3 checkpoints: step 1000, 2000, final
- For v2 on-site: load latest v1 checkpoint as starting adapter, continue training with same LR

## v1 → v2 upgrade

```python
# On v2 training script
model, tok = FastLanguageModel.from_pretrained(MODEL_NAME, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, ...)
model.load_adapter("./lora_v1", adapter_name="default")    # warm-start
# then continue training with v2 env (more agents, richer mechanics)
```

v2 env must be action-space-compatible with v1 (no removed action types). Add new action types as new enum values — old weights remain valid.

## Eval recipe (post-training)

```python
# training/eval.py
from peft import PeftModel

base, tok = FastLanguageModel.from_pretrained(MODEL_NAME, load_in_4bit=True)
trained = PeftModel.from_pretrained(base, "./lora_v1")

metrics = {"baseline": [], "trained": []}
for model_name, m in [("baseline", base), ("trained", trained)]:
    for episode in range(200):
        obs = env_client.reset()
        total, done = 0.0, False
        while not done:
            action = sample_action(m, tok, obs)
            obs = env_client.step(action)
            total += obs["reward"]
            done = obs["done"]
        metrics[model_name].append({
            "reward": total,
            "survived": obs["metadata"]["healthy_alive"] >= 1,
            "correct_vote": obs["metadata"]["vote_correct"],
        })

# dump to plots/
```

## Troubleshooting common failure modes

| Symptom | Fix |
|---|---|
| Loss NaN / divergence | Drop LR to 5e-6, reduce `num_generations` to 4 |
| All completions become identical (mode collapse) | Increase `temperature` to 1.1, add entropy bonus, randomize episode seeds more |
| Reward flatlines after 500 steps | Rubric magnitudes too small → multiply SurvivalRubric values by 2× |
| OOM | Switch to 4-bit with `bnb_4bit_compute_dtype=torch.float16`, reduce `max_seq_length` to 2048 |
| Model outputs non-JSON actions | Add one-shot example in system prompt, penalize parse failures with 0.01 reward |
