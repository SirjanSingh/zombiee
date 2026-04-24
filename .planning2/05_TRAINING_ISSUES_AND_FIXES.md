# Training Issues, Fixes, and Better Ideas

> Created 2026-04-24 by Antigravity analysis of the full codebase.
> Read this before running training on Colab or DGX.

---

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Environment (survivecity_env/) | ✅ Complete | All 9 files, well-structured, tests pass |
| Server (server/app.py) | ✅ Complete | OpenEnv-compliant endpoints |
| Baseline inference | ✅ Complete | Random + LLM modes |
| Training script (train.py) | ⚠️ Has Issues | See below — 4 bugs identified |
| Kaggle notebook | ✅ Exists | Kaggle-specific, needs Colab adaptation |
| Colab notebook | ❌ Missing | Judges explicitly require Colab-runnable notebook |
| Eval + plots | ⚠️ Placeholder | eval.py uses random baseline for "trained" — no LoRA loading |
| HF Space deploy | ❌ Not done | Phase 10 |

---

## Critical Bugs Found in Training Pipeline

### Bug 1: GRPO reward_fn only executes ONE step, not a full episode

**File:** `training/train.py` → `create_reward_fn()`

The reward function sends a single `/step` to the env per completion. But the env is stateful — multiple GRPO generations (8) call `/step` on the SAME episode in parallel, causing state corruption. The env doesn't get reset between GRPO generations.

```python
# CURRENT (broken):
def reward_fn(prompts, completions, **kwargs):
    for c in completions:
        action = json.loads(c)
        r = requests.post(f"{env_url}/step", json=action)  # ONE step on shared env
        rewards.append(r.json().get("reward", 0.01))
```

**Fix:** The reward_fn must reset the env, roll out a FULL episode per completion, and return the episode's total/terminal reward. Each call to reward_fn should be independent.

### Bug 2: Scenario dataset prompts are stale

**File:** `training/train.py` → `build_scenario_dataset()`

The dataset is built once at startup by calling `/reset` 200 times and capturing the initial observation descriptions. These become the prompts for GRPO. But:
- The descriptions are static snapshots of step 0
- GRPO generates completions (actions) for these prompts
- The reward_fn then calls `/step` with the action — but the env has moved on since the dataset was built

**Fix:** The prompt should be a generic scenario description (not a specific observation), or the reward_fn should handle the full reset→rollout loop internally.

### Bug 3: `num_generations=4/8` calls reward_fn with that many completions — but env is a singleton

GRPO generates N completions per prompt, then calls `reward_fn(prompts, completions)` with ALL of them. The reward_fn iterates over completions and sends each as a `/step` — but they all hit the same singleton `_env` instance in `server/app.py`. The env state gets corrupted by interleaved actions from different GRPO generations.

**Fix:** Either:
- (A) Use a local env instance inside reward_fn (no HTTP — import SurviveCityEnv directly), create a new instance per evaluation
- (B) Add an episode reset before each completion evaluation in the reward_fn

### Bug 4: Action parsing doesn't validate action_type

The `parse_completion_to_action` is missing — `create_reward_fn` uses `json.loads(text)` directly. If the model outputs a malformed action_type (e.g., `"move"` instead of `"move_up"`), the server returns a 500 error. The try/except catches it but doesn't help the model learn what valid actions look like.

---

## Better Ideas for Training

### Idea 1: Skip HTTP, use env directly (10× faster)

Instead of running the env as an HTTP server and calling it via requests, import `SurviveCityEnv` directly in the reward function. This eliminates:
- HTTP serialization/deserialization overhead
- Port management in Colab
- Singleton state corruption bugs

```python
from survivecity_env.env import SurviveCityEnv

def reward_fn(prompts, completions, **kwargs):
    rewards = []
    for c in completions:
        env = SurviveCityEnv()
        obs = env.reset()
        action = parse_action(c)
        obs = env.step(action)
        rewards.append(obs["reward"])
    return rewards
```

### Idea 2: Full-episode rollout in reward_fn (better learning signal)

Instead of scoring a single action, roll out the full episode and use the terminal reward. This gives GRPO a much stronger signal about what "good" actions look like in context.

### Idea 3: Use pre-tokenized Unsloth model on Colab

The Kaggle notebook already uses `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (pre-quantized). This is critical for Colab's T4 GPU (16GB). The vanilla `Qwen/Qwen2.5-3B-Instruct` would need to be quantized at load time, which is slower.

### Idea 4: Reduce training to fit Colab session

Colab free tier: ~12h with T4. Suggested tuning:
- `MAX_STEPS=500` (enough to show improvement curve)
- `NUM_GENERATIONS=4` (halves memory)
- `MAX_SEQ_LENGTH=2048` (fits T4)
- `SAVE_STEPS=50` (frequent saves in case session dies)

### Idea 5: Multi-step action scoring (middle ground)

Instead of full-episode vs single-step, score the model on a 5-step window:
- Reset env, advance to a random step
- Model generates 5 actions in sequence
- Score = sum of 5 step rewards
- Better credit assignment than single step, faster than full episode

---

## Dependency Compatibility Notes (for Colab)

| Package | Known issue | Fix |
|---------|-----------|-----|
| `torchao` | ≥0.8 requires `torch.int1` (torch ≥2.6). Colab ships torch 2.4/2.5 | Pin `torchao==0.7.0` |
| `unsloth` | Eager CUDA probe at import fails in CPU-only contexts | Install with `--no-deps` |
| `trl` | GRPOTrainer API changed between 0.8 and 0.12+ | Pin version matching Colab's torch |
| `transformers` | quantizer auto-import chain can break with wrong torchao | Sanity-check import after install |

---

## What to Do Next (Priority Order)

1. **Create Colab notebook** — adapted from Kaggle one, with Colab-specific auth (userdata instead of kaggle_secrets)
2. **Fix reward_fn** — use local env instance + full-episode rollout
3. **Run 500 steps on Colab** — generate real training curves
4. **Generate plots** — fix eval.py to load trained LoRA
5. **Deploy HF Space** — Dockerfile is ready
6. **Record demo video** — baseline vs trained transcripts
