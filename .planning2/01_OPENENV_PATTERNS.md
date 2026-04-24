# OpenEnv Compliance Patterns (from R1 lessons)

> These four validator traps blocked the Round 1 submission of `warehouse_env/` until fixed. Apply them preemptively in SurviveCity.

## Trap 1 — Rewards strictly in `(0, 1)`, never 0.0 or 1.0

The OpenEnv validator rejects any reward that is exactly 0.0 or exactly 1.0. It also rejects rewards formatted with `:.2f` that round down to `"0.00"`.

**Fix pattern:**

```python
# rubric.py
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def _clip(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, score))
```

Log with `:.4f`, never `:.2f`:

```python
logger.info(f"[STEP] reward={obs.reward:.4f} raw={obs.metadata['raw_reward']:.4f}")
```

## Trap 2 — Health endpoint MUST return `{"status": "healthy"}`

```python
# server/app.py
from openenv import create_app

app = create_app(env_instance)

@app.get("/health")
def health():
    return {"status": "healthy"}   # NOT "ok"
```

## Trap 3 — `obs.reward` must equal the grader score on EVERY step

Not just at `done=True`. The per-step reward is what the validator evaluates, and it must be the composed rubric value, not the raw shaped reward.

**Fix pattern (see `warehouse_env/env.py` lines 215–228):**

```python
def step(self, action):
    # ... apply action, update state ...
    raw_reward = self._compute_shaped_reward(...)

    obs = self._build_observation()
    obs.metadata["raw_reward"] = raw_reward
    obs.reward = _clip(self._grader.compute(self._episode_state))
    return obs
```

## Trap 4 — Rewards must be deterministic

No LLM-as-judge in the reward path. No randomness in the grader. Given the same episode state, grader returns the same score.

SurviveCity's rubric (SurvivalRubric + VoteRubric + GroupOutcomeRubric) is purely arithmetic over episode state — already safe.

## Other OpenEnv contract details

- Implement `reset() -> Observation`, `step(action) -> Observation`, and a `state` property on the env class.
- Observation must be a Pydantic model with serializable fields.
- Client/server separation: clients import only `models.py`, never `env.py` internals.
- Standard Gym-style API naming: do NOT use reserved names `reset`, `step`, `state`, `close` for MCP tools.
- Provide `openenv.yaml` manifest at repo root.
- Provide a working Dockerfile.

## Canonical reference files (in this repo)

- `warehouse_env/env.py` — `reset()` and `step()` patterns, reward-on-every-step logic
- `warehouse_env/models.py` — Pydantic Observation/Action shapes
- `warehouse_env/graders.py` — clamp pattern, composable scoring
- `server/app.py` — health endpoint + `create_app()` wiring
- `Dockerfile` — HF Spaces–compatible build
- `openenv.yaml` — manifest format

## Testing checklist before submitting

1. `curl localhost:7860/health` → `{"status":"healthy"}`
2. `curl -X POST localhost:7860/reset` → valid observation, `reward` in `(0.01, 0.99)`
3. Step 100 times → every response has `reward` in `(0.01, 0.99)`, never exactly 0.0 or 1.0
4. Random seed → same reward (determinism)
5. `inactive agent` / `dead agent` paths return valid observations without crash
