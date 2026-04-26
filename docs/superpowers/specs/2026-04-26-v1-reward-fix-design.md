# SurviveCity v1 — reward-function fix design

**Branch**: `v1new` (off `master`)
**Date**: 2026-04-26
**Author**: Team PyGuys
**Status**: design accepted, awaiting implementation plan

## Why this exists

The original v1 GRPO run (12 steps on Colab T4) and the extended retrain
(20 steps on Kaggle T4 → `noanya/zombiee-v1-extended`) both produced LoRAs
that look "trained" but cannot be distinguished from random + base-LM
behaviour in eval. Per-step trace analysis of the eval log shows the
dominant trained-policy failure mode is collective starvation before
step 50 (~75% of episodes), with two structural bugs in the reward
function explaining why no learning occurred:

1. **GRPO trained on the clipped reward (0.01 floor) instead of the raw
   signed reward.** Per-step rewards are small (±0.005 to ±0.10), and
   negative composite rewards clip to 0.01 for ~95% of timesteps. GRPO's
   group-normalised advantages collapse to zero variance for those steps,
   producing no gradient.
2. **`vote_reward` fires every step after step 51 instead of once.** The
   gate is `state.vote_resolved`, which is set sticky-True at step 51 and
   never cleared. Every post-vote step bakes in ±0.20 / ±0.30 / −0.05,
   dominating cumulative reward by ~10× and producing the constant
   `raw=−0.345` plateau visible in the eval log.

Three more issues compound the problem and are addressed in this design:

3. **Damage credit assignment hole.** `_AgentInternal.reset_step_flags()`
   is called at the START of `apply_agent_action`, which clears
   `damage_this_step` before that agent's `compose_reward` reads it. Both
   zombie collisions (set in `advance_zombies` after the round's last
   actor) and infected attacks (set in `_infected_attack` during the
   infected agent's turn) end up cleared before reaching the victim's
   reward. No agent ever receives a damage signal.
4. **No forage shaping.** Eat reward (+0.05) is too sparse to bootstrap
   the food→eat loop on a 100-step horizon with 4 food cells. Eval shows
   trained agents never reach food.
5. **Wait/broadcast spam is free.** No anti-camping cost; the model can
   collect +0.005/step indefinitely by issuing wait actions.

## Goals

After applying the design, on Kaggle T4 with 12h compute and `noanya/v1-zombiee`
as the HF target:

- GRPO sees non-zero reward variance from step 1 (smoke-test enforced)
- Trained policy survives ≥20% of eval episodes (vs. baseline 10% on
  `noanya/zombiee`)
- Mean total reward ≥ 1.0 (vs. baseline 0.80)
- Mean episode length ≥ 50 steps (vs. baseline 37.6)
- Starvation deaths fraction ≤ 40% (vs. baseline ~75%)

If at least 3 of 5 targets are met, the new LoRA replaces `noanya/zombiee`
as the headline artefact for the submission. Otherwise, fall back to
the original `noanya/zombiee` step-12 LoRA — `noanya/v1-zombiee/<timestamp>/`
remains as an isolated experiment that does not affect the existing
artefact.

## Out of scope

These are real issues but deferred for compute / time reasons:

- **Real model rollouts in GRPO inner loop** (priority #4 from the user's
  list). Replacing the random-action rollout with model-driven rollout
  multiplies per-step compute by ~10×; at v1's ~19 min/step baseline that
  becomes ~3h/step, leaving room for only 3-4 GRPO steps in the 12h
  budget. Requires a separate compute window.
- **Per-step raw reward accumulation across the rollout**. Currently
  `reward_fn` returns only the final-step `raw_reward`. Accumulating
  across the rollout would give a fuller signal but moves the diff
  beyond the user's "one-line train.py change" specification. Optional
  follow-up if step-1 reward variance is borderline.
- **Vote magnitude rebalance, broadcast cost, zombie-proximity penalty,
  infected deception incentive.** All identified in the user's design
  critique; deferred to keep the diff small and unambiguous.

## Approach

**Approach 2 — Potential-based shaping, plus wait penalty.** Selected
over a surgical-only or aggressive-rebalance variant because:

- Surgical-only does not fix the damage-credit hole (priority bug #2),
  whose absence means zombie-avoidance has no gradient.
- Aggressive-rebalance introduces 5 simultaneous hyperparameters with no
  compute budget to tune them; risks breaking learning entirely.
- Approach 2 fixes every flagged bug AND adds the single most impactful
  shaping term (forage potential) using the standard Ng et al. 1999
  potential-based form so the optimal policy is provably preserved.

Diff size: ~60 lines of functional code across 4 modified files
(`game.py`, `rubric.py`, `env.py`, `train.py`), plus ~110 lines of
new tests/smoke scripts in 2 new files (`tests/test_reward_fixes.py`,
`scripts/smoke_test_reward_variance.py`). No deletions.

## Architecture

Three rubric-related modules change, plus one line in train.py and a
new training callback. The reward signal flow:

```
agent acts in apply_agent_action()
  ├─ snapshot prev_food_dist (Manhattan distance to nearest food)
  ├─ apply movement / eat / vote / broadcast / wait
  ├─ snapshot cur_food_dist
  └─ damage from any source (zombies / infected / starvation)
       → pending_damage_reward accumulates (carries across rounds)

env.step() calls compose_reward(state, agent_id)
  └─ survival_reward reads pending_damage_reward + forage delta + wait cost
  └─ vote_reward checks vote_just_resolved (one-shot bool)
  └─ group_outcome (terminal only)

compose_reward drains agent.pending_damage_reward = 0.0 after summing

obs.metadata.raw_reward = unclipped sum
obs.reward = clipped (0.01, 0.99) — OpenEnv contract preserved

train.py reward_fn reads obs.metadata.raw_reward
  → GRPO sees real gradient
```

### Files touched

- `v1/survivecity_env/game.py` — new fields on `_AgentInternal` and
  `EpisodeState`, instrumentation at every damage site, `_min_food_dist`
  helper, snapshots in `apply_agent_action`, lifecycle in `advance_step`
- `v1/survivecity_env/rubric.py` — `survival_reward` body extended,
  `vote_reward` gate changed, `group_outcome_reward` docstring rewrite,
  `compose_reward` drains `pending_damage_reward` after summing
- `v1/survivecity_env/env.py` — no change (drain now owned by `compose_reward`)
- `v1/training/train.py` — read raw reward at line 174; new
  `TimestampedHubPushCallback`
- `v1/tests/test_reward_fixes.py` — new test file, 7 unit tests
- `v1/scripts/smoke_test_reward_variance.py` — new pre-launch smoke test

### Files NOT touched

`infection.py`, `models.py`, `prompts.py`, `postmortem.py`, `layout.py`,
`eval.py`, `inference.py`, frontend, report. The env's external contract
(`obs.reward` clipped to (0.01, 0.99)) is unchanged — only the
training pipeline reads the new raw signal.

### Branch and Hub targets

- GitHub: existing `https://github.com/SirjanSingh/zombiee` repo, new
  branch `v1new`. All commits land on this branch; merge to master only
  after eval confirms the fixes worked.
- HF model repo: `noanya/v1-zombiee`. Every artefact (intermediate
  checkpoints, final adapter, TB events, eval results) goes into a
  timestamped subfolder `noanya/v1-zombiee/<timestamp>/` where
  `<timestamp>` is the training-launch UTC time formatted as
  `YYYY-MM-DDTHHMMZ` (e.g., `2026-04-26T1300Z`). Future runs land in
  their own subfolder; no overwrites.
- HF model repos NOT touched: `noanya/zombiee` (the proven baseline),
  `noanya/zombiee-v1-extended` (the prior failed retrain).

## Components

### `v1/survivecity_env/game.py`

Add 4 fields to `_AgentInternal`:

```python
pending_damage_reward: float = 0.0     # carries damage cost across rounds; drained inside compose_reward
prev_food_dist_this_step: int = -1     # snapshot before action for forage shaping
cur_food_dist_this_step: int = -1      # snapshot after action
last_action_this_step: str = ""        # used by survival_reward to detect "wait"
```

Update `reset_step_flags()` so `pending_damage_reward` is NOT cleared
(`compose_reward` drains it after reading, see rubric.py changes below):

```python
def reset_step_flags(self):
    self.ate_this_step = False
    self.damage_this_step = 0
    self.died_this_step = False
    self.last_action_this_step = ""
    # pending_damage_reward NOT reset — compose_reward drains after reading
    # forage dists overwritten by next apply_agent_action's snapshots
```

Add 1 field to `EpisodeState`:

```python
vote_just_resolved: bool = False       # True for exactly the round when vote resolves; False otherwise
```

Add a helper near other layout helpers:

```python
def _min_food_dist(row: int, col: int) -> int:
    """Manhattan distance to nearest food cell. Returns 0 if standing on food, 99 if no food."""
    if not FOOD_CELLS:
        return 99
    return min(abs(fr - row) + abs(fc - col) for (fr, fc) in FOOD_CELLS)
```

Modify `apply_agent_action`:

```python
def apply_agent_action(state, agent_id, action_type, vote_target=None, message=None):
    agent = state.agents[agent_id]
    if not agent.is_alive: return

    agent.reset_step_flags()
    agent.last_action_this_step = action_type
    agent.prev_food_dist_this_step = _min_food_dist(agent.row, agent.col)

    # Existing hunger / starvation handling, but starvation HP loss now also accumulates pending damage:
    if agent.hunger >= 15:
        agent.hp -= 1
        agent.damage_this_step += 1
        agent.pending_damage_reward -= 0.10                     # NEW
        if agent.hp <= 0:
            _kill_agent(agent, state, "hunger")

    # Existing movement / eat / vote / broadcast / wait branches unchanged.

    # Existing infected attack call — _infected_attack now also accumulates pending_damage_reward (see below).

    # Existing safehouse healing.

    agent.cur_food_dist_this_step = _min_food_dist(agent.row, agent.col)
```

Modify `_infected_attack`:

```python
def _infected_attack(attacker, state):
    for other in state.agents:
        if other.agent_id == attacker.agent_id or not other.is_alive:
            continue
        if abs(other.row - attacker.row) <= 1 and abs(other.col - attacker.col) <= 1:
            other.hp -= 1
            other.damage_this_step += 1
            other.pending_damage_reward -= 0.10                 # NEW
            if other.hp <= 0:
                _kill_agent(other, state, "infected_attack")
            # Existing infection-spread cap unchanged.
```

Modify `advance_zombies` (zombie collision damage now also accumulates):

```python
# Inside the existing loop body where zombie hits an agent:
agent.hp -= 1
agent.damage_this_step += 1
agent.pending_damage_reward -= 0.10                             # NEW
if agent.hp <= 0:
    _kill_agent(agent, state, "zombie_attack")
```

Modify `advance_step` to manage the `vote_just_resolved` lifecycle:

```python
def advance_step(state):
    state.step_count += 1
    state.broadcasts = []
    state.vote_just_resolved = False                            # cleared at start of every round
    check_infection_reveal(state)
    if state.step_count == 50:
        state.vote_phase_active = True
    if state.step_count == 51 and not state.vote_resolved:
        resolve_vote(state)
        state.vote_just_resolved = True                         # set true for exactly this round
    check_terminal(state)
```

### `v1/survivecity_env/rubric.py`

Replace `survival_reward` body to add wait penalty, drain pending damage,
and apply forage potential delta:

```python
def survival_reward(state, agent_id):
    a = state.agents[agent_id]
    r = 0.0

    if a.is_alive:
        r += 0.005
        if a.ate_this_step:
            r += 0.05
        if a.hunger >= 10:
            r -= 0.05
        if a.last_action_this_step == "wait":                   # NEW: wait penalty
            r -= 0.002
        r += a.pending_damage_reward                            # NEW: replaces direct damage_this_step branch
        # Potential-based forage shaping (Ng et al. 1999):
        # F(s, a, s') = γ · Φ(s') − Φ(s) where Φ = -0.005 × min_food_dist.
        # γ=1.0; the shaped reward is +0.005 per cell closer, −0.005 per cell farther,
        # net-zero on round-trips. Provably preserves optimal policy.
        if a.prev_food_dist_this_step >= 0 and a.cur_food_dist_this_step >= 0:
            r += 0.005 * (a.prev_food_dist_this_step - a.cur_food_dist_this_step)  # NEW

    if a.died_this_step:
        r -= 0.50

    return r
```

Replace `vote_reward` gate to use `vote_just_resolved` (one-shot bool):

```python
def vote_reward(state, agent_id):
    if not state.vote_just_resolved:                            # NEW: one-shot gate
        return 0.0

    vote = state.votes_cast.get(agent_id)

    if agent_id == state.infected_id:
        if vote is None:               return -0.05
        if vote == state.infected_id:  return -0.30
        return 0.30

    if vote is None:                   return -0.05
    if vote == state.infected_id:      return 0.30
    return -0.20
```

Function is pure — no mutation. `vote_just_resolved` is owned by
`advance_step` exclusively.

Rewrite `group_outcome_reward` docstring to match the existing
per-agent-flat code (no behavioural change):

```python
def group_outcome_reward(state, agent_id):
    """Terminal reward based on group outcome (per-agent flat).

    Healthy team survives (≥1 healthy alive at end):
      +0.40 if THIS agent is alive and not infected
      +0.30 if infected was neutralized AND this agent is not infected

    Infected wins (all healthy dead):
      +0.40 if THIS agent is infected (the winner)
      -0.20 if THIS agent is dead AND not infected (a casualty)
    """
    # Body unchanged — this is a docstring fix only.
```

Update `compose_reward` to drain `pending_damage_reward` after summing
(the value was just consumed by `survival_reward`; symmetric mutation):

```python
def compose_reward(state, agent_id):
    raw = (
        survival_reward(state, agent_id)
        + vote_reward(state, agent_id)
        + group_outcome_reward(state, agent_id)
    )
    # Drain consumed pending damage: it was just added inside survival_reward.
    # Done here (single site) so all callers — env.step's compute path AND
    # _build_observation's terminal/reset path — get consistent semantics.
    state.agents[agent_id].pending_damage_reward = 0.0          # NEW
    clipped = _clip(raw)
    return clipped, raw
```

### `v1/survivecity_env/env.py`

No change required. `compose_reward` now owns the pending-damage drain,
which gives consistent semantics whether `compose_reward` is called from
`step()` (line 122) or from `_build_observation` (line 223-224). The
existing cumulative-reward bookkeeping at line 124 is unchanged.

Note: `vote_just_resolved` is owned by `advance_step` and does NOT need
to be drained anywhere else. It self-clears at the start of the next
round.

### `v1/training/train.py`

Line 174 — read raw reward, not clipped:

```python
# Was:
rewards.append(obs.get("reward", 0.01))
# Now:
rewards.append(obs.get("metadata", {}).get("raw_reward", 0.0))
```

Disable Trainer's default hub push (since it pushes to repo root, not a
timestamped subfolder) and use a custom callback:

```python
# In GRPOConfig: set push_to_hub=False (no Trainer-driven auto-push)
# Add the callback class near the top of train.py:

class TimestampedHubPushCallback(TrainerCallback):
    """Push entire output_dir to noanya/v1-zombiee/<timestamp>/ on every save.

    Replaces hub_strategy='every_save' (which pushes to repo root) with a
    timestamped subfolder so multiple runs accumulate in the same repo
    without overwriting each other.
    """
    def __init__(self, hub_repo_id: str, path_in_repo: str, hf_token: str):
        self.hub_repo_id = hub_repo_id
        self.path_in_repo = path_in_repo
        self.hf_token = hf_token

    def on_save(self, args, state, control, **kw):
        from huggingface_hub import upload_folder
        try:
            upload_folder(
                folder_path=args.output_dir,
                path_in_repo=self.path_in_repo,
                repo_id=self.hub_repo_id, repo_type="model",
                token=self.hf_token,
                commit_message=f"step {state.global_step}",
            )
        except Exception as e:
            # Hub push is best-effort; never raise into the training loop
            logger.warning(f"hub push failed at step {state.global_step}: {e}")

# In main(), pass via callbacks=[TimestampedHubPushCallback(...)] to GRPOTrainer.
# The timestamp is generated once at startup as datetime.utcnow().strftime("%Y-%m-%dT%H%MZ").
```

## Data flow

End-to-end signal trace, post-fix.

### One GRPO step

```
prompt (with [SEED:N]) → model emits 4 completions (num_generations=4)
                                   │
                                   ▼  per completion ↴
┌────────────────────────────────────────────────────────────────────┐
│ reward_fn (train.py)                                                │
│   1. _parse_action(completion) → action dict                        │
│   2. env = SurviveCityEnv(); env.reset(seed=ep_seed)                │
│   3. env.step(action)        ← model's only action (agent 0)        │
│   4. random rollout via env.step(rand_act) until done or 350 steps  │
│   5. return obs.metadata.raw_reward  ← unclipped final-step reward  │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
group-normalize → advantages → PPO/KL loss → LoRA update
                                   │
                                   ▼  on_save ↴
TimestampedHubPushCallback → upload_folder(output_dir → noanya/v1-zombiee/<ts>/)
```

### Inside one `env.step(action)` after the fixes

```
parsed.agent_id  →  apply_agent_action(state, agent_id, action_type, ...)
  1. agent.reset_step_flags()                     (last_action="", damage=0)
  2. agent.last_action_this_step = action_type    ← used by wait penalty
  3. agent.prev_food_dist_this_step = _min_food_dist(row, col)   ← BEFORE move
  4. hunger += 1 (or 1.5 for infected)
  5. if hunger ≥ 15: hp -= 1; pending_damage_reward -= 0.10      ← NEW
  6. apply movement / eat / vote / broadcast / wait
  7. if infected & revealed: _infected_attack
        └→ other.pending_damage_reward -= 0.10                   ← NEW
  8. safehouse healing
  9. agent.cur_food_dist_this_step = _min_food_dist(row, col)    ← AFTER move

if all alive agents acted this round:
  ├─ advance_zombies(state)
  │      └→ collision damage: agent.pending_damage_reward -= 0.10  ← NEW
  └─ advance_step(state)
       step_count += 1
       vote_just_resolved = False                ← cleared at start of every round
       if step_count == 51 and not vote_resolved:
           resolve_vote(state)
           vote_just_resolved = True             ← True for exactly this round
       check_terminal()

compose_reward(state, parsed.agent_id) → (clipped, raw)
  ├─ survival_reward:
  │     +0.005 alive + eat + hunger-penalty + wait-penalty
  │     + pending_damage_reward + forage_delta + death_penalty
  ├─ vote_reward: gated on vote_just_resolved (one-shot)
  ├─ group_outcome_reward: terminal only — per-agent flat
  └─ DRAIN: state.agents[agent_id].pending_damage_reward = 0.0  ← NEW

raw = sum (signed, unclipped)
clipped = max(0.01, min(0.99, raw))      ← OpenEnv contract for obs.reward

return obs:
  obs.reward             = clipped       (OpenEnv-compliant)
  obs.metadata.raw_reward = raw          (training-internal)
```

## Error handling

### Pre-launch failure modes (caught by unit tests)

| ID | Failure | Cause | Detection |
|---|---|---|---|
| FM-1 | `vote_just_resolved` stuck on True | `advance_step` doesn't clear it | Unit test: episode through step 60, assert False after step 52 |
| FM-2 | `pending_damage_reward` double-counted | `compose_reward` drain skipped or `reset_step_flags` clears it | Unit test: set pending=−0.20, call `compose_reward`, assert reward includes −0.20 AND `pending_damage_reward == 0` after; second call has no damage component |
| FM-3 | Forage delta sign flipped | Arithmetic error: `cur − prev` instead of `prev − cur` | Unit test: agent moves toward food, assert delta > 0 |
| FM-4 | Forage shaping fires when stationary | Snapshots set when no movement; should be net-zero | Unit test: agent waits, assert `prev == cur` |
| FM-5 | Wait penalty fires on dead agent | `last_action_this_step` stale | Unit test: kill agent, assert no −0.002 |
| FM-6 | TimestampedHubPushCallback fails | HF Hub 5xx, network blip | Callback wraps in try/except, logs error, never raises |
| FM-7 | `_min_food_dist` ValueError on empty FOOD_CELLS | `min()` of empty iterable | Helper returns sentinel 99 |

### Pre-launch smoke test

`v1/scripts/smoke_test_reward_variance.py` runs reward_fn against 8
different first-actions on the same seed. Asserts `std(rewards) > 0.01`.
Catches the floor-pinning bug ahead of training. Run on host CPU in
~30 seconds. **If it fails, training does not launch.**

### In-flight failure modes (caught via TB metrics)

| ID | Failure | Symptom | Mitigation |
|---|---|---|---|
| IF-1 | `reward_std == 0` at step 1 | Same as broken v2 / extended runs | Kill the run; smoke test should have caught it |
| IF-2 | `kl` spikes >0.1 then collapses | Reward magnitudes too large | Lower lr or raise beta; default lr=1e-5 should be conservative |
| IF-3 | `mean reward` plateaus near 0 | Forage signal dominated by penalties | Continue training; eval at end reveals truth |
| IF-4 | Hub push stalls training | HF Hub timeouts blocking main thread | Acceptable up to 30s/save. If >5min hang, restart with `push_to_hub=False` |

### What this design accepts

- Random rollout still drives the post-model-action episode tail
  (priority #4 deferred). Acceptable for this scope; with the other
  fixes in place, the per-completion reward should still vary enough
  across model first-actions to give GRPO a usable gradient.
- Reward returned to GRPO is the FINAL step's raw_reward only. Per-step
  shaping (forage delta, wait penalty) propagates indirectly via
  hunger/damage/terminal outcome. If learning is weak after step 25,
  accumulating per-step rewards across the rollout is a one-day
  follow-up.

## Testing

Four phases. Each must pass before the next runs.

### Phase 1 — Unit tests (host CPU, ~10s)

New file `v1/tests/test_reward_fixes.py` with 7 tests covering FM-1
through FM-7. Run via `pytest v1/tests/test_reward_fixes.py -v`. All 7
must pass.

Tests cover:

1. `vote_just_resolved` lifecycle (set true on resolve, cleared next round)
2. `vote_reward` returns 0 unless `vote_just_resolved` is True
3. `pending_damage_reward` carries across rounds and is drained by `compose_reward` (second call returns no damage component)
4. Forage shaping rewards closing distance, net-zero on round-trips
5. Wait penalty fires only for alive agents
6. `group_outcome_reward` returns per-agent flat amounts (matches docstring)
7. `_min_food_dist` handles empty FOOD_CELLS without ValueError

### Phase 2 — Reward variance smoke test (host CPU, ~30s)

`v1/scripts/smoke_test_reward_variance.py` — runs `reward_fn` against 8
different first-actions on `[SEED:12345]`, computes `std(rewards)`,
asserts `> 0.01`. Mandatory before launching the 9-hour training.

### Phase 3 — In-training assertions (Kaggle T4, live)

After step 1 logs to TensorBoard, watch for:

- `train/reward_std > 0` (must not be 0)
- `train/loss > 0` (must not be 0)
- `train/grad_norm > 0` (must not be 0)
- `train/kl` between 1e-3 and 1e-2 (healthy range)

If `reward_std == 0` at step 1, kill the run. Smoke test should have
caught this; if it slipped through, do not waste another 8 hours.

### Phase 4 — Eval (post-training, ~30 min on T4)

Run `v1/training/eval.py` against the new LoRA at
`noanya/v1-zombiee/<timestamp>/` with the broadcast-truncate fix
(already in `eval_v1_kaggle_extend.ipynb`).

Targets:

| Metric | Baseline (`noanya/zombiee` step-12, n=10) | Target (n=30) | Pass? |
|---|---|---|---|
| Survival rate | 10% (1/10) | ≥ 20% (≥6/30) | TBD |
| Mean total reward | 0.80 | ≥ 1.0 | TBD |
| Mean episode length | 37.6 | ≥ 50 | TBD |
| Starvation deaths fraction | ~75% | ≤ 40% | TBD |
| Vote accuracy | 0/1 | meaningfully measurable | TBD |

**Pass criterion**: ≥ 3 of 5 metrics improved. Otherwise fall back to
`noanya/zombiee` for the submission.

## Compute budget

12h total Kaggle T4 (15.6 GB).

- Setup (deps install, code patch verify, smoke test): ~30 min
- Training: 9-10h at ~19 min/step × 25-30 steps
- Eval: ~30-60 min for 30 episodes baseline + 30 trained
- Hub uploads: amortised ~30s per save, ~5 saves total = ~3 min
- Buffer: 30-60 min for unexpected issues

## Hyperparameter constants

| Parameter | Value | Rationale |
|---|---|---|
| Wait penalty | −0.002 flat | +0.005 alive bonus minus −0.002 wait = +0.003 net (still positive, but lower than non-wait actions). 100 waits = −0.2 cumulative — meaningful but not crushing. |
| Forage shaping coefficient | 0.005 per cell-delta | Same magnitude as the alive bonus; one-cell movement toward food matches the value of being alive that step. |
| Damage penalty per HP | −0.10 (unchanged from existing) | Consistent with current rubric; only the credit-assignment timing changes. |
| Vote scores | unchanged (±0.30 / ±0.20 / −0.05) | Magnitudes already comparable to cumulative survival once one-shot fix applied. |

## Rollback plan

If unit tests fail or smoke test fails: fix and re-test before launch.

If training run shows `reward_std == 0` at step 1: kill, debug, do not
launch a second 9h run without first re-running the smoke test on the
patched code.

If eval shows ≤ 2 of 5 targets improved: keep `noanya/v1-zombiee/<timestamp>/`
as an experiment record. Submission falls back to the existing
`noanya/zombiee` step-12 LoRA. Report stays unchanged. The `v1new`
branch can either be merged to `master` (docs-only changes that don't
affect runtime) or left as an isolated branch.

If eval shows ≥ 3 of 5 targets improved: update `report/v1/v1.tex`
Table 2 to reference the new LoRA, repoint the figure URLs at
`noanya/v1-zombiee/<timestamp>/eval_results/`, recompile the PDF, merge
`v1new` to `master`.

## References

- User's bug analysis (priorities 1-4 plus 12 design issues), 2026-04-26
- v1 eval log analysis (75% starvation deaths, post-vote constant
  rewards, `wait/move/wait` policy degeneracy)
- Ng, Harada, Russell (1999), "Policy invariance under reward
  transformations: Theory and application to reward shaping" — basis
  for potential-based forage shaping
- TRL 0.15.2 GRPOTrainer source for callback hooks and `every_save`
  default behavior
