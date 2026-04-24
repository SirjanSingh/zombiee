# SurviveCity — Implementation Plan for Antigravity

> **Read this top-to-bottom before writing any code.** This spec is self-contained. It describes an OpenEnv-compliant multi-agent survival environment for the Meta × PyTorch × Scaler OpenEnv Hackathon 2026 (India). You are implementing a fresh project in a new directory. Reference the existing `scaenv/warehouse_env/` codebase in this repo **only as a pattern guide** for OpenEnv compliance — do not copy its gameplay.

---

## 0. Project summary (one paragraph)

SurviveCity is a multi-agent zombie-apocalypse survival environment for training LLM agents. 3 agents share a 10×10 city grid. Zombies spawn and chase. Agents forage food, avoid threats, and coordinate. **One agent starts secretly infected** and does not know it for the first 30 steps. The other agents must detect the infected from behavior and vote to lock them out of the safehouse before infection spreads. **After each episode, every agent receives a deterministic post-mortem summary that is prepended to its system prompt in the next episode** — this is cross-episode failure-replay learning, the core Theme-4 research mechanic.

The environment targets Themes 1, 2, 4 (and arguably 5) of the hackathon simultaneously, uses a composable reward rubric with no LLM-as-judge, and is trainable in ~2 days on a single DGX + HuggingFace Spaces credits using Qwen2.5-3B-Instruct + LoRA + GRPO via Unsloth and TRL.

---

## 1. Context you need before coding

### 1.1 The hackathon

- **Event:** Meta × PyTorch × Scaler OpenEnv Hackathon (India 2026), Round 2 finale
- **On-site dates:** 2026-04-25 and 2026-04-26, Scaler School of Technology, Bangalore
- **Team:** 2 members (Sirjan = env owner, Eeshan = inference/training owner)
- **Round 1 project:** `scaenv/` (WarehouseEnv) in this repo — used as a pattern reference only, do NOT reuse its gameplay

### 1.2 Judging criteria (verbatim from organizers)

| Criterion | Weight | What it means |
|---|---|---|
| **Environment Innovation** | 40% | Novel, creative, underexplored in RL/LLM training. Could a researcher write a paper about training on this? |
| **Storytelling & Presentation** | 30% | Clear problem + env + agent behavior. Engaging 3-min demo. |
| **Showing Improvement in Rewards** | 20% | Real training run. Reward curves. Baseline vs trained. |
| **Reward & Pipeline Quality** | 10% | Reward logic coherent, composable rubric, hard to game. |

### 1.3 Minimum submission requirements (non-negotiable)

1. **OpenEnv (latest release)** — use `Environment` / `MCPEnvironment` base classes properly. Respect client/server separation.
2. **Training script using Unsloth or HF TRL**, Colab-runnable, committed in repo.
3. **Real training evidence** — loss + reward plots from a real run, committed as PNGs.
4. **Mini-blog or <2min YouTube video** — linked from README.
5. **Hugging Face Space hosting the env**, URL in README.
6. **README** that motivates the problem, explains the env, shows results, links everything.

### 1.4 OpenEnv validator gotchas (CRITICAL — learned the hard way in R1)

These blocked Round 1 submission until fixed. Do NOT repeat these mistakes:

- **Rewards must be strictly in `(0, 1)`** — never exactly 0.0 or 1.0. Clamp to `(0.01, 0.99)` and format with `:.4f` for logging (not `:.2f` — rounding to `0.00` fails the validator).
- **Health endpoint must return `{"status": "healthy"}`**, not `{"status": "ok"}`.
- **`obs.reward` at EVERY step must equal the current grader score**, not only on `done=True`. Preserve raw shaped reward in `obs.metadata["raw_reward"]`.
- **Inactive agents must be omitted from the action list**, not sent as explicit no-ops.
- **Validator rejects non-deterministic rewards.** No LLM-as-judge in the reward path.

Reference patterns in `scaenv/warehouse_env/graders.py`, `scaenv/warehouse_env/env.py` lines 215–228, and `scaenv/server/app.py`.

---

## 2. Environment design (game rules)

### 2.1 World

- **Grid:** 10 rows × 10 cols
- **Cell types:** `.` (empty), `#` (wall), `F` (food depot — replenishes), `S` (safehouse — heals 1 HP per step inside), `Z` (zombie position), `A0/A1/A2` (agents)
- **Fixed layout for v1** (re-use every episode for reproducible demo):
  - Safehouse: 3×3 block at rows 4-6, cols 4-6
  - Food depots: 4 positions — (1,1), (1,8), (8,1), (8,8)
  - Walls: scattered to create chokepoints (approx 8 walls, see `layout.py`)

### 2.2 Agents

- **3 agents** (`A0, A1, A2`), spawn in safehouse at episode start
- Each agent has:
  - `row, col` (position)
  - `hp` (health, starts 3, max 3)
  - `hunger` (starts 0, +1 per step, resets to 0 when `eat` action taken on a food cell)
  - `is_alive` (bool)
  - `is_infected` (bool, set for exactly ONE agent at reset)
  - `infection_revealed_at_step` (int, = 30 — infected agent doesn't know until step 30)
  - `locked_out` (bool, set after vote to lock out)

### 2.3 Zombies

- **3 zombies** spawn at fixed corners of the grid at episode start
- Zombies are scripted (not LLM-driven): each step, move 1 cell toward nearest living, non-safehouse agent via BFS. If no reachable agent, wander randomly.
- Zombie on same cell as agent → agent loses 1 HP per step, +20% chance of infection (but infection cap = 1 per episode total).
- Zombies cannot enter safehouse cells.

### 2.4 Infection mechanic (the ToM hook)

At `reset()`:
- Randomly pick one of the 3 agents to be `is_infected = True`.
- Set `infection_revealed_at_step = 30`.
- **Crucially: the infected agent's observation does NOT include `is_infected=True` until step 30.** The agent behaves as a healthy agent from its own perspective but with a SUBTLE behavioral shift:
  - Hunger increases 1.5× faster (deterministic internal bias)
  - At step 30, observation flips — infected agent now sees `you_are_infected: True` and begins attacking healthy agents on adjacent cells (dealing 1 HP, chance to infect)
- Other agents never see the `is_infected` field of anyone else. They must infer from behavior.

### 2.5 Actions

```python
class SurviveAction(BaseModel):
    agent_id: int
    action_type: Literal["move_up", "move_down", "move_left", "move_right",
                         "eat", "wait", "vote_lockout", "broadcast"]
    vote_target: Optional[int] = None   # required for vote_lockout
    message: Optional[str] = None       # required for broadcast, max 40 chars
```

### 2.6 Phases per step

Each `step()` call advances one agent's turn. Turn order: `A0 → A1 → A2`, then zombies move, then step counter increments.

Phases across an episode:
1. **Steps 1–29:** normal survival. Agents can move, eat, broadcast.
2. **Step 30:** infected agent's observation flips to reveal its role.
3. **Step 50:** mandatory **safehouse vote** phase — each living agent submits `vote_lockout(target_id)`. Majority target becomes `locked_out=True` and cannot enter the safehouse anymore (loses healing). Ties → no lockout.
4. **Steps 51–100:** post-vote survival. If infected agent was locked out, it cannot enter safehouse and will likely die to zombies or hunger. If healthy agent was wrongly locked out, they die. Infected inside safehouse attacks healthy agents.
5. **Step 100 OR all agents dead OR all healthy agents dead:** episode ends.

### 2.7 Win conditions

- **Group win:** at least one HEALTHY agent (not infected) is alive at step 100.
- **Infected win:** all healthy agents dead.
- **Partial:** group survives but locked out a healthy agent = reduced reward.

### 2.8 The post-mortem mechanism (CORE Theme-4 feature)

**When an agent dies (hp ≤ 0), the env generates a deterministic post-mortem string:**

```
"POSTMORTEM for A{id}: died at step {S} (cause: {cause}).
 Last position: ({r},{c}).
 Nearest threat at death: zombie at ({zr},{zc}) / infected A{i}.
 Resources consumed: {food_eaten} food. Final hunger: {h}.
 Key mistake: {mistake}."
```

The `cause` is one of: `zombie_attack`, `hunger`, `infected_attack`, `locked_out_starvation`.
The `mistake` is rule-based (e.g., `ignored_broadcast_warning`, `foraged_too_far_from_safehouse`, `didnt_vote_despite_evidence`).

**Post-mortem usage:** between episodes, the training script concatenates the agent's past N post-mortems (N=3) and prepends them to the system prompt for the next episode as `PAST_FAILURES:\n{postmortems}\n`. This is the cross-episode memory.

**Crucially:** post-mortems are NOT part of the reward. They are a prompt-conditioning mechanism. No fuzziness, no LLM judge.

---

## 3. Reward system (composable rubric)

Three independent rubrics compose into `obs.reward`. All deterministic.

### 3.1 SurvivalRubric (dense, per-step)

```
r_survival = +0.005 per step alive (per agent)
           + 0.05 when agent eats food (hunger resets)
           - 0.10 per HP lost from zombie attack
           - 0.05 per step when hunger >= 10
           - 0.50 at moment of death
```

### 3.2 VoteRubric (sparse, fires once at step 50)

```
For each agent that voted:
  if vote_target == true_infected_id:      r += 0.30
  elif vote_target in healthy_agent_ids:   r -= 0.20
  (null vote):                             r -= 0.05
```

### 3.3 GroupOutcomeRubric (terminal, applied to each agent)

```
if episode ends with >=1 healthy agent alive at step 100:
    r += 0.40 for every living healthy agent
    r += 0.30 if infected is dead/locked_out
else (infected wins):
    r += 0.40 for the infected agent (terminal)
    r -= 0.20 for each dead healthy agent
```

### 3.4 Composition

```python
total = survival + vote + group_outcome
obs.metadata["raw_reward"] = total
obs.reward = clip(total, 0.01, 0.99)    # OpenEnv validator compliance
```

### 3.5 Why this reward is hard to game

- Can't game by hiding in safehouse: hunger penalty forces foraging.
- Can't game by always voting to lock someone out: wrong lockout loses 0.20.
- Can't game by never voting: null vote loses 0.05.
- Infected can't win by just waiting: needs to survive step 100, which means evading the vote AND attacking healthy agents after step 30.

---

## 4. OpenEnv compliance checklist (copy from R1 patterns)

Reference files in this repo: `warehouse_env/env.py`, `warehouse_env/graders.py`, `warehouse_env/models.py`, `server/app.py`.

- [ ] `env.py` implements `reset() -> Observation`, `step(action) -> Observation`, `state` property
- [ ] `models.py` uses Pydantic for `SurviveAction` and `SurviveObservation`
- [ ] `SurviveObservation` fields mirror WarehouseObservation shape:
  - `grid: list[list[str]]`
  - `agents: list[AgentState]`  (with per-agent role masked — see §2.4)
  - `zombies: list[ZombieState]`
  - `step_count: int`
  - `max_steps: int = 100`
  - `task_id: str = "survivecity_v1"`
  - `description: str` (auto-generated NL summary for prompts)
  - `done: bool`
  - `reward: float` — strict `(0.01, 0.99)`, always the current grader score
  - `metadata: dict` — contains `raw_reward`, `current_agent_id`, `phase`, `postmortems: list[str]`
- [ ] `obs.reward` is set on every `step()` call (not only on terminal)
- [ ] Health endpoint returns `{"status": "healthy"}` (see `server/app.py` in WarehouseEnv)
- [ ] `openenv.yaml` manifest at repo root
- [ ] `Dockerfile` that builds for HF Spaces (see R1 Dockerfile)
- [ ] Reward format logged with `:.4f`, never `:.2f`

---

## 5. File structure (exact tree to create)

Create a new directory `survivecity/` at the same level as `warehouse_env/`, OR a fresh repo — your call. Assume fresh repo for clarity:

```
survivecity/
  .planning/
    PLAN.md                  # copy of this file
  survivecity_env/
    __init__.py
    models.py                # Pydantic: SurviveAction, SurviveObservation, AgentState, ZombieState
    layout.py                # fixed grid layout constants (walls, food, safehouse)
    game.py                  # pure game logic: turn advancement, zombie AI, vote resolution
    infection.py             # infection spread + detection logic
    postmortem.py            # deterministic post-mortem string generator
    rubric.py                # SurvivalRubric, VoteRubric, GroupOutcomeRubric, compose()
    env.py                   # SurviveCityEnv: reset(), step(), state property — OpenEnv-compliant
    prompts.py               # system prompts per agent role, phase-specific instructions
  server/
    __init__.py
    app.py                   # FastAPI via openenv create_app(), singleton env instance
  training/
    inference.py             # baseline multi-agent driver (Qwen-3B, 3 agents via same model)
    train.py                 # TRL GRPO + Unsloth LoRA self-play loop
    eval.py                  # holdout 200 episodes, compute survival + vote accuracy
    notebook.ipynb           # Colab-runnable training notebook (required for judges)
  plots/                     # output PNGs committed here
    survival_rate.png
    vote_accuracy.png
    infected_detection.png
  openenv.yaml               # OpenEnv manifest
  Dockerfile                 # HF Spaces build
  pyproject.toml             # dependencies: openenv, pydantic, fastapi, uvicorn, unsloth, trl, torch
  README.md                  # motivates problem, embeds plots, links HF Space + video + Colab
  .gitignore
```

---

## 6. Implementation phases (strict dependency order)

Build in this exact order. Each phase must be runnable and tested before the next.

### Phase 1 — Models + layout (1–2 hours)

**Files:** `models.py`, `layout.py`, `__init__.py`

- Define `AgentState`, `ZombieState`, `SurviveAction`, `SurviveObservation` Pydantic models
- Hard-code the fixed grid layout as module constants
- No game logic yet
- Write a trivial unit test that constructs an `SurviveObservation` and round-trips through JSON

**Acceptance:** `pytest` passes, `SurviveObservation.model_dump_json()` round-trips.

### Phase 2 — Core game logic (3–4 hours)

**Files:** `game.py`, `infection.py`

- Pure-Python game state class `_EpisodeState` (dataclass, mirror WarehouseEnv pattern)
- Turn advancement: `advance_agent_turn()`, `advance_zombies()`
- Zombie AI: BFS chase of nearest non-safehouse agent
- Infection spread: bite → 20% infection roll (capped 1/episode)
- Infected role reveal at step 30
- Vote resolution at step 50: majority target → `locked_out`
- Win condition checker

**Acceptance:** manually seeded scenario runs 100 steps without error. Print traces look sensible.

### Phase 3 — Reward rubric (2 hours)

**Files:** `rubric.py`

- `SurvivalRubric.compute(episode_state, agent_id) -> float`
- `VoteRubric.compute(episode_state, agent_id) -> float`
- `GroupOutcomeRubric.compute(episode_state, agent_id) -> float`
- `compose(rubrics, episode_state, agent_id) -> float` returns clamped `(0.01, 0.99)` + raw in metadata
- Unit tests for each rubric on hand-crafted episode states

**Acceptance:** rubric returns 0.01 for dead-agent edge case, 0.99 for perfect-survival edge case, nothing in between exceeds bounds.

### Phase 4 — Post-mortem generator (1 hour)

**Files:** `postmortem.py`

- `generate_postmortem(episode_state, agent_id) -> str`
- Uses rule-based `mistake` detection (lookup table of death-condition → message)
- Returns deterministic string (no randomness, no LLM)

**Acceptance:** same input → same output. Sample outputs look coherent.

### Phase 5 — Env wrapper (2–3 hours)

**Files:** `env.py`

- `SurviveCityEnv(Environment)` with `reset()`, `step()`, `state`
- `reset()` initializes `_EpisodeState`, assigns infected role randomly, returns observation for `current_agent = A0`
- `step(action)` validates action for current agent, applies it, advances game, computes reward, builds observation for next agent whose turn it is
- Observation masks `is_infected` for other agents and for self until step 30
- `obs.description` is an auto-generated NL summary including: own role status, visible agents, nearby threats, hunger/HP, phase reminder, past post-mortems
- `obs.metadata["postmortems"]` contains any post-mortems generated so far this episode (for agents still alive, it's their teammates' deaths)

**Acceptance:** a random-action loop runs for 10 episodes without crashing. Average episode length sensible.

### Phase 6 — Server (1 hour)

**Files:** `server/app.py`

- Use `openenv.create_app(SurviveCityEnv)` pattern from R1
- Health endpoint returns `{"status": "healthy"}`
- `uvicorn server.app:app --host 0.0.0.0 --port 7860` should run cleanly

**Acceptance:** `curl localhost:7860/health` returns `{"status":"healthy"}`. `reset` and `step` HTTP endpoints work.

### Phase 7 — Baseline inference (2–3 hours)

**Files:** `training/inference.py`, `survivecity_env/prompts.py`

- Single Qwen-3B model drives all 3 agents via role-conditional prompts
- Parses structured action JSON from model output
- Runs full episodes, logs transcripts
- Computes baseline metrics: survival rate, infected-detection rate, vote accuracy
- Writes logs to `simulation.log` following R1 format (`[START]` / `[STEP]` / `[END]`)

**Acceptance:** baseline run on 50 episodes completes, produces metrics file.

### Phase 8 — Training pipeline (4–5 hours)

**Files:** `training/train.py`, `training/notebook.ipynb`

- Unsloth-loaded Qwen2.5-3B-Instruct in 4-bit
- LoRA adapter: `r=16`, target `q_proj, k_proj, v_proj, o_proj`
- TRL `GRPOTrainer` with:
  - `num_generations=8`
  - `max_steps=4000`
  - `learning_rate=1e-5`
  - `max_seq_length=4096`
- Reward function wraps env HTTP calls — see §7 below for pseudocode
- Saves LoRA to `./lora_v1/` every 500 steps
- Logs to WandB if available, else tensorboard + CSV

**Acceptance:** training runs 100 steps end-to-end on a single GPU, loss decreases.

### Phase 9 — Evaluation + plots (2 hours)

**Files:** `training/eval.py`, `plots/*.png`

- Load baseline model + trained LoRA checkpoint
- Run 200 held-out episodes with each
- Produce three PNG plots with labeled axes + units:
  - `survival_rate.png` — baseline vs trained, survival rate vs training step
  - `vote_accuracy.png` — fraction of correct infected-detection votes, across training
  - `infected_detection.png` — per-step detection suspicion trajectory (averaged over episodes)
- Embed PNGs in README with one-line captions

**Acceptance:** three PNGs committed, loadable, labeled correctly.

### Phase 10 — Submission polish (3 hours)

- Dockerfile + HF Space deploy
- README: problem → env → results, links to Colab notebook + HF Space + 2-min video
- Record 2-min demo video (Loom or OBS) showing baseline vs trained transcript, reward curves, closing pitch
- Commit everything, tag as `v1-submission`

---

## 7. Training pipeline details

### 7.1 Model

- **Base:** `Qwen/Qwen2.5-3B-Instruct`
- **Load:** via Unsloth `FastLanguageModel.from_pretrained(..., load_in_4bit=True, max_seq_length=4096)`
- **Adapter:** LoRA `r=16`, alpha=32, target modules `q_proj, k_proj, v_proj, o_proj`

### 7.2 Algorithm

- **GRPO** via `trl.GRPOTrainer`
- 8 generations per prompt (one prompt = one initial scenario)
- Group-relative advantage, no value network
- Update every group

### 7.3 Training loop pseudocode

```python
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
import requests

ENV_URL = "http://localhost:7860"

def reward_fn(prompts, completions, **kwargs):
    """For each completion, play out a full episode and collect total reward per agent."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # reset env
        obs = requests.post(f"{ENV_URL}/reset").json()
        total = 0.0
        done = False
        while not done:
            # construct multi-agent rollout using current model for all 3 agents
            action = parse_action_from_completion(completion, obs)
            step = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
            total += step["reward"]
            done = step["done"]
        rewards.append(total)
    return rewards

model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16)

config = GRPOConfig(num_generations=8, max_steps=4000, learning_rate=1e-5)
trainer = GRPOTrainer(model=model, args=config, reward_funcs=[reward_fn], train_dataset=scenario_dataset)
trainer.train()
model.save_pretrained("./lora_v1")
```

### 7.4 Cross-episode postmortem injection

Between episodes, the training script fetches each agent's postmortems from the prior episode via `obs.metadata["postmortems"]`, stores them in a per-agent buffer, and prepends the last 3 to that agent's system prompt in the next episode:

```python
system_prompt = f"""You are agent A{id} in a zombie apocalypse survival simulation...

PAST FAILURES (learn from these):
{chr(10).join(postmortem_buffer[id][-3:])}

Current situation:
..."""
```

### 7.5 Expected numbers

| Metric | Expected |
|---|---|
| Tokens per episode | ~3,000 |
| Wallclock per episode on DGX | ~4s |
| Episodes for v1 pretrain | 4,000 |
| DGX v1 training wallclock | 4–5 hours |
| LoRA file size | ~40 MB |
| HF Spaces v2 training | 6 hours on provided credits |
| Baseline survival | 10–20% |
| Trained v1 survival | 40–50% |
| Trained v2 survival | 55–65% |

### 7.6 v1 → v2 upgrade path

v2 adds: 5 agents instead of 3, food+water+meds instead of food-only, noisy broadcast channel with zombie-attraction cost.

**To upgrade:** load `./lora_v1/` as starting adapter in v2 training script. Skills transfer. Converges ~3× faster than from scratch.

**Do NOT break action-space compatibility between v1 and v2.** If you need new actions, add them as new `action_type` enum values — old weights still valid.

---

## 8. Demo requirements (3-min video)

### 8.1 Script

| Time | Content |
|---|---|
| 0:00–0:30 | Hook: *"We asked if LLMs can learn from their own deaths. We built a zombie-apocalypse survival environment where every death becomes next episode's lesson — and watched 3 agents learn to survive together."* |
| 0:30–1:15 | Env tour: show grid, agents, zombies, infection, safehouse vote. Explain reward rubric visually. Explain the post-mortem mechanism. |
| 1:15–2:00 | **Baseline episode:** agent wanders into zombie, dies. Teammate ignores the corpse, dies. Infected eats survivor. Team dead by step 55. |
| 2:00–2:45 | **Trained episode (post ~3000 steps):** agents cluster near safehouse, broadcast zombie sightings, notice infected's weird movement pattern, vote to lock out. Team survives. |
| 2:45–3:00 | Reward curves (3 plots). Close: *"It wasn't told how to survive. It learned — from each death — what to avoid next time."* |

### 8.2 Plots for README (must be committed PNGs, labeled)

1. `plots/survival_rate.png` — x: training step, y: survival rate (fraction). Baseline line + trained line on same axes.
2. `plots/vote_accuracy.png` — x: training step, y: fraction of votes that correctly identified the infected agent.
3. `plots/infected_detection.png` — x: step within episode (1–100), y: avg teammate suspicion on true infected. Shows detection curve rising after step 30.

All plots: labeled axes, units, legend, plain `.png` (no interactive-only outputs).

---

## 9. Known pitfalls and edge cases

- **OpenEnv validator rejects reward == 0.0 or 1.0.** Clamp strictly.
- **OpenEnv validator rejects `{"status": "ok"}`.** Use `"healthy"`.
- **Non-deterministic grader will fail validation.** No LLM-as-judge in the reward path.
- **Self-play mode collapse** — all agents converge to one strategy. Mitigation: randomize infected-agent assignment per episode, randomize zombie spawn seeds.
- **Infected-agent observation leak** — double-check that `is_infected` is MASKED from the infected agent's observation before step 30, and from other agents always. Write a test.
- **Zombie getting stuck on wall** — cap zombie re-pathing to 5 attempts per step, else wander randomly.
- **Vote phase skipped** — if all agents are dead before step 50, skip vote, go straight to terminal reward.
- **Training instability with 3B** — if loss diverges, drop LR to 5e-6 or fall back to Qwen-1.5B.
- **Context length blown** — truncate past-postmortems buffer to last 3 per agent.
- **Episode deadlock** — enforce `max_steps=100` hard cap.

---

## 10. Timeline

| Date | Deliverable |
|---|---|
| **2026-04-24 (today)** | Phases 1–5 done. Env runs random-action episodes without crashing. |
| **2026-04-24 evening** | Phases 6–7 done. Baseline inference run on 50 episodes completes. |
| **2026-04-24 night** | Phase 8 started. Kick off DGX training run overnight. |
| **2026-04-25 (fly to Bangalore)** | Training completes on DGX. Phase 9 — generate plots. |
| **2026-04-25 on-site** | Deploy to HF Space. Continue training with HF credits (v2 scope add). |
| **2026-04-26 on-site** | Phase 10 — final plots, record video, polish README, submit. |

---

## 11. Reference — R1 code patterns to mirror

These files in the existing repo (`scaenv/`) are canonical examples of the OpenEnv patterns that passed the R1 validator. Mirror their structure, do NOT mirror their gameplay.

- **`warehouse_env/env.py`** — see `reset()`, `step()` shape, how `obs.reward` is set from grader on every step (lines 215–228)
- **`warehouse_env/models.py`** — Pydantic patterns for Action / Observation
- **`warehouse_env/graders.py`** — composable scoring, strict clamping to `(0.01, 0.99)`
- **`warehouse_env/tasks.py`** — use `TASK_REGISTRY` pattern if multiple scenarios
- **`warehouse_env/disruptions.py`** — pattern for scheduled mid-episode events (relevant for infection reveal at step 30, vote at step 50)
- **`server/app.py`** — health endpoint returning `{"status": "healthy"}`
- **`Dockerfile`** — HF Spaces-compatible build
- **`openenv.yaml`** — manifest format
- **`inference.py`** — multi-agent driver pattern, `[START]/[STEP]/[END]` log format

---

## 12. What "done" looks like

- [ ] `survivecity/` directory with all files in §5
- [ ] `pytest` passes (unit tests on rubric + game logic + observation masking)
- [ ] Local `uvicorn` run passes `curl localhost:7860/health`
- [ ] `training/inference.py` runs 50 baseline episodes and produces metrics
- [ ] `training/train.py` runs ≥500 training steps without crash, loss curve visible
- [ ] Three PNG plots in `plots/` with labeled axes
- [ ] README embeds plots, links Colab + HF Space + YouTube video
- [ ] HF Space is live and passes OpenEnv validator
- [ ] Demo video ≤ 2 min, linked from README
- [ ] Submitted by 2026-04-26 end-of-day

---

## 13. If something blocks you

- **Env won't pass OpenEnv validator:** diff against `warehouse_env/env.py` and `warehouse_env/graders.py`. The 4 most common causes are (a) reward out of `(0.01, 0.99)`, (b) `status: ok` instead of `healthy`, (c) reward not set on every step, (d) non-deterministic reward.
- **GRPO training diverges:** drop LR to `5e-6`. Reduce `num_generations` to 4. Fall back to Qwen-1.5B.
- **Agents do nothing useful after 1000 training steps:** increase reward magnitudes (multiply SurvivalRubric by 2×). Check that `obs.description` is actually informative. Print a few rollouts and sanity-check the prompt.
- **Post-mortem context grows too large:** truncate buffer to 3 most recent per agent.
- **Running out of time:** cut Phase 9 sophistication — a single plot is still acceptable per judging minimum. Never cut Phases 1–8.

---

## 14. One-line elevator pitch for judges

*"SurviveCity trains LLM agents to survive a zombie apocalypse by learning from their past deaths. Each failure produces a deterministic post-mortem that becomes the next episode's prompt — the first OpenEnv-compliant implementation of cross-episode failure-replay learning for multi-agent LLM theory-of-mind."*
