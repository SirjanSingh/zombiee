# 07 — v2 Design & Implementation

> Status (2026-04-25): code complete, smoke-tested locally. **Not yet trained.**
> See `version2.md` (root, gitignored) for the full design rationale and the
> open questions still under discussion. This file documents what's *in the
> codebase* — file-by-file map + how to run.

---

## v2 in one screen

| Aspect | v1 | v2 |
|---|---|---|
| Grid | 10×10 | **15×15** |
| Agents | 3 (A0..A2) | **5** (A0..A4) |
| Starting infected | 1 (random) | **2** (one biter, one saboteur) |
| Bite spread | none | **p=0.35** on adjacency, hash-seeded RNG |
| Reveal step(s) | 30 | biter=25, saboteur=60 |
| Resources | food only | **food + water + medicine** |
| Inventory | none | **3 slots/agent**, items: food / water / medicine |
| Voting | once @ t=50 | **3 rounds** @ t=30, 60, 90 (resolve at T+1) |
| Broadcasts | free | **noise meter** → over threshold gives zombies +1 free step |
| Day/night | none | day 0-24, night 25-49, day 50-74, night 75-99 |
| Zombies | 3 fixed | start 3, **waves at t=25/50/75** (+2/+3/+3, cap 12) |
| Action types | 8 | **14** (added drink, scan, pickup, drop, give, inject) |
| Reward rubrics | 3 | **10** (added 7 — see `survivecity_v2_env/rubric.py`) |
| Reward clip | (0.01, 0.99) | (0.01, 0.99) — same OpenEnv contract |
| Server port | 7860 | **7861** |
| Package name | survivecity 1.0.0 | survivecity-v2 2.0.0 |

The v2 action space is a **strict superset** of v1, so a v1 LoRA loads
zero-shot on v2 and produces parseable (but suboptimal) actions. This is
what enables the planned transfer-learning experiment.

---

## File map (everything lives under `v2/`)

```
v2/
├── pyproject.toml         name="survivecity-v2"; [train] pinned to the DGX-tested
│                          set (torch 2.5.1+cu121, transformers 4.40.2, peft 0.10.0,
│                          trl 0.8.6, datasets 2.19.1, accelerate 0.30.1, bnb >=0.41,
│                          torchao==0.7.0); [unsloth] = unsloth git@cu121-torch250
├── README.md              install / train / eval / simulate quick-start
├── Dockerfile             slim FastAPI server image (env-only, port 7861;
│                          for OpenEnv submission and external evaluators)
├── Dockerfile.dgx         FULL DGX training container — CUDA 12.1.1-devel base,
│                          known-good pin set mirrored from v1/Dockerfile.dgx,
│                          torchao 0.7.0 force-reinstalled to dodge the torch.int1
│                          regression, sanity-checked import chain at build time.
│                          Default CMD runs `training.train` with DGX-friendly
│                          flags (bf16, num_generations=8, LoRA r=32 α=64, --no-4bit).
├── openenv.yaml           OpenEnv submission manifest (port 7861)
├── .gitignore             excludes checkpoints/ adapter_*, eval_results/*.json/*.png, results/transcripts/
├── checkpoints/.gitkeep   local LoRA storage
├── eval_results/.gitkeep  eval JSON + bar/history PNGs
├── results/.gitkeep       simulator transcript output
├── notebooks/             (placeholder for DGX training + 15GB eval launcher)
├── survivecity_v2_env/    THE ENV (10 modules) — see breakdown below
├── server/
│   ├── __init__.py
│   └── app.py             FastAPI on port 7861, single in-process env, /reset /step /state /health
├── tests/
│   ├── __init__.py
│   └── test_v2.py         18 tests: reset, reward bounds, all action types, determinism,
│                          bite RNG distribution, hash01 uniformity, inventory, wave caps,
│                          vote resolution, inject outcomes, day/night phases
└── training/
    ├── __init__.py
    ├── inference.py       parse_action, random_action, make_llm_action_fn
    ├── train.py           DGX 30GB GRPO trainer (bf16/fp16 auto, optional 4-bit, hub push)
    ├── eval.py            15GB eval — runs N episodes, writes JSON + bars + history
    └── simulator.py       rich text-mode visualizer (ANSI colour, per-step grid + state + diff banners)
```

### `survivecity_v2_env/` modules

| File | What it does | Read this when... |
|---|---|---|
| `__init__.py` | Public API: `SurviveCityV2Env`, models | n/a |
| `layout.py` | 15×15 grid constants — F=8, W=4, M=2, S=3×3, walls, agent spawns, zombie spawns, **wave spawn pool** | adding/moving cells |
| `models.py` | Pydantic `SurviveAction` (14 types), `SurviveObservation`, `AgentState` (now with thirst, infection_state, infection_role, bite_at_step, inventory) | extending the action JSON |
| `inventory.py` | 3-slot list bookkeeping: `add_item`, `remove_at`, `remove_first`, `find_first_slot`. Cap = 3. | inventory mechanic edits |
| `spawn.py` | Wave scheduler: `WAVE_SCHEDULE = {25:2, 50:3, 75:3}`, `MAX_ZOMBIES=12`, `pick_wave_spawn_cells` | tuning wave difficulty |
| `infection.py` | **Bite RNG**: BLAKE2b-keyed `_hash01`, `should_bite`, `cue_visible`. Constants: `P_BITE=0.35`, `LATENT_DURATION=15`, `BITER_REVEAL_STEP=25`, `SABOTEUR_REVEAL_STEP=60`, `CUE_FALSE_POSITIVE_RATE=0.30`, `CUE_MISS_RATE=0.30`. Also masking + behavioural cues. | balance tuning |
| `postmortem.py` | Bite-history-aware death summaries; phase tag (day1/night1/day2/night2); 7 mistake categories | extending diagnostic detail |
| `rubric.py` | **10 rubrics** + `compose_reward` clip + `per_rubric_breakdown` (used by simulator) | reward shaping |
| `game.py` | The bulk: `EpisodeState` dataclass, `apply_agent_action` switchboard, action handlers, `advance_zombies` (BFS chase + extra step on noise > threshold), `advance_step` (waves, day/night, noise decay, latent→revealed transitions, vote resolution, resource respawns, terminal check) | most mechanic edits |
| `prompts.py` | `SYSTEM_PROMPT_TEMPLATE` (v2-aware), `build_system_prompt`, `format_observation_description` (filters far-away agents/zombies at night) | LLM prompt iteration |
| `env.py` | `SurviveCityV2Env` wrapper — `reset/step/state` matching OpenEnv contract, builds masked observations, threads metadata (vote_correct, bite_history, rubric_breakdown, last_inject_result, last_scan_result, etc.) | wrapper-layer changes |

---

## Action reference (the full v2 SurviveAction)

```python
SurviveAction(
    agent_id: int,
    action_type: Literal[                        # 14 total
        "move_up","move_down","move_left","move_right",  # v1
        "eat","wait","vote_lockout","broadcast",          # v1
        "drink","scan","pickup","drop","give","inject",   # v2
    ],
    vote_target: Optional[int] = None,           # v1, vote_lockout
    message: Optional[str] = None,               # v1, broadcast (max 40)
    scan_target: Optional[int] = None,           # v2
    inject_target: Optional[int] = None,         # v2 (None == self)
    gift_target: Optional[int] = None,           # v2 (give: receiver)
    item_slot: Optional[int] = None,             # v2 (drop / give / inject: which slot 0..2)
    item_type: Optional[Literal["food","water","medicine"]] = None,  # v2 (pickup hint)
)
```

Lenient validation: extra fields are ignored; unknown action_types fall
through to no-op. This matches v1's behaviour and keeps the JSON parser
permissive (a v1 LoRA's old action shape is always valid).

## Reward reference (10 rubrics)

```
1. survival_reward          (per-step, per-agent, dense)         — same as v1
2. iterated_vote_reward     (one-shot per vote phase resolved)   — sums across t=30,60,90
3. group_outcome_reward     (terminal)                           — adapted for 2 starting infected
4. thirst_reward            (per-step, dense)                    — +0.005 alive, +0.03 drank, -0.05 thirst>=10
5. broadcast_economy_reward (per-broadcast over threshold)       — -0.02 per broadcast above noise threshold
6. night_survival_reward    (per-step in night windows)          — +0.01
7. infection_dodge_reward   (per-step + transition)              — +0.02 healthy, -0.10 latent->revealed
8. medication_reward        (one-shot on inject outcome)         — +0.30 self_cured, +0.40 other_cured, -0.05 wasted
9. hoarding_penalty_reward  (terminal, per unused slot)          — -0.05 each
10. wave_survival_reward    (one-shot at T+1 for T in {25,50,75})— +0.05 if alive
```

All rubrics' contributions sum, then clip into (0.01, 0.99). The clip
preserves OpenEnv compliance even if a single step accidentally accumulates
to ±large.

## Determinism contract

- Episode RNG seeded from `seed` argument to `reset()`.
- Bite outcomes use BLAKE2b on `(episode_seed, step, biter_id, victim_id)` —
  no `random.random()` involvement, so bite outcomes don't shift if some
  unrelated branch advances the episode RNG.
- Cue visibility uses BLAKE2b on `(episode_seed, step, observer_id, target_id)`.
- Scan accuracy uses BLAKE2b on `(episode_seed, step, scanner_id, target_id)`.
- Wave spawn cells use `state.rng.sample(...)` — depends on episode RNG order.

Test: `tests/test_v2.py::test_episode_determinism` runs the same seed twice
and asserts identical (step, reward, n_alive, n_zombies, n_bites) trajectory.

---

## How to use (cheatsheet)

### Train on DGX (30 GB VRAM) — containerised (preferred)

The pin set in `v2/Dockerfile.dgx` is mirrored from `v1/Dockerfile.dgx` (which
the team validated on DGX in April 2026). The CRITICAL pin is
`torchao==0.7.0` — it must be force-reinstalled AFTER transformers, otherwise
`torch.int1` references in newer torchao crash `import transformers`.

```bash
docker build -f v2/Dockerfile.dgx -t survivecity-v2-dgx v2/
docker run --rm --gpus all \
    -e HUGGINGFACE_TOKEN=$HF_TOKEN \
    -v "$(pwd)/v2/checkpoints:/app/checkpoints" \
    -v "$(pwd)/v2/eval_results:/app/eval_results" \
    survivecity-v2-dgx
```

Default CMD saturates the DGX for GRPO: bf16 base (no 4-bit),
num_generations=12, grad_accum=4 (48 generations evaluated per step),
max_completion_length=512, gradient checkpointing on, optim=adamw_torch_fused,
max_steps=100, save_steps=10, save_total_limit=10. Override the CMD to add
`--push-to-hub --hub-model-id <user>/zombiee-v2 --resume-from-checkpoint auto`.

### Train on DGX bare-metal

```bash
cd v2
pip install -e .[train]            # exact pin set (no Unsloth)
pip install -e .[train,unsloth]    # + Unsloth fast kernels (Ampere+ only)
python -m training.train \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --max-steps 100 \
    --save-steps 10 \
    --save-total-limit 10 \
    --output-dir ./checkpoints \
    --num-generations 12 \
    --grad-accum-steps 4 \
    --max-completion-length 512 \
    --lora-r 32 --lora-alpha 64 \
    --max-seq-length 4096 \
    --no-4bit \
    --push-to-hub --hub-model-id <user>/zombiee-v2 \
    --resume-from-checkpoint auto
```

For the v2-warmstart-from-v1 transfer experiment:

```bash
python -m training.train --warmstart-from noanya/zombiee  ...rest as above
```

For full-precision (no 4-bit) on plenty-of-VRAM A100/H100:

```bash
python -m training.train --no-4bit ...
```

### Eval on a separate 15 GB box (T4)

```bash
cd v2
python -m training.eval \
    --lora-path <user>/zombiee-v2 --revision checkpoint-100 \
    --baseline-episodes 30 --trained-episodes 10 \
    --eval-step 100 \
    --output-dir ./eval_results
```

`--lora-path` accepts:
- a local path (e.g. `./checkpoints/checkpoint-100`)
- a Hub repo id (then `--revision` selects a branch / tag like `checkpoint-100`)
- `None` (the script logs a warning and falls back to random; useful for
  pipeline sanity-checks before any LoRA exists).

Output:
- `eval_results/eval_step_0100.json` — full per-episode records + aggregates
- `eval_results/eval_step_0100_bars.png` — 4-panel bar chart
- `eval_results/eval_history.png` — auto-merged trend across all eval_step_*.json

### Simulate one episode (visual playthrough)

```bash
cd v2
python -m training.simulator \
    --lora-path ./checkpoints/checkpoint-100 \
    --seed 42 \
    --max-steps 100 \
    --output ./results/transcripts/sim42.txt
```

Renders, per step:
- Full ASCII grid (with ANSI colour for cell types and agents)
- All agent states (HP, hunger, thirst, inventory, infection masking)
- Zombie list
- Each agent's action + reward delta
- Diff banners: zombie waves, day/night flips, bites, deaths, vote results
- Episode-end summary + cumulative per-agent reward + all post-mortems

Useful for the demo video.

---

## Balance levers (what to tune if v2 is unwinnable)

If the random-policy survival floor is too low (<1 %) or too high (>10 %),
the candidates to adjust are, in order of impact:

1. `infection.P_BITE` — currently 0.35. Drop to 0.25 if too many bites land.
2. `spawn.WAVE_SCHEDULE` — currently `{25:2, 50:3, 75:3}`. Shrink to
   `{25:2, 50:2, 75:2}` to soften late-game.
3. `spawn.MAX_ZOMBIES` — 12 cap. Lower to 9.
4. `MEDICINE_CELLS` count in `layout.py` — currently 2. Bump to 3 to make
   medicine slightly more available.
5. `LATENT_DURATION` (15 steps) — extend if agents can't get to medicine in time.
6. `infection_progression` death timer in `game._check_reveals` — currently
   30 steps post-bite without medicine kills the agent. Loosen to 40 if
   agents are getting railroaded.

**Do not** raise `P_BITE` above 0.45 without re-running the balance sweep —
the bite cap is "1 per biter per step", and >0.45 makes early-episode
multi-bite scenarios near-certain.

---

## Pitfalls / gotchas worth pinning

- **Episode early-termination:** v2 ends as soon as `n_healthy_alive == 0`.
  Random-policy episodes often die at step 10-18 because A1 (biter at safehouse)
  attacks adjacent agents starting at step 25 — wait, the biter only reveals
  at step 25, so why earlier deaths? Random policy with 14 actions (most
  not movement-related) means agents stay near safehouse but lose HP to
  hunger (15 steps to starvation) and thirst (15 steps to dehydration).
- **Vote at step 30:** action is recorded only if `state.step_count == 30`
  exactly, AND the agent's turn is processed during step 30. With early
  deaths, votes may not all land — test setup needs agents to be alive
  through step 30.
- **OpenEnv compliance is contract, not advice:** every observation's
  `reward` field MUST be in (0, 1). If you add a rubric, route it through
  `compose_reward` so the final clip applies. Don't bypass.
- **Hub repo separation:** v1 LoRA in `noanya/zombiee`, v2 LoRA in
  `noanya/zombiee-v2`. Don't mix or the cross-checkpoint `eval_history.png`
  becomes incoherent (different action spaces, different rubric counts).
- **Inventory observability:** an agent sees ONLY its own inventory.
  Don't expose inventory of other agents in the observation dict — it
  would break the social-deduction angle.

---

## What's not in v2 yet (deferred to v3 or later)

- Saboteur-poisoned items (`give` of poisoned food). Mentioned in
  `version2.md`'s Inventory section as a v3 idea.
- RAG-style cross-episode memory (post-mortems are textual but only the
  last 3 are prepended; no retrieval over the full history).
- Multiple maps (currently fixed 15×15 layout).
- Reward engineering (the simplified reward hook from v1 — first model
  action + random rollout — is preserved in v2's train.py for
  apples-to-apples GRPO compute parity).
