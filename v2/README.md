# SurviveCity v2

5-agent multi-resource zombie survival env with **bite transmission**,
**spawn waves**, **iterated voting (3 rounds)**, **broadcast economy**,
**day/night cycle**, and **per-agent inventory**. Action space is a strict
superset of v1, so a v1 LoRA loads zero-shot.

## Layout

```
v2/
├── survivecity_v2_env/   # env code (10 modules)
├── training/             # train.py (DGX 30GB), eval.py (15GB), simulator.py
├── server/               # FastAPI app on port 7861
├── tests/                # pytest
├── checkpoints/          # local LoRA checkpoints (gitignored)
├── eval_results/         # eval_step_NNNN.json + plots (gitignored)
└── results/              # simulator transcripts (gitignored)
```

## Install

From the repo root:

```bash
pip install -e v2/[train]            # for DGX training
pip install -e v2/                   # env + server only (15GB eval host)
pip install -e v2/[train,unsloth]    # if you want Unsloth fast kernels (Ampere+ only)
```

## Train on DGX (30 GB VRAM)

```bash
cd v2
python -m training.train \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --max-steps 200 \
    --save-steps 25 \
    --output-dir ./checkpoints \
    --num-generations 8 \
    --lora-r 32 \
    --lora-alpha 64 \
    --max-seq-length 4096 \
    --push-to-hub --hub-model-id <user>/zombiee-v2 \
    --resume-from-checkpoint auto
```

Checkpoints land in `checkpoints/checkpoint-{N}/` and are pushed to the Hub
after every save (so a 15GB box can pull and eval them mid-training).

## Eval on a separate 15 GB box (T4)

```bash
cd v2
python -m training.eval \
    --lora-path <user>/zombiee-v2 \
    --baseline-episodes 30 \
    --trained-episodes 10 \
    --eval-step 50 \
    --output-dir ./eval_results
```

The `--lora-path` accepts either a local checkpoint directory or a Hub repo
id (it will `snapshot_download` if the latter). Output goes to
`eval_results/eval_step_<N>.json` plus `eval_step_<N>_bars.png` (per-metric
bar chart) and an `eval_history.png` (cross-checkpoint trend, auto-updates
across runs).

To eval a specific in-progress checkpoint:

```bash
python -m training.eval --lora-path ./checkpoints/checkpoint-100 --eval-step 100
```

## Simulate one episode (rich text visualizer)

```bash
cd v2
python -m training.simulator \
    --lora-path ./checkpoints/checkpoint-100 \
    --seed 42 \
    --max-steps 100 \
    --output ./results/transcripts/sim_seed42.txt
```

Renders the full grid, all agent states, zombie positions, actions, rewards,
broadcasts, and phase changes (waves, votes, day/night, infection reveals)
each step. Use `--policy random` to compare against the baseline.

## Run server (OpenEnv compliance check)

```bash
cd v2
uvicorn server.app:app --host 0.0.0.0 --port 7861
# in another terminal
curl -s -X POST http://localhost:7861/reset -H 'Content-Type: application/json' -d '{"seed":42}' | head -c 200
```

## Run tests

```bash
cd v2
pytest -q
```

## What changed vs v1

| Aspect | v1 | v2 |
|---|---|---|
| Grid | 10×10 | **15×15** |
| Agents | 3 | **5** (A0..A4) |
| Infected | 1 (static) | **2** (biter + saboteur), both randomly assigned |
| Bite spread | none | **p=0.35** on adjacency (deterministic seeded RNG) |
| Resources | food only | **food + water + medicine** |
| Inventory | none (eat-on-pickup) | **3 slots/agent** |
| Voting | once (t=50) | **3 rounds** (t=30, 60, 90) |
| Broadcast | free | **noise meter** → zombies +1 step toward agents over threshold |
| Day/night | none | **day/night cycle** (t=0-24 day, 25-49 night, 50-74 day, 75-99 night) |
| Zombies | 3 fixed | start 3 + **waves at t=25/50/75** (+2/+3/+3, cap 12) |
| Action types | 8 | **14** (added drink, scan, pickup, drop, give, inject) |
| Reward rubrics | 3 | **10** (added 7 new rubrics, all clip into (0.01, 0.99)) |

A v1 LoRA loaded onto v2 will never emit the new action types but still
produces parseable v2 actions — so zero-shot transfer is valid (just
suboptimal, which is the whole point of the transfer-evaluation experiment).
