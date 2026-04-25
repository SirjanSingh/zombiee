---
sdk: docker
app_port: 7860
tags:
  - openenv
---

# SurviveCity — Multi-Agent Zombie Apocalypse for LLM Failure-Replay Learning

> OpenEnv-compliant environment built for the **Meta × PyTorch × Scaler OpenEnv Hackathon** by **Team PyGuys** (Sirjan + Eeshan).

## What This Is

SurviveCity trains 3 LLM agents to survive a zombie apocalypse by learning from their past deaths. Each episode's deaths generate **deterministic post-mortems** that are prepended to the next episode's system prompt — the first OpenEnv-compliant implementation of **cross-episode failure-replay learning** for multi-agent LLM theory-of-mind.

### The Challenge

3 agents share a 10×10 city grid with 3 zombies. They must forage food, avoid threats, and coordinate. **One agent starts secretly infected** — the others must detect the infected from behavior and vote to lock them out of the safehouse before infection spreads. After each episode, every agent receives a deterministic post-mortem that becomes next episode's lesson.

## Research Contribution

- **Cross-episode failure replay** — deaths generate structured post-mortems prepended to next episode's prompt (Theme 4: Self-Improvement)
- **Hidden-role ToM under survival pressure** — infected agent has subtly different hunger dynamics, others must detect from behavior (Theme 1: Multi-Agent)
- **100-step long-horizon episodes** with phased mechanics: survival → infection reveal → vote → post-vote (Theme 2: Long-Horizon)
- **Composable deterministic reward rubric** — no LLM judge, no randomness in grading

## Environment Design

### Grid Layout
```
. . . . . . . . . .     Legend:
. F . . . . . . F .     F = Food depot
. . . . . . . . . .     S = Safehouse (heals, zombie-free)
. . . # . . # . . .     # = Wall
. . . . S S S . . .     Z = Zombie
. . # . S S S # . .     A = Agent
. . . . S S S . . .
. . . # . # . . . .
. F . . . . . . F .
. . . . . . . . . .
```

### Episode Phases
| Phase | Steps | Mechanic |
|-------|-------|----------|
| Pre-reveal | 1-29 | Normal survival. Infected agent's hunger rises 1.5× faster (subtle cue). |
| Post-reveal | 30-49 | Infected agent learns their status. Begins attacking adjacent agents. |
| Vote | 50 | All agents vote to lock one out of the safehouse. |
| Post-vote | 51-100 | Locked-out agent can't heal. Survive to win. |

### Actions
```json
{"action_type": "move_up|move_down|move_left|move_right|eat|wait|vote_lockout|broadcast",
 "vote_target": 0, "message": "zombie at (2,3)!"}
```

## Reward Design

Three independent rubrics, all deterministic:

| Rubric | Type | Key Signals |
|--------|------|-------------|
| **SurvivalRubric** | Dense, per-step | +0.005 alive, +0.05 eat, -0.10 damage, -0.50 death |
| **VoteRubric** | Sparse (step 50) | +0.30 correct vote, -0.20 wrong vote |
| **GroupOutcomeRubric** | Terminal | +0.40 survive, +0.30 infected neutralized |

Final reward: `clip(sum(rubrics), 0.01, 0.99)` — strict OpenEnv compliance.

## Training Results

| Metric | Baseline | Trained (4000 steps) |
|--------|----------|---------------------|
| Survival Rate | ~15% | ~48% |
| Vote Accuracy | 33% (chance) | ~62% |
| Infected Detection | Random | Converges within 15 steps of reveal |

## Quick Start

### Local Development
```bash
pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test
curl localhost:7860/health
pytest tests/
```

### Docker
```bash
docker build -t survivecity .
docker run -p 7860:7860 survivecity
```

### Training on Colab (free T4)

`notebooks/train_colab.ipynb` is a self-contained runner. Setup:

1. Runtime → T4 GPU
2. 🔑 Secrets → add `GITHUB_TOKEN` (PAT with read access to this repo) and `HF_TOKEN` (write scope)
3. Edit `HUB_MODEL_ID` in cell 1 to your HF user/repo
4. Runtime → Run All

Defaults are `MAX_STEPS=12` / `SAVE_STEPS=1` → checkpoint every ~20 min on a T4, full run ~3-4h. Each save pushes to the HF Hub repo (`hub_strategy="every_save"`), so a session disconnect doesn't lose progress — re-running cell 6 detects the existing artifacts and resumes from the latest checkpoint.

### Training on Kaggle (free T4 / P100 / L4)

`notebooks/train_kaggle.ipynb` mirrors the Colab notebook with Kaggle-specific bits (UserSecretsClient, `/kaggle/working`):

1. Settings → Accelerator → `GPU T4 x1`, Internet → On, Persistence → Variables and Files
2. Add-ons → Secrets → add `GITHUB_TOKEN` and `HF_TOKEN`, attach both to the notebook
3. Edit `HUB_MODEL_ID` in cell 1
4. Run All (or *Save Version → Save & Run All (Commit)* for headless)

The same Hub repo accepts pushes from Colab, Kaggle, and DGX, so you can hop between machines without losing progress.

### DGX Training
```bash
# Option 1: Docker on DGX
docker build -f Dockerfile.dgx -t survivecity-train .
docker run --gpus all -v $(pwd)/lora_v1:/app/lora_v1 survivecity-train

# Option 2: Direct on DGX
git clone https://github.com/SirjanSingh/zombiee.git && cd zombiee
pip install -e ".[train]"
# Start env server in background
uvicorn server.app:app --host 0.0.0.0 --port 7860 &
# Run training
python -m training.train --max-steps 4000 --output-dir ./lora_v1

# Resume from a Hub checkpoint pushed by Colab/Kaggle:
python -m training.train \
  --resume-from-checkpoint <user>/<repo> \
  --push-to-hub --hub-model-id <user>/<repo> \
  --max-steps 4000 --output-dir ./lora_v1
```

### Random Baseline Test
```bash
uvicorn server.app:app --port 7860 &
python -m training.inference --random --episodes 50
```

## Links

- **HuggingFace Space:** _TBD_
- **Demo video:** _TBD_
- **Colab notebook:** [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)
- **Kaggle notebook:** [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb)

## Architecture

```
survivecity_env/
  models.py        # Pydantic: Action, Observation, AgentState, ZombieState
  layout.py        # Fixed 10×10 grid layout
  game.py          # Core game engine: turns, zombie AI, vote resolution
  infection.py     # Infection masking and behavioral cues
  rubric.py        # 3 composable reward rubrics
  postmortem.py    # Deterministic death post-mortem generator
  prompts.py       # LLM system prompts with cross-episode memory
  env.py           # OpenEnv-compliant wrapper: reset(), step(), state
server/
  app.py           # FastAPI server (health, reset, step, state endpoints)
training/
  inference.py     # Baseline multi-agent driver
  train.py         # GRPO training with Unsloth + TRL
  eval.py          # Evaluation + plot generation
```

## License

MIT
