# SurviveCity — Antigravity Context Package

Load these files in order. Each is self-contained but `00_PLAN.md` is the authoritative spec.

| # | File | When to load |
|---|---|---|
| 00 | `00_PLAN.md` | **Always first.** Full spec, game rules, phases, acceptance criteria. |
| 01 | `01_OPENENV_PATTERNS.md` | Before touching `env.py` / `models.py` / `server/app.py`. The R1 validator traps. |
| 02 | `02_REWARD_DESIGN.md` | Before writing `rubric.py`. Exact formulas + gaming-resistance analysis. |
| 03 | `03_TRAINING_SETUP.md` | Before writing `training/train.py`. Unsloth + TRL GRPO recipe. |
| 04 | `04_DEMO_AND_SUBMISSION.md` | Before Phase 10. What judges need to see. |
| 05 | `05_TRAINING_ISSUES_AND_FIXES.md` | **Critical.** Bugs found in train.py reward_fn, fixes, and better ideas for training. |

## Hackathon context (TL;DR)

- **Event:** Meta × PyTorch × Scaler OpenEnv Hackathon R2, on-site 2026-04-25/26, Bangalore.
- **Team:** 2 people (Sirjan + Eeshan).
- **Judging:** 40% innovation, 30% storytelling, 20% training curve, 10% pipeline quality.
- **Target themes:** 1 (multi-agent), 2 (long-horizon), 4 (self-improvement). Hit all three.
- **R1 project (for patterns only):** `warehouse_env/` in this repo — do not reuse its gameplay.

## Project one-liner

SurviveCity trains 3 LLM agents to survive a zombie apocalypse by learning from their past deaths. Each episode's deaths generate deterministic post-mortems that are prepended to the next episode's system prompt. One agent is secretly infected at episode start — others must detect and vote to lock them out. First OpenEnv-compliant implementation of cross-episode failure-replay learning for multi-agent LLM theory-of-mind.
