# SurviveCity — Planning / Context Package

Load files in order. `00_PLAN.md` is the authoritative spec for v1.
Files 06-08 cover what actually shipped + the v2 follow-on.

| # | File | When to load |
|---|---|---|
| 00 | `00_PLAN.md` | **Always first.** Full v1 spec, game rules, phases, acceptance criteria. |
| 01 | `01_OPENENV_PATTERNS.md` | Before touching env / models / server. The R1 validator traps. |
| 02 | `02_REWARD_DESIGN.md` | Before changing rubric.py. Exact formulas + gaming-resistance analysis. |
| 03 | `03_TRAINING_SETUP.md` | Before changing training/train.py. Unsloth + TRL GRPO recipe. |
| 04 | `04_DEMO_AND_SUBMISSION.md` | Before submitting. What judges need to see. |
| 05 | `05_TRAINING_ISSUES_AND_FIXES.md` | Bugs found in v1's train.py reward_fn (now FIXED in v1 + v2). |
| 06 | `06_V1_FINAL_STATE.md` | **What actually shipped as v1**: file map, training results, step-12 eval numbers, Hub artefacts. |
| 07 | `07_V2_DESIGN_AND_IMPL.md` | v2 design + implementation summary. File-by-file walkthrough, action/reward references, train/eval/simulator commands. |
| 08 | `08_REPO_LAYOUT.md` | Top-level repo layout after the v1/ reorg. What stays at root vs lives in v1/ vs v2/. |

## Repo state (2026-04-25)

```
zombiee/
├── .claude/        Claude Code config + CONTEXT.md (orientation pointer)
├── .planning2/     this directory (planning + design docs)
├── version2.md     v2 design notes (private brainstorm; gitignored)
├── v1/             everything that shipped as v1 (frozen for reproducibility)
└── v2/             v2 implementation (in development; DGX-targeted)
```

The reorg consolidated all v1 code/configs into `v1/` so v2 can develop
side-by-side without import or config conflicts. See `08_REPO_LAYOUT.md`.

## Hackathon context (TL;DR)

- **Event:** Meta × PyTorch × Scaler OpenEnv Hackathon R2, on-site 2026-04-25/26, Bangalore.
- **Team:** 2 people (Sirjan + Eeshan).
- **Judging:** 40% innovation, 30% storytelling, 20% training curve, 10% pipeline quality.
- **Target themes:** 1 (multi-agent), 2 (long-horizon), 4 (self-improvement). Hit all three.

## Project one-liner

SurviveCity trains LLM agents to survive a zombie apocalypse by learning
from their past deaths. Each episode's deaths generate deterministic
post-mortems that are prepended to the next episode's system prompt.
v1 ships a 3-agent / 1-infected baseline. v2 adds bite transmission,
zombie waves, iterated voting, multi-resource (food/water/medicine),
inventory, day/night, and broadcast economy — the action space is a
strict superset of v1, so a v1 LoRA loads zero-shot for transfer
evaluation.
