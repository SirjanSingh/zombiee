# 08 — Repo Layout (post v1/v2 reorg)

> Effective 2026-04-25 after the reorg that consolidated v1 into `v1/`.

---

## Top-level layout

```
zombiee/
├── .claude/         Claude Code config + CONTEXT.md (orientation pointer)
│   ├── settings.local.json
│   └── CONTEXT.md
├── .planning2/      planning + design documents (this directory)
│   ├── README.md
│   ├── 00_PLAN.md                       (v1 spec)
│   ├── 01_OPENENV_PATTERNS.md           (R1 validator traps)
│   ├── 02_REWARD_DESIGN.md              (v1 rubric formulas)
│   ├── 03_TRAINING_SETUP.md             (Unsloth/TRL recipe)
│   ├── 04_DEMO_AND_SUBMISSION.md
│   ├── 05_TRAINING_ISSUES_AND_FIXES.md  (bugs found mid-v1, all FIXED)
│   ├── 06_V1_FINAL_STATE.md             (what shipped: training run + eval numbers)
│   ├── 07_V2_DESIGN_AND_IMPL.md         (v2 file map + how to train/eval/sim)
│   └── 08_REPO_LAYOUT.md                (this file)
├── .git/            git metadata
├── .gitignore       repo-level patterns (Python build, training artefacts, version2.md)
├── version2.md      v2 design notes / brainstorm. GITIGNORED. Authoritative for design rationale.
├── v1/              EVERYTHING that shipped as v1 (frozen)
└── v2/              v2 implementation (in development, DGX-targeted)
```

`v1/` and `v2/` each own their own `pyproject.toml`, `Dockerfile`,
`openenv.yaml`, `README.md`, and `notebooks/` — fully self-contained.

---

## Why we moved to v1/ + v2/ side-by-side

Three reasons:

1. **Code lineage clarity.** Each version is reproducible from its own
   subtree. No `if version == "2": ...` branches polluting the v1 env.
2. **Independent installs.** `pip install -e v1/` and `pip install -e v2/`
   coexist; the package names differ (`survivecity` vs `survivecity-v2`)
   so they don't shadow each other. v2 imports `survivecity_v2_env`, so
   neither version's modules are on each other's import path.
3. **Independent submissions.** OpenEnv submissions are folder-scoped via
   `openenv.yaml`. v1's manifest binds port 7860; v2's binds port 7861;
   judges can run both concurrently.

---

## What stays at the repo root (and why)

| Item | Why root |
|---|---|
| `.claude/` | Tool config — Claude Code reads this from the working directory. |
| `.planning2/` | Cross-version planning docs. v1 docs ref v2 follow-on; would feel wrong scoped to a single version's subtree. |
| `.git/`, `.gitignore` | Git is repo-scoped by definition. The .gitignore covers patterns for both v1 (`v1/checkpoints` style) and v2. |
| `version2.md` | A brainstorm doc that spans both versions (compares v1 to v2). Gitignored as private. |

---

## What's *not* at the repo root any more

If you remember any of these from before the reorg, they're now under `v1/`:

- `survivecity_env/`        → `v1/survivecity_env/`
- `server/`                 → `v1/server/`
- `training/`               → `v1/training/`
- `notebooks/`              → `v1/notebooks/`
- `tests/`                  → `v1/tests/`
- `scripts/`                → `v1/scripts/`
- `report/`                 → `v1/report/`
- `pyproject.toml`          → `v1/pyproject.toml`
- `openenv.yaml`            → `v1/openenv.yaml`
- `Dockerfile`, `Dockerfile.dgx` → `v1/Dockerfile`, `v1/Dockerfile.dgx`
- `README.md`               → `v1/README.md`

`git mv` was used so `git log --follow` continues to track history across
the move.

---

## Working in this repo

### Working on v1 (frozen, bugfix-only)

```bash
cd v1
pip install -e .          # installs `survivecity` package
pytest -q                 # runs v1 tests
uvicorn server.app:app --port 7860
python -m training.eval --lora-path noanya/zombiee --eval-step 12
```

If you need to modify v1, do it in `v1/`. Don't import from `v1.survivecity_env`
inside `v2/` — keep them isolated.

### Working on v2 (active development)

```bash
cd v2
pip install -e .[train]   # installs `survivecity-v2` + torch/trl/peft
pytest -q                 # runs v2 tests
uvicorn server.app:app --port 7861
python -m training.train  ...                      # DGX 30GB
python -m training.eval  --lora-path ...           # 15GB eval box
python -m training.simulator --seed 42 --output ...
```

### Cross-version (transfer experiment)

The transfer experiment (planned) trains v2 with a v1 LoRA as warm-start:

```bash
cd v2
python -m training.train --warmstart-from noanya/zombiee  ...
```

The v1 LoRA loads as the initial PEFT adapter; subsequent GRPO steps
fine-tune it on v2 prompts. This is the headline science contribution
of v2.

---

## Where things go (decision tree for new files)

- "It's a v1 bugfix" → goes inside `v1/...` matching the original path.
- "It's a v2 feature/code" → goes inside `v2/...`.
- "It's a planning doc that mentions both versions" → `.planning2/`.
- "It's a Claude Code config / orientation" → `.claude/`.
- "It's a brainstorm doc not meant for the repo" → root, gitignored, name
  it like `version2.md` so the existing pattern matches.
- "It's a cross-version eval / comparison artefact" → it doesn't exist yet;
  if you make one, propose `comparison/` at the repo root and update this
  doc + `.gitignore`.

---

## Things that touch BOTH versions

These are deliberately the only cross-version coupling points:

1. **HF Hub repo separation:** `noanya/zombiee` (v1, frozen) vs
   `noanya/zombiee-v2` (v2, active). Treat them as separate "products"
   sharing only a base model.
2. **Base model:** both versions train against `Qwen/Qwen2.5-3B-Instruct`
   (eval) / `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (v1 train). v2 train
   defaults to the non-prequant fp16 base for cleaner DGX gradients.
3. **Action-space superset:** v1 LoRA loads on v2 → produces parseable v2
   actions (just won't ever emit `drink`/`scan`/`pickup`/`drop`/`give`/`inject`).
4. **Reward clip range:** both clip into (0.01, 0.99). Don't change either.
