# Claude Code orientation — SurviveCity v1 + v2

> Drop-in context for new Claude sessions or new humans on this repo.
> Read this once, then jump to `.planning2/` for the deep dives.

---

## Project in 3 sentences

SurviveCity is a multi-agent OpenEnv environment where LLM agents survive
a zombie apocalypse and learn from their past deaths via deterministic
post-mortems prepended to the next episode's prompt. v1 (3 agents,
1 hidden infected, single vote at t=50) shipped on 2026-04-25 with a
12-step GRPO LoRA on Qwen2.5-3B (Colab T4), achieving 100% JSON parse
rate, 1.7× baseline mean reward, and 2.0× baseline episode length on
step-12 eval (n_b=30, n_t=10). v2 (5 agents, 2 hidden infected with
bite-transmission, multi-resource, inventory, iterated voting, day/night,
zombie waves) is code-complete and locally smoke-tested but not yet
trained — DGX target is 30 GB VRAM, eval target is a separate 15 GB box.

---

## Where to look first

| If you want to... | Read |
|---|---|
| Understand v1's full original spec | `.planning2/00_PLAN.md` |
| Understand what v1 actually shipped (numbers, files, bugs fixed) | `.planning2/06_V1_FINAL_STATE.md` |
| Understand v2's design + impl + how to run | `.planning2/07_V2_DESIGN_AND_IMPL.md` |
| Understand the repo layout | `.planning2/08_REPO_LAYOUT.md` |
| Re-derive the bugs found in v1 training | `.planning2/05_TRAINING_ISSUES_AND_FIXES.md` |
| Avoid OpenEnv R1 validator traps | `.planning2/01_OPENENV_PATTERNS.md` |
| See the full v2 brainstorm (private) | `version2.md` (root, gitignored) |
| Browse v2 code | `v2/survivecity_v2_env/` |
| Browse v1 code | `v1/survivecity_env/` |
| See the rendered v1 report | `v1/report/v1/v1.tex` (compile via `make pdf`) |

---

## Repo layout (quick reference)

```
zombiee/
├── .claude/         this dir — settings.local.json + CONTEXT.md
├── .planning2/      planning docs (00..08)
├── version2.md      v2 brainstorm (gitignored)
├── v1/              frozen v1 (env + training + report)
└── v2/              active v2 (env + DGX training + 15GB eval + simulator)
```

`.gitignore` patterns at the root cover Python build artefacts, training
checkpoints, eval JSONs, and `version2.md`. `v2/.gitignore` adds
v2-specific runtime dir patterns.

---

## Common commands you'll likely run

```bash
# v1 eval (any checkpoint, anywhere with 15GB+ VRAM)
cd v1 && python -m training.eval --lora-path noanya/zombiee --eval-step 12

# v1 server (port 7860)
cd v1 && uvicorn server.app:app --host 0.0.0.0 --port 7860

# v2 tests (no GPU needed)
cd v2 && pytest -q

# v2 simulator (random policy, no GPU needed) — pretty visualisation
cd v2 && python -m training.simulator --seed 42 --no-color

# v2 simulator with a trained LoRA
cd v2 && python -m training.simulator --lora-path ./checkpoints/checkpoint-100 --seed 42 \
    --output ./results/transcripts/sim42.txt

# v2 training on DGX (30GB)
cd v2 && python -m training.train \
    --max-steps 200 --save-steps 25 \
    --lora-r 32 --lora-alpha 64 \
    --num-generations 8 \
    --push-to-hub --hub-model-id <user>/zombiee-v2

# v2 eval on a separate 15GB box
cd v2 && python -m training.eval \
    --lora-path <user>/zombiee-v2 --revision checkpoint-100 \
    --baseline-episodes 30 --trained-episodes 10 \
    --eval-step 100
```

---

## Headline numbers you'll be asked about

**v1 step-12 eval (the one in the report):**
- baseline n=30 vs trained n=10
- survival 0% → 10%
- mean reward 0.46 → 0.80 (1.7×)
- mean ep length 19.1 → 37.6 (2.0×)
- 100% JSON parse rate (zero parse failures across the trained eval)
- one trained ep hit full 100-step survival with reward 1.965

**v2:** not yet trained.

---

## Things to be careful about

1. **Don't push v2 training results to `noanya/zombiee`.** Use a separate
   Hub repo (e.g. `noanya/zombiee-v2`) so the cross-checkpoint comparison
   stays clean.
2. **Don't break v1.** The `v1/` tree is frozen — any modification has to
   preserve the step-12 eval numbers. New work goes in `v2/`.
3. **OpenEnv reward clip is contractual.** Every observed reward must be
   strictly in (0, 1). v1 and v2 both clip to (0.01, 0.99) via
   `compose_reward`. Don't bypass.
4. **HF tokens:** the user has leaked `HUGGINGFACE_TOKEN` in chat in the
   past. If you see one, flag it for revocation and avoid baking it into
   any committed file.
5. **Bite-RNG determinism in v2** is the load-bearing OpenEnv property.
   It uses BLAKE2b on `(episode_seed, step, biter_id, victim_id)`, NOT
   `random.random()`. Don't "improve" this.
6. **DGX VRAM is 30GB**, eval VRAM is **15GB**. v2 training defaults
   already match this; don't crank `num_generations` above 8 or
   `max_completion_length` above 320 without a memory check.

---

## Project authors / context

- **Team:** Sirjan Singh + Eeshan Singh ("PyGuys")
- **Hackathon:** Meta × PyTorch × Scaler OpenEnv Hackathon, India 2026
  (R2 on-site 2026-04-25/26, Bangalore). v1 was the R2 submission.
- **Themes targeted:** 1 (multi-agent), 2 (long-horizon), 4 (self-improvement).
- **Code repo:** `github.com/SirjanSingh/zombiee`
- **HF Hub:** `noanya/zombiee` (v1, private), `noanya/zombiee-v2` (planned)
