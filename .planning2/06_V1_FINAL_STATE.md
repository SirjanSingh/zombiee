# 06 — v1 Final State (what actually shipped)

> Snapshot taken 2026-04-25. Treat v1 as **frozen** going forward. Any new
> mechanics belong in v2. The only acceptable v1 changes are bugfixes that
> preserve the exact training/eval numbers below.

---

## File map (everything lives under `v1/`)

```
v1/
├── pyproject.toml         # name="survivecity" version="1.0.0"
├── openenv.yaml           # OpenEnv submission manifest (port 7860)
├── README.md              # how-to: install, run server, train, eval
├── Dockerfile / Dockerfile.dgx
├── survivecity_env/       # the env package (10 files)
│   ├── __init__.py        # public API: AgentState, ZombieState, SurviveAction, SurviveObservation, SurviveCityEnv
│   ├── models.py          # Pydantic models (8 action types: move_*, eat, wait, vote_lockout, broadcast)
│   ├── layout.py          # 10x10 grid, 4 food cells, 3x3 safehouse, 8 walls
│   ├── game.py            # core mechanics (single infected, vote at t=50, BFS zombies)
│   ├── infection.py       # masking + behavioural cues (literal text — was a v1 weakness)
│   ├── rubric.py          # 3 rubrics: survival + vote + group_outcome, clipped (0.01, 0.99)
│   ├── postmortem.py      # rule-based death summaries (Theme-4 hook)
│   ├── prompts.py         # SYSTEM_PROMPT_TEMPLATE for 3-agent world
│   └── env.py             # SurviveCityEnv wrapper (reset/step/state)
├── server/app.py          # FastAPI on port 7860
├── training/
│   ├── train.py           # GRPO + Unsloth/transformers fallback (~430 lines, has the V100 fp16 fallback path)
│   ├── eval.py            # baseline vs trained, generates 3 PNG plots
│   └── inference.py       # action_fn helpers
├── tests/                 # test_survivecity.py + smoke_test_server.py
├── notebooks/
│   ├── train_colab.ipynb  # the notebook that successfully trained the v1 LoRA
│   ├── train_kaggle.ipynb # parallel attempt (had bnb+DataParallel issue, fixed via CUDA_VISIBLE_DEVICES=0)
│   ├── eval_colab.ipynb   # eval harness (cell 9 has root-layout fallback for trainer_state.json)
│   └── eval_kaggle.ipynb
├── scripts/
│   ├── check_hub_checkpoints.py
│   ├── dgx_autorun.sh
│   ├── gpu_hold.{py,sh}
└── report/v1/             # the LaTeX report (final v1.tex, refs.bib, figures/, Makefile)
```

---

## Training run (the one that produced the shipped LoRA)

| Setting | Value |
|---|---|
| Hardware | Single Colab Tesla T4 (15.6 GB VRAM) |
| Base model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (4-bit pre-quantised; loaded via Unsloth fast path) |
| LoRA | r=16, α=32, dropout=0, no bias, target=`q,k,v,o_proj` |
| Algorithm | GRPO (TRL), group size 4, KL β=0.04, temperature 0.9 |
| Optimizer | lr=1e-5, cosine schedule → 0 |
| Batch | per-device=1, grad-accum=16 |
| Steps | **12** total (max_steps=12, save_steps=1) |
| Wallclock | **3 h 53 min** (≈1166 s/step, dominated by GRPO rollouts inside reward_fn) |
| Hub | pushed every save to `noanya/zombiee` (private), `hub_strategy="every_save"` |
| Run id | `runs/Apr25_02-16-25_c3b2d410c5bd/` (TB event file fingerprints the Colab host) |

**Final-step training metrics (from `checkpoint-12/trainer_state.json`):**
loss=0.0002, grad_norm=0.225, lr=0.0 (decayed to 0), reward=0.0313, reward_std=0.0109, KL=0.0043.

The KL drift was small throughout (<5e-3) — consistent with the small
per-group reward variance. Don't expect dramatic policy change in the
trained LoRA; the wins are mostly format learning + survival nudge.

---

## Step-12 evaluation (the headline numbers)

Eval ran on a separate Colab T4, loaded the LoRA over fp16 `Qwen/Qwen2.5-3B-Instruct`.
Timestamp: 2026-04-25T08:14:25Z. JSON: `noanya/zombiee/eval_results/eval_step_0012.json`.

| Metric | Baseline (n=30) | Trained (n=10) | Δ |
|---|---|---|---|
| Survival rate | 0.0 % (0/30) | **10.0 %** (1/10) | +10 pp |
| Vote accuracy | undefined (no baseline ep reached step 50) | 0.0 % (0/1) | n/a |
| Mean total reward | 0.457 | **0.797 ± 0.41** | +0.34 (1.7×) |
| Mean episode length | 19.13 ± 7.27 | **37.6 ± 22.1** | +18.5 (2.0×) |
| Parse failure rate | 0 % (random has no JSON) | **0 %** (across all trained-eval actions) | — |

The single trained survivor episode (eval ep=2) ran the full 100 steps with
total reward 1.965. The 9 non-survivor episodes still averaged 0.667 reward
(>baseline 0.457), so the policy is doing *something* even when it doesn't
make it to the horizon.

There was also an earlier eval at ≈step 10 (filename `eval_step_0000.json`,
05:38 UTC, n=5/n=20) that showed 20% survival — that smaller sample
collapsed back to 10% on the proper step-12 eval. Treat **step-12 as the
headline**, step-10 as a within-window sanity check.

---

## Key bugs found and fixed during the v1 run

These were uncovered during execution and fixed in-place. The current v1 code
has all of them resolved — list is here so future readers know not to
reintroduce them.

1. **GRPO reward_fn used a shared singleton env over HTTP.** Multiple
   GRPO generations corrupted state. Fixed: each completion gets its own
   local `SurviveCityEnv`. See `v1/training/train.py:create_reward_fn`.
2. **GRPO scenario prompts embed an episode `[SEED:N]` token** so the
   reward function can recreate the exact env state for fair within-group
   comparison.
3. **Reward signal weakness:** reward_fn evaluates only the *first* model
   action; remaining ~99 steps are uniform-random rollouts. This is a
   deliberate compute concession — the gradient is weak but real. v2's
   reward hook keeps the same shape (don't try to "fix" it inside v1).
4. **Pydantic `ValidationError` on broadcast >40 chars.** Fixed by truncating
   in `llm_action`. The 40-char cap is enforced at the model layer.
5. **Kaggle dual-T4 + bnb 4-bit + DataParallel** crashed via auto device-map.
   Fixed by `CUDA_VISIBLE_DEVICES=0` so only one GPU is visible to the
   loader.
6. **V100 (DGX) lacks native bf16.** v1 train.py mirrors transformers'
   strict capability check so `bf16=True` doesn't trip the validator —
   falls back to fp16 on cc<8.
7. **TRL >=0.15 expects `model.warnings_issued` dict.** Unsloth's fast loader
   bypasses the transformers init that creates it. v1 `_seed_warnings_issued`
   walks the PEFT wrapper chain and seeds it.
8. **Eval cell 12 expected `trainer_state.json` at the root** of the Hub repo
   (it's only in `checkpoint-N/` subdirs with `hub_strategy="every_save"`).
   Fixed with a root-layout fallback that defaults `global_step` to 0.

---

## Hub artefacts (`noanya/zombiee`, private)

```
adapter_model.safetensors          29.5 MB   (final step-12 LoRA, root)
adapter_config.json                          (LoRA r=16 α=32 q,k,v,o)
checkpoint-10/, checkpoint-11/, checkpoint-12/   (full snapshots; save_total_limit=3)
runs/Apr25_02-16-25_c3b2d410c5bd/  TB event file from the actual training run
runs/Apr24_18-29-50_a6256c64723a/  earlier failed Colab attempt — IGNORE
runs/Apr25_03-05-49_d322251ff1cb/  Kaggle attempt — IGNORE
runs/Apr25_03-35-16_b3ef2df4f8a9/  Kaggle attempt — IGNORE
eval_results/eval_step_0000.json   ≈step-10 (smaller sample, 05:38 UTC)
eval_results/eval_step_0000_bars.png
eval_results/eval_step_0012.json   step-12 — HEADLINE eval
eval_results/eval_step_0012_bars.png
eval_results/eval_history.png
```

---

## Things the report claims (and that the data supports)

1. OpenEnv compliance was first-pass — all 4 R1 validator traps preempted.
2. **100 % JSON parse rate** across the trained eval. Format was fully learned.
3. Mean reward 1.7× baseline, mean episode length 2.0× baseline (step-12).
4. One episode hit full 100-step survival with reward 1.965.
5. Emergent ToM broadcast: *"I notice A2 is very hungry and may be infected soon"* (truncated by the env's 40-char cap).

## Things the report explicitly does NOT claim

- Survival-rate dominance. 10 % is one episode out of ten — wide CI.
- Vote accuracy improvement. The single vote in the trained eval was wrong.
- Cross-checkpoint trend. Step-10 → step-12 actually went 20 % → 10 %; we
  framed it honestly as "larger sample, more conservative point estimate".

---

## Don't-touch list (v1 frozen)

- `v1/survivecity_env/*.py` — env mechanics. Mods break reproducibility.
- `v1/pyproject.toml` version stays at `1.0.0`.
- `v1/openenv.yaml` port stays 7860 (v2 uses 7861).
- The Hub repo `noanya/zombiee` — do not push training results from v2
  there. v2 should use a separate repo (e.g. `noanya/zombiee-v2`) so the
  transfer-learning comparison stays clean.

---

## Open follow-ups parked from v1

- HF Space deployment (`v1/Dockerfile` is ready; just `huggingface-cli upload`).
- 2-min demo video (script: env mechanics + post-mortem injection viz).
- Larger-N step-12 eval (n=50 trained / n=100 baseline) if compute frees up.
