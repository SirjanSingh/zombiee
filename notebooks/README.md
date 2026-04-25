# Cross-machine training (Kaggle ⇄ DGX)

The DGX is shared and frequently saturated. This directory contains a Kaggle
notebook that runs the same GRPO training as `Dockerfile.dgx`, using the
**Hugging Face Hub** as a checkpoint bridge so the run can hop between the two
without losing progress.

## Files

| Path | Purpose |
|---|---|
| `train_kaggle.ipynb` | Self-contained Kaggle notebook. Installs deps, clones the repo, starts the env server, runs `training/train.py`, pushes checkpoints to the Hub after every save. |

## How resume works end-to-end

```
              ┌─────────── Hugging Face Hub ───────────┐
              │   sirjansingh/zombiee-qwen-grpo-lora    │
              │   ├── checkpoint-100/                   │
              │   ├── checkpoint-200/                   │
              │   ├── trainer_state.json                │
              │   └── adapter_model.safetensors         │
              └──────────────┬──────────────────────────┘
                  push_to_hub │             │ resume_from_checkpoint
                              ▼             ▼
              ┌─────────────────┐   ┌──────────────────┐
              │ Kaggle notebook │   │ DGX docker run   │
              │  (T4 / P100)    │   │  (V100, when     │
              │                 │   │   GPU is free)   │
              └─────────────────┘   └──────────────────┘
```

`training/train.py` accepts:

- `--push-to-hub --hub-model-id <user/repo>` — uploads `checkpoint-N/` after each save (`hub_strategy="every_save"`) and a final model on success.
- `--resume-from-checkpoint <spec>` where `<spec>` is one of:
  - `auto` — pick newest `checkpoint-*` under `--output-dir`
  - `<local path>` — resume from that directory
  - `<user/repo>` — `snapshot_download` from the Hub, then resume
- `--save-total-limit N` — keep only the last `N` checkpoints on disk (older ones are deleted; older ones on the Hub stay until you prune the repo).

## First-time Kaggle setup

1. **Push the latest code to GitHub** (`git push origin master`) so the notebook can clone it.
2. On <https://huggingface.co/settings/tokens>, create a token with **Write** permissions.
3. Open `notebooks/train_kaggle.ipynb` on Kaggle (File → Import Notebook → upload it, or use *New Notebook* and paste).
4. Settings:
   - **Accelerator**: `GPU T4 x1` (best free option). `P100` and `L4` also work; the script auto-detects bf16 vs fp16.
   - **Internet**: On.
   - **Persistence**: Variables and Files.
5. **Add-ons → Secrets** → add `HUGGINGFACE_TOKEN` = your token, attach to notebook.
6. In the *Configuration* cell, set `HUB_MODEL_ID` to your own HF user/repo (e.g. `sirjansingh/zombiee-qwen-grpo-lora`).
7. **Run All**. (Or *Save Version → Save & Run All (Commit)* to run headless on Kaggle infra and free up the tab.)

The Hub repo is auto-created on first push.

## Resuming on the DGX once a GPU is free

```bash
git pull
docker build -f Dockerfile.dgx -t survivecity-train .

docker run --rm --gpus '"device=N"' --shm-size=8g \
  -e HUGGINGFACE_TOKEN=hf_xxx \
  -v $(pwd)/lora_v1:/app/lora_v1 \
  survivecity-train \
  bash -c "uvicorn server.app:app --host 0.0.0.0 --port 7860 & sleep 3 && \
    python -m training.train \
      --resume-from-checkpoint sirjansingh/zombiee-qwen-grpo-lora \
      --push-to-hub --hub-model-id sirjansingh/zombiee-qwen-grpo-lora \
      --max-steps 4000 \
      --output-dir /app/lora_v1"
```

Replace `device=N` with whichever GPU is free (`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -n -r | head`) and `hf_xxx` with your token.

The same Hub repo will accept pushes from both Kaggle and DGX — `every_save`
uploads create a new commit each time, and `--resume-from-checkpoint` always
pulls the most recent `trainer_state.json` + adapter weights.

## Resuming on Kaggle (after a 12-h timeout)

Just re-run the notebook. Cell 6 (`Detect existing checkpoints on the Hub`)
sees the existing artifacts and prepends `--resume-from-checkpoint <repo>` to
the launch command automatically.

## Memory-tightening knobs (if Kaggle's T4 OOMs)

In the *Configuration* cell of the notebook, lower:

- `NUM_GENERATIONS` (4 → 2 → 1) — biggest activation-memory lever in GRPO
- `MAX_SEQ_LENGTH` (2048 → 1024)
- `MODEL_NAME` → `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (last resort)

The DGX (V100 32 GB) can run the full Qwen2.5-3B / `NUM_GENERATIONS=8` /
`MAX_SEQ_LENGTH=4096` config from `Dockerfile.dgx`'s default `CMD`; Kaggle's
16 GB T4 needs the trimmed defaults shown in the notebook.

## "My Colab/Kaggle session died — did I lose anything?"

**No** — as long as `--push-to-hub` was set (it is, in both notebooks), every
checkpoint up to the last successful save lives on the Hub at
`huggingface.co/<HUB_MODEL_ID>`. The `hub_strategy="every_save"` setting in
`training/train.py` uploads each `checkpoint-N/` immediately after it's
written to disk, before the next training step begins.

Inspect what survived:

```bash
# Just list:
python scripts/check_hub_checkpoints.py --hub-model-id noanya/zombiee

# Show training progress (step, loss, lr) from the latest checkpoint:
python scripts/check_hub_checkpoints.py --hub-model-id noanya/zombiee --info

# Pull the latest checkpoint locally:
python scripts/check_hub_checkpoints.py --hub-model-id noanya/zombiee \
    --download ./recovered
```

Then resume from anywhere:

| Where | How |
|---|---|
| Same Kaggle/Colab notebook | Just re-run it. Cell 6 auto-detects the Hub checkpoint and resumes. |
| DGX (single GPU) | `python -m training.train --resume-from-checkpoint noanya/zombiee --push-to-hub --hub-model-id noanya/zombiee --output-dir ./lora_v1` |
| DGX (auto-pick GPU) | `HUGGINGFACE_TOKEN=hf_xxx ./scripts/dgx_autorun.sh` (see below) |

## DGX autorun script

`scripts/dgx_autorun.sh` watches `nvidia-smi` and launches a training
container as soon as a GPU has enough free memory. It survives container
crashes (each launch resumes from the same Hub checkpoint), and will spin up
**additional** containers on other GPUs as they free up — up to `MAX_JOBS`.

Prereqs:
1. `docker build -f Dockerfile.dgx -t survivecity-train .` (do this once).
2. Export your HF token: `export HUGGINGFACE_TOKEN=hf_xxx`.

Run:

```bash
# 1 job, requires 10 GB free on a GPU before launching
./scripts/dgx_autorun.sh

# Tighter memory budget, allow up to 2 parallel jobs
MIN_FREE_GB=8 MAX_JOBS=2 ./scripts/dgx_autorun.sh

# See what it would do without actually launching
DRY_RUN=1 ./scripts/dgx_autorun.sh
```

Tunables (env vars):

| Var | Default | Meaning |
|---|---|---|
| `MIN_FREE_GB` | `10` | Minimum free GPU memory before considering a GPU |
| `MAX_JOBS` | `1` | Cap on parallel training containers |
| `POLL_INTERVAL` | `60` | Seconds between `nvidia-smi` scans |
| `HUB_MODEL_ID` | `noanya/zombiee` | HF Hub repo id |
| `MAX_STEPS` | `4000` | Passed through to `training/train.py` |
| `SAVE_STEPS` | `100` | Passed through to `training/train.py` |
| `OUTPUT_ROOT` | `./lora_v1` | Host dir; per-GPU subdirs are mounted into containers |
| `DRY_RUN` | `0` | If `1`, prints the launch command without running it |

Containers are named `survivecity-train-gpuN`. Stop everything with
`Ctrl-C` — the script's `trap` cleans up all launched containers.
