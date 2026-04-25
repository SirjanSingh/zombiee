# Cross-machine training (Colab ⇄ Kaggle ⇄ DGX)

Free GPU sessions disconnect frequently. These notebooks run the same GRPO
training loop and use the **Hugging Face Hub** as a checkpoint bridge so the
run can hop between machines without losing progress.

## Files

| Path | Target | Purpose |
|---|---|---|
| `train_colab.ipynb` | Google Colab (T4) | Installs deps, clones the repo (private-repo-safe via `http.extraheader`), starts the env server, runs `training/train.py`, pushes a checkpoint to HF Hub after every gradient update. |
| `train_kaggle.ipynb` | Kaggle (T4 / P100 / L4) | Same flow as the Colab notebook, but uses `kaggle_secrets.UserSecretsClient` and `/kaggle/working` paths. |

Both notebooks share defaults: `MAX_STEPS=12`, `SAVE_STEPS=1`, `NUM_GENERATIONS=4`,
`gradient_accumulation_steps=16` (set in `training/train.py`). On a T4 each
GRPO step takes ~15-20 min, so you get a checkpoint roughly every 20 minutes
and a full run finishes in ~3-4h.

## How resume works end-to-end

```
              ┌─────────── Hugging Face Hub ───────────┐
              │   <user>/<repo>                         │
              │   ├── checkpoint-1/                     │
              │   ├── checkpoint-2/                     │
              │   ├── trainer_state.json                │
              │   └── adapter_model.safetensors         │
              └──────────────┬──────────────────────────┘
                  push_to_hub │             │ resume_from_checkpoint
                              ▼             ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐
        │ Colab notebook  │ │ Kaggle notebook │ │ DGX docker run   │
        │  (T4)           │ │  (T4 / P100)    │ │  (V100)          │
        └─────────────────┘ └─────────────────┘ └──────────────────┘
```

`training/train.py` accepts:

- `--push-to-hub --hub-model-id <user/repo>` — uploads `checkpoint-N/` after each save (`hub_strategy="every_save"`) and a final model on success.
- `--resume-from-checkpoint <spec>` where `<spec>` is one of:
  - `auto` — pick newest `checkpoint-*` under `--output-dir`
  - `<local path>` — resume from that directory
  - `<user/repo>` — `snapshot_download` from the Hub, then resume
- `--save-total-limit N` — keep only the last `N` checkpoints on disk (older ones are deleted; older ones on the Hub stay until you prune the repo).

## Required secrets (both notebooks)

| Secret | Scope | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | Fine-grained PAT with `Contents: Read` on the repo (or classic with `repo`) | Authenticate `git clone` for the private repo |
| `HF_TOKEN` | HuggingFace token, **write** | Push checkpoints to your HF Hub repo |

The clone cell embeds the GitHub token into a `git -c http.extraheader=...`
config so it never appears in `argv` or tracebacks. Errors are scrubbed before
being raised. Use a fine-grained PAT scoped to a single repo to limit blast
radius if a token leaks.

## First-time setup

### Colab

1. Push the latest code to GitHub so the notebook can clone it
2. Open `notebooks/train_colab.ipynb` on Colab (File → Open → GitHub, or upload the file)
3. Runtime → Change runtime type → T4 GPU
4. 🔑 Secrets (left sidebar) → add `GITHUB_TOKEN` and `HF_TOKEN`, toggle Notebook access ON for both
5. Edit `HUB_MODEL_ID` in the Configuration cell
6. Runtime → Run All

After the install cell (cell 3) finishes the first time, do **Runtime → Restart Session** then **Run All** again so the upgraded packages are picked up cleanly.

### Kaggle

1. Push the latest code to GitHub
2. Open `notebooks/train_kaggle.ipynb` on Kaggle (File → Import Notebook)
3. Settings:
   - Accelerator: `GPU T4 x1` (P100 or L4 also work)
   - Internet: On
   - Persistence: Variables and Files
4. Add-ons → Secrets → add `GITHUB_TOKEN` and `HF_TOKEN`, attach both to the notebook
5. Edit `HUB_MODEL_ID` in the Configuration cell
6. Run All (or *Save Version → Save & Run All (Commit)* to run headless on Kaggle's infra and free up the tab)

### DGX

```bash
git pull
docker build -f Dockerfile.dgx -t survivecity-train .

docker run --rm --gpus '"device=N"' --shm-size=8g \
  -e HUGGINGFACE_TOKEN=hf_xxx \
  -v $(pwd)/lora_v1:/app/lora_v1 \
  survivecity-train \
  bash -c "uvicorn server.app:app --host 0.0.0.0 --port 7860 & sleep 3 && \
    python -m training.train \
      --resume-from-checkpoint <user>/<repo> \
      --push-to-hub --hub-model-id <user>/<repo> \
      --max-steps 4000 \
      --output-dir /app/lora_v1"
```

Pick a free GPU index with:
```
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -n -r | head
```

## Resuming after a session timeout

Just re-run the notebook (or the DGX command) end-to-end. Cell 6
(*Detect Existing Checkpoints on Hub*) sees the existing artifacts on the Hub
and prepends `--resume-from-checkpoint <repo>` to the launch command
automatically.

## Memory-tightening knobs (if a free T4 OOMs)

In the Configuration cell of either notebook, lower:

- `NUM_GENERATIONS` (4 → 2 → 1) — biggest activation-memory lever in GRPO
- `MAX_SEQ_LENGTH` (2048 → 1024)
- `MODEL_NAME` → `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (last resort)

`training/train.py` also auto-throttles `num_generations` based on free VRAM:
< 6 GB → 2, < 12 GB → 4. So usually you don't need to touch it manually.

The DGX (V100 32 GB) can run the full Qwen2.5-3B / `NUM_GENERATIONS=8` /
`MAX_SEQ_LENGTH=4096` config from `Dockerfile.dgx`'s default `CMD`; Kaggle's
16 GB T4 needs the trimmed defaults shown in the notebook.

## Why so many short steps instead of one long run?

Free Colab/Kaggle sessions die unpredictably (idle timeout, queue eviction,
flaky network). With `MAX_STEPS=500 / SAVE_STEPS=50` the first save fires
~3h into training; if you get killed at 2h you have nothing. With
`MAX_STEPS=12 / SAVE_STEPS=1` the first save fires after step 1 (~20 min) and
every step after that. Worst-case you lose 19 min of compute, not 3h.

For long stable runs (DGX, paid Colab Pro+) bump `MAX_STEPS` and relax
`SAVE_STEPS` — the same notebook works.

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
