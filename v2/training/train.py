"""GRPO training pipeline for SurviveCity v2 — DGX-tuned (30 GB VRAM).

Defaults reflect "fit within a 12-hour DGX session, save every step":
    --model-name              Qwen/Qwen2.5-3B-Instruct
    --max-steps               15         (matches v1's save-every-step cadence
                                          but with a higher step ceiling)
    --save-steps              1          (every step → 15 checkpoints total)
    --save-total-limit        15         (keep all 15 on disk + Hub)
    --num-generations         12         (bigger group → stronger GRPO gradient)
    --gradient-accum-steps    4          (4 prompts/step × 12 gens = 48 evals/step)
    --max-completion-length   512        (longer responses, more tokens to learn from)
    --lora-r                  32
    --lora-alpha              64
    --max-seq-length          4096

Time budget: at ~24 min/step on A100, 15 steps ≈ 6 h of training plus Hub
push overhead and warmup; comfortably fits in a 12-hour DGX session. On V100
(no native bf16) expect ~50 min/step → 12.5 h total — right at the limit, so
prefer A100/H100 if you have the choice.

Memory strategy: bf16 base model + gradient checkpointing enabled on the
non-4bit path. Lets num_generations=12 fit alongside max_completion_length=512
without OOM at peak. Optimizer is `adamw_torch_fused` for an extra 3-5%
throughput on Ampere+.

VRAM holder: DISABLED. The block in main() that allocated a "holder" tensor
to pin headroom on shared GPUs has been commented out. In practice, on a
shared DGX the co-tenant already had their memory before our process started,
so the holder couldn't reclaim it; it just shrunk our own training budget.
If you want it back (e.g. on a single-tenant GPU you fully control),
uncomment the block in main() and pass `--vram-reserve-gb 16` or similar.

Hub push: opt-in via --push-to-hub. With `hub_strategy="every_save"`, every
save-step pushes the entire output_dir to the Hub repo, so the 15GB eval box
can pull mid-training and run training.eval against any checkpoint.

Usage:
    python -m training.train [args]

The script uses a LOCAL SurviveCityV2Env in the GRPO reward function — no
HTTP server required during training (same pattern as v1).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("survivecity_v2.train")

# Crash diagnostics: dumps Python tracebacks for every thread on fatal signals
# (SIGSEGV, SIGABRT, SIGBUS, SIGFPE) and on `kill -USR1 <pid>` for live probes.
# Without this, an OS-level kill (OOM, SIGSEGV from CUDA, etc.) leaves zero
# clue why training "just stopped".
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1)
except (AttributeError, ValueError):
    pass  # Windows / non-POSIX
# Force line buffering so the log we tail actually shows the LAST line before
# the kill, not whatever happened to be in the 8KB stdio buffer.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass


def _report_existing_checkpoints(output_dir: str) -> None:
    """Print any leftover checkpoint-* dirs and their disk usage at startup.

    Surfaces stale checkpoints from a previous run BEFORE training starts —
    so you notice if save-total-limit kept old ones around, or if a previous
    crashed run is about to be silently overwritten / resumed.
    """
    if not os.path.isdir(output_dir):
        logger.info(f"checkpoints: output_dir={output_dir} doesn't exist yet (fresh start).")
        return
    ckpts = sorted(
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    )
    if not ckpts:
        logger.info(f"checkpoints: no checkpoint-* dirs under {output_dir} (fresh start).")
        return
    total_gb = 0.0
    for c in ckpts:
        p = os.path.join(output_dir, c)
        try:
            sz = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(p) for f in fs
            ) / 1e9
        except OSError:
            sz = float("nan")
        total_gb += sz if sz == sz else 0
        logger.info(f"checkpoints: found {c}  ({sz:.2f} GB)")
    logger.info(
        f"checkpoints: {len(ckpts)} existing under {output_dir}, total {total_gb:.2f} GB. "
        f"Pass --resume-from-checkpoint auto to continue from the latest."
    )


def _start_mem_watchdog(interval_s: int = 30) -> None:
    """Background thread that logs RSS + GPU memory every interval_s seconds.

    Lets you tell apart "killed by linux OOM" (RSS climbing → big jump → dead)
    from "killed externally" (RSS flat right up to the moment of death).
    Runs as a daemon so it doesn't block process exit.
    """
    try:
        import psutil
        proc = psutil.Process()
        get_rss_gb = lambda: proc.memory_info().rss / 1e9
        get_avail_gb = lambda: psutil.virtual_memory().available / 1e9
    except ImportError:
        import resource
        get_rss_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        get_avail_gb = lambda: float("nan")

    def loop():
        while True:
            try:
                msg = f"[mem] RSS={get_rss_gb():.1f}GB sys_avail={get_avail_gb():.1f}GB"
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info(0)
                    alloc = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    msg += (
                        f"  GPU_free={free/1e9:.1f}/{total/1e9:.1f}GB "
                        f"alloc={alloc:.1f}GB reserved={reserved:.1f}GB"
                    )
                logger.info(msg)
            except Exception as e:
                logger.warning(f"mem watchdog error: {type(e).__name__}: {e}")
            time.sleep(interval_s)

    t = threading.Thread(target=loop, daemon=True, name="mem-watchdog")
    t.start()
    logger.info(f"mem watchdog started (interval={interval_s}s)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--save-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-generations", type=int, default=12)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-prompt-length", type=int, default=1536)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-scenarios", type=int, default=200)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    # Resume / hub flags
    p.add_argument(
        "--resume-from-checkpoint", default=None,
        help="Path to a checkpoint dir, 'auto' for latest under --output-dir, "
             "or a HF Hub repo id (snapshot-downloaded).")
    p.add_argument(
        "--push-to-hub", action="store_true",
        help="Push every checkpoint to --hub-model-id (requires HUGGINGFACE_TOKEN).")
    p.add_argument(
        "--hub-model-id", default="noanya/zombiee-v2",
        help="HF Hub repo id. Default 'noanya/zombiee-v2' is the team's v2 repo "
             "(separate from v1's 'noanya/zombiee' so v1 artefacts stay frozen).")
    p.add_argument(
        "--hub-private", action="store_true", default=True,
        help="Create the hub repo as private. Default True for unreleased work; "
             "use --hub-public to make a fresh repo public.")
    p.add_argument(
        "--hub-public", dest="hub_private", action="store_false",
        help="Make the hub repo public (overrides --hub-private).")
    p.add_argument("--save-total-limit", type=int, default=15,
                   help="Keep at most this many checkpoints on disk locally. "
                        "Default 15 keeps every save from a 15-step / save_steps=1 run "
                        "(~30 MB adapter × 15 ≈ 450 MB on Hub).")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True,
                   help="Enable gradient checkpointing to fit larger num_generations × "
                        "max_completion_length within VRAM. Trades ~30%% per-step time for ~40%% "
                        "memory headroom — a worthwhile swap on a 30GB box at num_generations=12.")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false",
                   help="Disable gradient checkpointing (only safe with smaller num_generations).")
    p.add_argument("--optim", default="adamw_torch_fused",
                   help="Optimizer name. adamw_torch_fused is faster on Ampere+ (sm_80+); "
                        "fall back to adamw_torch on V100/T4.")
    p.add_argument(
        "--warmstart-from", default=None,
        help="Optional HF Hub repo id (or local path) to a v1 LoRA. "
             "Loaded as the initial PEFT adapter instead of training from "
             "scratch — this is the v2-warmstart-from-v1 transfer experiment.")
    p.add_argument(
        "--allow-fp16", action="store_true",
        help="Force fp16 even on Ampere+; default auto-picks bf16 on Ampere+ and fp16 elsewhere.")
    p.add_argument(
        "--no-4bit", action="store_true",
        help="Skip bitsandbytes 4-bit and load the base model in bf16/fp16 at full weight precision. "
             "Recommended on 30GB+ A100/H100 boxes — gives cleaner gradients than 4-bit.")
    p.add_argument(
        "--vram-reserve-gb", type=int, default=0,
        help="DEPRECATED — VRAM holder is disabled. Was: amount of GPU memory to leave "
             "free for training while holding the rest. In practice, on a shared GPU the "
             "co-tenant already had their memory BEFORE we started, so the holder couldn't "
             "claim that back; it just shrunk the budget for our own training peak. "
             "Default 0 = no holder. Pass any positive value to re-enable (the allocation "
             "block is commented out below — uncomment if you really want it).")
    return p.parse_args()


def build_scenario_dataset(num_scenarios: int = 200, seed: int = 42):
    """Build a GRPO scenario dataset from local v2 env resets.

    Each prompt embeds [SEED:N] so the reward function can recreate the exact
    env state for fair within-group comparison.
    """
    from datasets import Dataset
    from survivecity_v2_env.env import SurviveCityV2Env
    from survivecity_v2_env.prompts import build_system_prompt

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    rng = random.Random(seed)
    prompts = []
    for i in tqdm(range(num_scenarios), desc="build_scenario_dataset"):
        try:
            ep_seed = rng.randint(0, 999999)
            env = SurviveCityV2Env()
            obs = env.reset(seed=ep_seed)
            desc = obs.get("description", "")
            prompt = build_system_prompt(0, f"[SEED:{ep_seed}]\n{desc}")
            prompts.append({"prompt": prompt, "scenario_id": i})
        except Exception as e:
            logger.warning(f"Scenario {i} failed: {e}")
    logger.info(f"Built {len(prompts)} v2 scenario prompts")
    return Dataset.from_list(prompts)


def create_reward_fn():
    """GRPO reward function using LOCAL v2 env instances.

    Each completion gets its own SurviveCityV2Env reset to the SAME seed
    embedded in the prompt. Model emits action for agent 0 step 0; the
    remaining ~99 steps of the episode are rolled out with random actions
    so the terminal reward captures downstream effects of the model's choice.
    """
    from survivecity_v2_env.env import SurviveCityV2Env
    from training.inference import (
        parse_action, random_action, RANDOM_NON_VOTE_ACTIONS,
    )
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    # Track call count + error stats so we can spot a silently-degrading reward
    # function (e.g. env throwing on every call → all rewards = 0.01).
    state = {"calls": 0, "errors": 0}

    def reward_fn(prompts, completions, **kwargs):
        state["calls"] += 1
        rewards: list[float] = []
        n = len(prompts)
        iterator = tqdm(
            list(zip(prompts, completions)),
            total=n,
            desc=f"reward_fn rollouts (call #{state['calls']})",
            leave=False,
        )
        for prompt, completion in iterator:
            try:
                seed_match = re.search(r"\[SEED:(\d+)\]", prompt)
                ep_seed = int(seed_match.group(1)) if seed_match else (
                    abs(hash(prompt)) % 1_000_000
                )
                env = SurviveCityV2Env()
                obs = env.reset(seed=ep_seed)

                action = parse_action(completion, agent_id=0)
                if action is None:
                    rewards.append(0.01)
                    continue

                obs = env.step(action)

                # Roll out the remainder of the episode with random actions
                rollout_rng = random.Random(ep_seed + 7)
                steps = 0
                while not obs.get("done", False) and steps < 600:
                    aid = obs.get("metadata", {}).get("current_agent_id", 0)
                    sc = obs.get("step_count", 0)
                    if sc in (30, 60, 90):
                        rand_act = {
                            "agent_id": aid,
                            "action_type": "vote_lockout",
                            "vote_target": rollout_rng.choice([0, 1, 2, 3, 4]),
                        }
                    else:
                        rand_act = {
                            "agent_id": aid,
                            "action_type": rollout_rng.choice(RANDOM_NON_VOTE_ACTIONS),
                        }
                    obs = env.step(rand_act)
                    steps += 1
                rewards.append(obs.get("reward", 0.01))
            except Exception as e:
                # Was logger.debug — invisible at INFO level. Bumped to warning
                # so a degraded env actually shows up in the log instead of
                # silently producing all-0.01 rewards.
                state["errors"] += 1
                if state["errors"] <= 5 or state["errors"] % 50 == 0:
                    logger.warning(
                        f"reward_fn error #{state['errors']} "
                        f"({type(e).__name__}): {e}"
                    )
                rewards.append(0.01)
        nonzero = sum(1 for r in rewards if r != 0.01)
        logger.info(
            f"reward_fn call #{state['calls']}: {n} completions, "
            f"{nonzero} non-default rewards, total errors so far={state['errors']}"
        )
        return rewards

    return reward_fn


def _resolve_resume(spec: str | None, output_dir: str):
    if not spec:
        return None
    if spec == "auto":
        if os.path.isdir(output_dir) and any(
            d.startswith("checkpoint-") for d in os.listdir(output_dir)
        ):
            return True
        logger.info(f"No checkpoint-* dir under {output_dir}; starting fresh.")
        return None
    if os.path.isdir(spec):
        return spec
    if "/" in spec and not spec.startswith(("./", "/")):
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=spec, local_dir=os.path.join(output_dir, "_resume")
        )
        logger.info(f"Downloaded {spec} -> {local}")
        return local
    return spec


def _seed_warnings_issued(m, depth: int = 0):
    """Seed `warnings_issued` dict on every PEFT wrapper (TRL >=0.15 quirk)."""
    if m is None or depth > 6:
        return
    try:
        if not isinstance(getattr(m, "warnings_issued", None), dict):
            m.warnings_issued = {}
    except Exception:
        pass
    for attr in ("base_model", "model"):
        inner = getattr(m, attr, None)
        if inner is not None and inner is not m:
            _seed_warnings_issued(inner, depth + 1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Surface leftover checkpoints from a prior run BEFORE we touch anything,
    # so the operator notices stale state (and can pass --resume-from-checkpoint
    # auto if they wanted to continue).
    _report_existing_checkpoints(args.output_dir)

    # Fire up the RSS/GPU mem watchdog NOW, before model load — captures the
    # baseline so a sudden RSS jump in the run is obvious.
    _start_mem_watchdog(interval_s=30)

    # Precision selection
    cuda_ok = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability(0) if cuda_ok else (0, 0)
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    use_bf16 = cuda_ok and cap[0] >= 8 and cuda_major >= 11 and not args.allow_fp16
    use_fp16 = cuda_ok and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if cuda_ok:
        free_b, total_b = torch.cuda.mem_get_info(0)
        logger.info(
            f"GPU={torch.cuda.get_device_name(0)} cc={cap[0]}.{cap[1]} "
            f"cuda={torch.version.cuda} bf16={use_bf16} fp16={use_fp16} "
            f"VRAM free={free_b/1e9:.1f}GB total={total_b/1e9:.1f}GB"
        )
    else:
        logger.warning("CUDA not available — training will be CPU-only (slow).")

    # --------------------------------------------------------------
    # VRAM holder — DISABLED.
    #
    # Background: the holder pre-allocated a tensor for "unused" VRAM so
    # co-tenants on a shared GPU couldn't claim it mid-run. In practice,
    # on a shared DGX the co-tenant ALREADY had their memory before our
    # process started, so the holder never actually reclaimed anything.
    # All it did was shrink our own training budget by ~5 GB. With
    # 22 GB free + holder reserving 16 + safety 1 = 5 GB held, our
    # training only had 17 GB left — and Config A peak hits ~14-18 GB,
    # so the holder pushed us to OOM.
    #
    # The whole block below is commented out. If you want the holder
    # back (e.g. on a single-tenant GPU where you really do own all of
    # it), uncomment + pass `--vram-reserve-gb 16` (or whatever).
    # --------------------------------------------------------------
    if cuda_ok:
        logger.info("VRAM holder disabled (commented out). Training uses whatever the GPU has free.")
    # if cuda_ok and args.vram_reserve_gb > 0:
    #     try:
    #         free_b, _ = torch.cuda.mem_get_info(0)
    #         free_gb = free_b / 1e9
    #         safety_gb = 1.0
    #         hold_gb = free_gb - args.vram_reserve_gb - safety_gb
    #         if hold_gb > 0.5:
    #             torch._zombiee_vram_holder = torch.empty(
    #                 int(hold_gb * (1024 ** 3)),
    #                 dtype=torch.uint8,
    #                 device="cuda:0",
    #             )
    #             free_after_b, _ = torch.cuda.mem_get_info(0)
    #             logger.info(
    #                 f"VRAM holder pinned: {hold_gb:.2f} GB held on GPU 0 "
    #                 f"(reserved {args.vram_reserve_gb} GB for training, "
    #                 f"{safety_gb:.1f} GB safety). "
    #                 f"Free VRAM after holder: {free_after_b/1e9:.2f} GB."
    #             )
    #         else:
    #             logger.info(
    #                 f"Skipping VRAM holder: free {free_gb:.1f} GB - reserve "
    #                 f"{args.vram_reserve_gb} GB - safety {safety_gb:.1f} GB = "
    #                 f"{hold_gb:.1f} GB (need >0.5 GB to bother)."
    #             )
    #     except Exception as e:
    #         etype = type(e).__name__
    #         if "out of memory" in str(e).lower() or "OutOfMemoryError" in etype:
    #             logger.warning(
    #                 f"VRAM holder allocation hit OOM ({etype}): {str(e)[:120]}. "
    #                 f"Continuing without the holder."
    #             )
    #         else:
    #             raise

    device_map = {"": 0} if cuda_ok else "cpu"

    # Model + LoRA via transformers + peft (no Unsloth requirement here; it can
    # still be used by setting UNSLOTH_DISABLE=0 and manually swapping the loader).
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import (
        get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.no_4bit or not cuda_ok:
        bnb_config = None
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    if bnb_config is not None:
        # prepare_model_for_kbit_training enables gradient_checkpointing internally
        # AND ensures input grads are routed correctly through the embedding layer
        # (required for GRPO's backward pass).
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        # Non-4bit path: enable gradient checkpointing manually. The use_reentrant=False
        # variant is required for newer transformers (>=4.40) to avoid silent grad loss.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Ensure inputs require grad so checkpointed activations propagate backward
        # through the LoRA layers. Without this, GRPO's loss has no path to LoRA params.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing ENABLED (non-4bit path, use_reentrant=False)")

    # Either warmstart from a v1 (or v2) LoRA, or initialise a fresh LoRA
    if args.warmstart_from:
        logger.info(f"Warm-starting from existing LoRA: {args.warmstart_from}")
        adapter_path = args.warmstart_from
        if "/" in adapter_path and not os.path.isdir(adapter_path) and not adapter_path.startswith(("./", "/")):
            from huggingface_hub import snapshot_download
            adapter_path = snapshot_download(
                repo_id=args.warmstart_from,
                local_dir=os.path.join(args.output_dir, "_warmstart"),
            )
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    if cuda_ok:
        free_b, _ = torch.cuda.mem_get_info(0)
        logger.info(f"GPU memory after model load: free={free_b/1e9:.2f}GB")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|PAD_TOKEN|>"})
            model.resize_token_embeddings(len(tokenizer))

    dataset = build_scenario_dataset(args.num_scenarios, args.seed)

    from trl import GRPOTrainer, GRPOConfig

    push_to_hub = bool(args.push_to_hub and args.hub_model_id)
    if args.push_to_hub and not args.hub_model_id:
        logger.warning("--push-to-hub set without --hub-model-id; disabling hub push.")

    # Optimizer pick: adamw_torch_fused on Ampere+ (~3-5% throughput gain).
    # Fall back to plain adamw_torch on pre-Ampere (V100/T4) — fused needs sm_80+.
    chosen_optim = args.optim
    if chosen_optim == "adamw_torch_fused" and (not cuda_ok or cap[0] < 8):
        logger.info(
            f"adamw_torch_fused needs sm_80+ (got cc={cap[0]}.{cap[1]}); "
            "falling back to adamw_torch."
        )
        chosen_optim = "adamw_torch"

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        # Log every step — with only 15 total, we want every datapoint in TB.
        logging_steps=1,
        save_total_limit=args.save_total_limit,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        bf16=use_bf16,
        fp16=use_fp16,
        bf16_full_eval=use_bf16,
        fp16_full_eval=use_fp16,
        tf32=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=chosen_optim,
        report_to=args.report_to if args.report_to != "none" else None,
        push_to_hub=push_to_hub,
        hub_model_id=args.hub_model_id if push_to_hub else None,
        hub_private_repo=args.hub_private,
        hub_strategy="every_save" if push_to_hub else "end",
        seed=args.seed,
        # HF Trainer auto-disables tqdm when stdout isn't a TTY (i.e. inside
        # docker, under nohup, when piped through tee). Force it on so the
        # progress bar actually shows.
        disable_tqdm=False,
    )
    # Per-step generation budget = num_gen * (per_device_batch * grad_accum).
    # With defaults (num_gen=12, batch=1, grad_accum=4): 48 generations evaluated per
    # GRPO update. Compare v1 (num_gen=4, batch=1, grad_accum=16): 64 evals/step.
    gen_per_step = (
        args.num_generations * args.per_device_batch_size * args.grad_accum_steps
    )
    logger.info(
        f"GRPOConfig: bf16={config.bf16} fp16={config.fp16} "
        f"num_gen={config.num_generations} max_steps={config.max_steps} "
        f"save_steps={config.save_steps} grad_accum={config.gradient_accumulation_steps} "
        f"max_compl_len={config.max_completion_length} optim={chosen_optim} "
        f"grad_ckpt={args.gradient_checkpointing} gen_evals_per_step={gen_per_step}"
    )

    _seed_warnings_issued(model)

    # Per-step progress callback. tqdm is unreliable in docker/nohup logs
    # (rewrites lines, gets buffered, etc.) — this guarantees one clean
    # "step N/M" line per training step in the log file you're tailing.
    from transformers import TrainerCallback
    class StepProgressCallback(TrainerCallback):
        def __init__(self):
            self._t_step_start = None
            self._t_run_start = None
        def on_train_begin(self, args_, state_, control_, **kw):
            self._t_run_start = time.time()
            logger.info(
                f"[progress] training begin: max_steps={state_.max_steps} "
                f"num_train_epochs={args_.num_train_epochs}"
            )
        def on_step_begin(self, args_, state_, control_, **kw):
            self._t_step_start = time.time()
        def on_step_end(self, args_, state_, control_, **kw):
            dur = time.time() - (self._t_step_start or time.time())
            elapsed = time.time() - (self._t_run_start or time.time())
            done = state_.global_step
            total = state_.max_steps or 1
            remaining = max(total - done, 0)
            eta_min = (dur * remaining) / 60 if dur else 0
            logger.info(
                f"[progress] step {done}/{total}  "
                f"step_time={dur:.1f}s  elapsed={elapsed/60:.1f}min  "
                f"eta={eta_min:.1f}min"
            )
        def on_save(self, args_, state_, control_, **kw):
            logger.info(f"[progress] checkpoint saved at step {state_.global_step}")
        def on_log(self, args_, state_, control_, logs=None, **kw):
            if logs:
                snippet = {k: v for k, v in logs.items() if k in
                           ("loss", "reward", "grad_norm", "learning_rate", "kl")}
                if snippet:
                    logger.info(f"[metrics] step {state_.global_step}: {snippet}")

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[create_reward_fn()],
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[StepProgressCallback()],
    )

    resume = _resolve_resume(args.resume_from_checkpoint, args.output_dir)
    if resume is not None:
        logger.info(f"Resuming from checkpoint: {resume}")

    try:
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving checkpoint before exit.")
        trainer.save_model(args.output_dir)
        raise
    except Exception as e:
        # Without this, any non-KeyboardInterrupt exception bubbles up as a
        # raw traceback that's easy to miss when scrolling 12h of logs.
        # logger.exception writes the full traceback at ERROR level — and
        # also flushes (because we set line_buffering=True at top of file).
        logger.exception(
            f"trainer.train CRASHED: {type(e).__name__}: {e}. "
            f"Attempting to save partial state to {args.output_dir} ..."
        )
        try:
            trainer.save_model(args.output_dir)
        except Exception as save_err:
            logger.error(f"Could not save partial state: {save_err}")
        raise

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if push_to_hub:
        logger.info(f"Pushing final model to hub: {args.hub_model_id}")
        trainer.push_to_hub(commit_message="final v2 model")
    logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
