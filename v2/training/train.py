"""GRPO training pipeline for SurviveCity v2 — DGX-tuned (30 GB VRAM).

Defaults reflect "training on a 30GB DGX node with bf16 capability":
    --model-name      Qwen/Qwen2.5-3B-Instruct  (transferable to v2 from v1)
    --max-steps       200
    --save-steps      25                        (8 saves over a full run)
    --num-generations 8                         (was 4 for T4 v1)
    --lora-r          32                        (was 16)
    --lora-alpha      64                        (was 32)
    --max-seq-length  4096

Hub push is opt-in via --push-to-hub; if enabled, every checkpoint goes to
hub_model_id with hub_strategy="every_save", so a 15GB eval box can pull a
mid-training checkpoint and run training.eval against it without waiting for
the full run to finish.

Usage:
    python -m training.train [args]

The script uses a LOCAL SurviveCityV2Env in the GRPO reward function (no HTTP
server required during training — same pattern as v1's train.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("survivecity_v2.train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-prompt-length", type=int, default=1536)
    p.add_argument("--max-completion-length", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-scenarios", type=int, default=200)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    # Resume / hub flags
    p.add_argument(
        "--resume-from-checkpoint", default=None,
        help="Path to a checkpoint dir, 'auto' for latest under --output-dir, "
             "or a HF Hub repo id (snapshot-downloaded).")
    p.add_argument(
        "--push-to-hub", action="store_true",
        help="Push every checkpoint to --hub-model-id (requires HUGGINGFACE_TOKEN).")
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--hub-private", action="store_true")
    p.add_argument("--save-total-limit", type=int, default=4,
                   help="Keep at most this many checkpoints on disk locally.")
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
    return p.parse_args()


def build_scenario_dataset(num_scenarios: int = 200, seed: int = 42):
    """Build a GRPO scenario dataset from local v2 env resets.

    Each prompt embeds [SEED:N] so the reward function can recreate the exact
    env state for fair within-group comparison.
    """
    from datasets import Dataset
    from survivecity_v2_env.env import SurviveCityV2Env
    from survivecity_v2_env.prompts import build_system_prompt

    rng = random.Random(seed)
    prompts = []
    for i in range(num_scenarios):
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

    def reward_fn(prompts, completions, **kwargs):
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
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
                logger.debug(f"reward_fn error: {e}")
                rewards.append(0.01)
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
        model = prepare_model_for_kbit_training(model)

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

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=10,
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
        report_to=args.report_to if args.report_to != "none" else None,
        push_to_hub=push_to_hub,
        hub_model_id=args.hub_model_id if push_to_hub else None,
        hub_private_repo=args.hub_private,
        hub_strategy="every_save" if push_to_hub else "end",
        seed=args.seed,
    )
    logger.info(
        f"GRPOConfig: bf16={config.bf16} fp16={config.fp16} "
        f"num_gen={config.num_generations} max_steps={config.max_steps} "
        f"save_steps={config.save_steps} grad_accum={config.gradient_accumulation_steps}"
    )

    _seed_warnings_issued(model)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[create_reward_fn()],
        train_dataset=dataset,
        processing_class=tokenizer,
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

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if push_to_hub:
        logger.info(f"Pushing final model to hub: {args.hub_model_id}")
        trainer.push_to_hub(commit_message="final v2 model")
    logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
