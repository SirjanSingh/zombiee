"""GRPO training pipeline for SurviveCity.

Qwen2.5-3B-Instruct + LoRA + GRPO via Unsloth and TRL.

Usage:
    python -m training.train [--max-steps 4000] [--env-url http://localhost:7860]
"""

from __future__ import annotations

import argparse
import json
import logging
import random

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("survivecity.train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--env-url", default="http://localhost:7860")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--output-dir", default="./lora_v1")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="tensorboard")
    return p.parse_args()


def build_scenario_dataset(env_url, num_scenarios=200, seed=42):
    import requests
    from datasets import Dataset
    from survivecity_env.prompts import build_system_prompt

    rng = random.Random(seed)
    prompts = []
    for i in range(num_scenarios):
        try:
            r = requests.post(f"{env_url}/reset", json={"seed": rng.randint(0, 999999)})
            r.raise_for_status()
            obs = r.json()
            prompt = build_system_prompt(0, obs.get("description", ""))
            prompts.append({"prompt": prompt, "scenario_id": i})
        except Exception as e:
            logger.warning(f"Scenario {i} failed: {e}")
    return Dataset.from_list(prompts)


def create_reward_fn(env_url):
    import requests

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for c in completions:
            try:
                text = c.strip()
                if text.startswith("```"):
                    text = text.split("```")[1].removeprefix("json")
                action = json.loads(text)
                action.setdefault("agent_id", 0)
                r = requests.post(f"{env_url}/step", json=action, timeout=5)
                r.raise_for_status()
                rewards.append(r.json().get("reward", 0.01))
            except Exception:
                rewards.append(0.01)
        return rewards

    return reward_fn


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model_name, load_in_4bit=True, max_seq_length=args.max_seq_length)
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            use_gradient_checkpointing="unsloth")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, device_map="auto")
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none"))

    dataset = build_scenario_dataset(args.env_url, 200, args.seed)

    from trl import GRPOTrainer, GRPOConfig
    config = GRPOConfig(
        output_dir=args.output_dir, num_generations=args.num_generations,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=args.lr, max_steps=args.max_steps,
        save_steps=args.save_steps, logging_steps=10,
        max_prompt_length=1024, max_completion_length=512,
        temperature=args.temperature, beta=args.beta,
        report_to=args.report_to if args.report_to != "none" else None,
        seed=args.seed)

    trainer = GRPOTrainer(
        model=model, args=config,
        reward_funcs=[create_reward_fn(args.env_url)],
        train_dataset=dataset, processing_class=tokenizer)

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
