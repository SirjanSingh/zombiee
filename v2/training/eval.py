"""SurviveCity v2 evaluation — runs N episodes per config, stores results.

Designed to run on a 15GB box (T4 / Colab). Loads any checkpoint (local
directory OR HF Hub repo id), runs baseline (random) and trained policies,
computes the v2 metric set, writes:

    eval_results/eval_step_<N>.json
    eval_results/eval_step_<N>_bars.png
    eval_results/eval_history.png        (auto-merged across runs)

Usage:
    python -m training.eval --lora-path ./checkpoints/checkpoint-100 --eval-step 100
    python -m training.eval --lora-path noanya/zombiee-v2 --eval-step 100
    python -m training.eval --lora-path noanya/zombiee-v2 --revision checkpoint-100

If --lora-path is None or unloadable, the trained policy falls back to
random — useful for sanity-checking the eval pipeline before any LoRA exists.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import logging
import os
import random
import re
import sys
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("survivecity_v2.eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-episodes", type=int, default=30)
    p.add_argument("--trained-episodes", type=int, default=10)
    p.add_argument("--lora-path", default=None,
                   help="Local checkpoint dir OR HF Hub repo id. Optional.")
    p.add_argument("--revision", default=None,
                   help="Hub revision/branch (e.g. 'checkpoint-100'). Used only "
                        "if --lora-path is a Hub repo id.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct",
                   help="Base model the LoRA was trained against.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output-dir", default="./eval_results")
    p.add_argument("--eval-step", type=int, default=0,
                   help="Logical step number for filename / history tracking.")
    p.add_argument("--push-results-to-hub", action="store_true",
                   help="Upload the eval JSON + bars PNG + history PNG to "
                        "--hub-model-id under eval_results/.")
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--max-steps-per-episode", type=int, default=600,
                   help="Safety cap on env.step calls per episode.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Episode runner — returns one record per episode
# ---------------------------------------------------------------------------

def run_one_episode(
    env,
    seed: int,
    action_fn,
    max_steps: int = 600,
) -> dict:
    """Run one v2 episode under `action_fn` and return a structured record."""
    obs = env.reset(seed=seed)
    starting_infected = list(obs.get("metadata", {}).get("starting_infected", []))

    cum_reward = 0.0
    n_steps = 0
    parse_failures = 0
    actions_called = 0

    # Vote-phase tracking (per round)
    vote_phases = (30, 60, 90)
    votes_per_phase: dict[int, dict[int, Optional[int]]] = {p: {} for p in vote_phases}

    # Inject and broadcast tracking
    inject_outcomes: list[str] = []
    broadcasts: list[str] = []

    while not obs.get("done", False) and n_steps < max_steps:
        actions_called += 1
        agent_id = obs.get("metadata", {}).get("current_agent_id", 0)

        try:
            action = action_fn(agent_id, obs)
        except Exception as e:
            logger.debug(f"action_fn error: {e}")
            from training.inference import random_action
            action = random_action(agent_id, obs)
            parse_failures += 1

        # Track inject and broadcasts BEFORE the env mutates state
        if isinstance(action, dict):
            if action.get("action_type") == "broadcast":
                msg = action.get("message")
                if isinstance(msg, str) and msg:
                    broadcasts.append(f"A{agent_id}: {msg[:40]}")
            elif action.get("action_type") == "vote_lockout":
                step_now = obs.get("step_count", 0)
                if step_now in vote_phases:
                    votes_per_phase[step_now][agent_id] = action.get("vote_target")

        obs = env.step(action)
        cum_reward += obs.get("reward", 0.5)

        # Inject result is set in metadata (per-step transient)
        last_inject = obs.get("metadata", {}).get("last_inject_result")
        if last_inject:
            inject_outcomes.append(last_inject)

        n_steps += 1

    meta = obs.get("metadata", {})

    # Per-vote-phase accuracy
    vote_phase_acc: dict[int, Optional[float]] = {}
    for phase_step, votes in votes_per_phase.items():
        if not votes:
            vote_phase_acc[phase_step] = None
            continue
        correct = 0
        total = 0
        for voter_id, target in votes.items():
            if target is None:
                continue
            total += 1
            if target in starting_infected:
                correct += 1
        vote_phase_acc[phase_step] = (correct / total) if total else None

    inject_correct = sum(1 for r in inject_outcomes if r in {"self_cured", "other_cured"})
    inject_total = sum(1 for r in inject_outcomes if r in {
        "self_cured", "other_cured", "wasted_on_healthy", "wasted_on_revealed"
    })
    medication_roi = (inject_correct / inject_total) if inject_total else None

    starting_infected_neutralised = all(
        (not a["is_alive"]) or a["locked_out"]
        for a in obs.get("agents", [])
        if a["agent_id"] in starting_infected
    ) if starting_infected else False

    rec = {
        "seed": seed,
        "steps": obs.get("step_count", 0),
        "actions_called": actions_called,
        "parse_failures": parse_failures,
        "total_reward": cum_reward,
        "done": obs.get("done", False),
        "n_alive_final": meta.get("n_alive", 0),
        "n_healthy_alive_final": meta.get("n_healthy_alive", 0),
        "n_zombies_final": meta.get("n_zombies", 0),
        "starting_infected": starting_infected,
        "currently_infected_final": meta.get("currently_infected_count", 0),
        "infection_chain_length": len(meta.get("bite_history", [])),
        "starting_infected_neutralised": starting_infected_neutralised,
        "lockout_results": meta.get("lockout_results", {}),
        "vote_phase_accuracy": vote_phase_acc,
        "n_inject_actions": inject_total,
        "n_inject_correct": inject_correct,
        "medication_roi": medication_roi,
        "n_broadcasts": len(broadcasts),
        "n_postmortems": len(meta.get("postmortems", [])),
        "survived": (meta.get("n_healthy_alive", 0) >= 1),
        "wave_survivors": {
            wave: sum(
                1 for a in obs.get("agents", [])
                if a["is_alive"] and (obs.get("step_count", 0) >= wave)
            ) if obs.get("step_count", 0) >= wave else None
            for wave in (25, 50, 75)
        },
    }
    return rec


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    n = len(records)

    rewards = [r["total_reward"] for r in records]
    steps = [r["steps"] for r in records]
    survived = sum(1 for r in records if r["survived"])

    inv_util = []
    for r in records:
        # We don't have inventory_final in record (kept light); skip for now
        pass

    # Vote accuracy aggregation per phase
    phase_acc: dict[int, dict[str, float | int]] = {}
    for phase in (30, 60, 90):
        accs = [r["vote_phase_accuracy"].get(phase) for r in records]
        accs = [a for a in accs if a is not None]
        if accs:
            phase_acc[phase] = {
                "mean": sum(accs) / len(accs),
                "n_episodes_with_votes": len(accs),
            }
        else:
            phase_acc[phase] = {"mean": None, "n_episodes_with_votes": 0}

    n_inject_total = sum(r["n_inject_actions"] for r in records)
    n_inject_correct = sum(r["n_inject_correct"] for r in records)

    parse_total = sum(r["actions_called"] for r in records)
    parse_failures = sum(r["parse_failures"] for r in records)

    return {
        "n_episodes": n,
        "survival_rate": survived / n,
        "n_survivors": survived,
        "mean_total_reward": sum(rewards) / n,
        "min_total_reward": min(rewards),
        "max_total_reward": max(rewards),
        "std_total_reward": _std(rewards),
        "mean_episode_length": sum(steps) / n,
        "std_episode_length": _std(steps),
        "vote_phase_accuracy": phase_acc,
        "medication_roi": (n_inject_correct / n_inject_total) if n_inject_total else None,
        "n_inject_total": n_inject_total,
        "n_inject_correct": n_inject_correct,
        "infection_containment_rate": sum(
            1 for r in records
            if r["currently_infected_final"] <= len(r["starting_infected"])
        ) / n,
        "infection_isolation_rate": sum(
            1 for r in records if r["starting_infected_neutralised"]
        ) / n,
        "parse_failure_rate": (parse_failures / parse_total) if parse_total else 0.0,
        "actions_called_total": parse_total,
        "mean_broadcasts_per_episode": sum(r["n_broadcasts"] for r in records) / n,
    }


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# LoRA loading
# ---------------------------------------------------------------------------

def load_trained_model(model_name: str, lora_path: Optional[str], revision: Optional[str]):
    """Try to load base model + LoRA. Returns (model, tokenizer) or (None, None)."""
    if not lora_path:
        return None, None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        logger.warning(f"Torch / transformers / peft not available: {e}")
        return None, None

    # Resolve lora_path: local dir or hub repo id
    resolved = lora_path
    if not os.path.isdir(lora_path):
        try:
            from huggingface_hub import snapshot_download
            resolved = snapshot_download(
                repo_id=lora_path,
                revision=revision,
                local_dir=os.path.join(".", "_lora_cache", lora_path.replace("/", "__")),
            )
            logger.info(f"Downloaded LoRA: {lora_path} (rev={revision}) -> {resolved}")
        except Exception as e:
            logger.warning(f"Could not resolve LoRA path '{lora_path}': {e}")
            return None, None

    # Verify adapter files exist
    has_adapter = any(
        f in os.listdir(resolved)
        for f in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    )
    if not has_adapter:
        logger.warning(f"No adapter files in {resolved} — falling back to random.")
        return None, None

    logger.info(f"Loading base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    logger.info(f"Loading LoRA from: {resolved}")
    model = PeftModel.from_pretrained(base, resolved)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def make_bars_plot(baseline_agg: dict, trained_agg: dict, out_path: str, eval_step: int):
    """Render a 4-panel bar chart: survival, mean reward, mean ep length, medication ROI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("Survival rate",
         baseline_agg.get("survival_rate", 0.0),
         trained_agg.get("survival_rate", 0.0)),
        ("Mean total reward",
         baseline_agg.get("mean_total_reward", 0.0),
         trained_agg.get("mean_total_reward", 0.0)),
        ("Mean episode length",
         baseline_agg.get("mean_episode_length", 0.0),
         trained_agg.get("mean_episode_length", 0.0)),
        ("Medication ROI",
         baseline_agg.get("medication_roi") or 0.0,
         trained_agg.get("medication_roi") or 0.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    for ax, (name, b, t) in zip(axes, metrics):
        bars = ax.bar(["Baseline", "Trained"], [b, t], color=["#ff8c42", "#3b82f6"])
        ax.set_title(name, fontsize=11)
        for bar, val in zip(bars, [b, t]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(abs(b), abs(t), 1.0),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"v2 eval @ step {eval_step}  "
        f"(baseline n={baseline_agg.get('n_episodes', 0)} vs "
        f"trained n={trained_agg.get('n_episodes', 0)})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def update_history_plot(output_dir: str):
    """Scan all eval_step_*.json under output_dir and render a trend chart.

    Always re-renders from scratch — the history plot is a faithful summary
    of every JSON the eval has ever written, so it's safe to re-run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(
        glob.glob(os.path.join(output_dir, "eval_step_*.json")),
        key=lambda p: int(re.search(r"eval_step_(\d+)", p).group(1)) if re.search(r"eval_step_(\d+)", p) else 0,
    )
    if not files:
        return

    steps: list[int] = []
    base_surv: list[float] = []
    trained_surv: list[float] = []
    base_reward: list[float] = []
    trained_reward: list[float] = []
    base_len: list[float] = []
    trained_len: list[float] = []

    for f in files:
        m = re.search(r"eval_step_(\d+)", f)
        if not m:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        steps.append(int(m.group(1)))
        b = data.get("baseline_aggregate", {})
        t = data.get("trained_aggregate", {})
        base_surv.append(b.get("survival_rate") or 0.0)
        trained_surv.append(t.get("survival_rate") or 0.0)
        base_reward.append(b.get("mean_total_reward") or 0.0)
        trained_reward.append(t.get("mean_total_reward") or 0.0)
        base_len.append(b.get("mean_episode_length") or 0.0)
        trained_len.append(t.get("mean_episode_length") or 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [
        ("Survival rate", base_surv, trained_surv),
        ("Mean total reward", base_reward, trained_reward),
        ("Mean episode length", base_len, trained_len),
    ]
    for ax, (title, b, t) in zip(axes, pairs):
        ax.plot(steps, b, "o-", color="#ff8c42", label="Baseline")
        ax.plot(steps, t, "s-", color="#3b82f6", label="Trained")
        ax.set_xlabel("training step")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("v2 cross-checkpoint eval history", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "eval_history.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ---------------------------------------------------------------------------
# Optional Hub upload of eval results
# ---------------------------------------------------------------------------

def push_results(hub_model_id: str, files: list[str]):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed; skipping push.")
        return
    api = HfApi()
    for f in files:
        if not os.path.exists(f):
            continue
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"eval_results/{os.path.basename(f)}",
            repo_id=hub_model_id,
            commit_message=f"eval upload: {os.path.basename(f)}",
        )
        logger.info(f"Pushed {f} -> {hub_model_id}/eval_results/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from survivecity_v2_env.env import SurviveCityV2Env
    from training.inference import random_action

    rng = random.Random(args.seed)

    # Baseline run — random actions
    logger.info(f"Baseline: {args.baseline_episodes} episodes (random policy)")
    baseline_records: list[dict] = []
    for ep in range(args.baseline_episodes):
        seed = rng.randint(0, 999_999)
        env = SurviveCityV2Env()
        rec = run_one_episode(
            env, seed,
            lambda aid, obs, _r=rng: random_action(aid, obs, rng=_r),
            max_steps=args.max_steps_per_episode,
        )
        baseline_records.append(rec)
    baseline_agg = aggregate(baseline_records)
    logger.info(f"  baseline survival={baseline_agg['survival_rate']:.2%} "
                f"reward={baseline_agg['mean_total_reward']:.3f} "
                f"len={baseline_agg['mean_episode_length']:.1f}")

    # Trained run — load LoRA if available
    model, tokenizer = load_trained_model(args.model_name, args.lora_path, args.revision)
    trained_records: list[dict] = []
    rng_trained = random.Random(args.seed + 7)

    if model is not None:
        from training.inference import make_llm_action_fn
        action_fn = make_llm_action_fn(model, tokenizer, max_new_tokens=args.max_new_tokens)
        trained_is_real = True
        logger.info(f"Trained: {args.trained_episodes} episodes (LoRA: {args.lora_path})")
    else:
        action_fn = lambda aid, obs, _r=rng_trained: random_action(aid, obs, rng=_r)
        trained_is_real = False
        logger.info(f"Trained: {args.trained_episodes} episodes "
                    f"(NO LORA — falling back to random)")

    for ep in range(args.trained_episodes):
        seed = rng_trained.randint(0, 999_999)
        env = SurviveCityV2Env()
        rec = run_one_episode(env, seed, action_fn, max_steps=args.max_steps_per_episode)
        trained_records.append(rec)
    trained_agg = aggregate(trained_records)
    logger.info(f"  trained survival={trained_agg['survival_rate']:.2%} "
                f"reward={trained_agg['mean_total_reward']:.3f} "
                f"len={trained_agg['mean_episode_length']:.1f}")

    # Persist results
    out_json = os.path.join(args.output_dir, f"eval_step_{args.eval_step:04d}.json")
    out_bars = os.path.join(args.output_dir, f"eval_step_{args.eval_step:04d}_bars.png")

    record = {
        "version": "v2",
        "eval_step": args.eval_step,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": {
            "model_name": args.model_name,
            "lora_path": args.lora_path,
            "revision": args.revision,
            "baseline_episodes": args.baseline_episodes,
            "trained_episodes": args.trained_episodes,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "trained_is_real": trained_is_real,
        },
        "baseline_aggregate": baseline_agg,
        "trained_aggregate": trained_agg,
        "baseline_records": baseline_records,
        "trained_records": trained_records,
    }
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Wrote {out_json}")

    make_bars_plot(baseline_agg, trained_agg, out_bars, args.eval_step)
    update_history_plot(args.output_dir)

    if args.push_results_to_hub and args.hub_model_id:
        files_to_push = [
            out_json,
            out_bars,
            os.path.join(args.output_dir, "eval_history.png"),
        ]
        push_results(args.hub_model_id, files_to_push)

    # Print a concise summary table to stdout
    print("=" * 70)
    print(f"v2 eval @ step {args.eval_step}  ({record['timestamp_utc']})")
    print("=" * 70)
    print(f"{'metric':<32} {'baseline':>15} {'trained':>15}")
    print("-" * 70)
    keys_to_print = [
        ("survival_rate",      "Survival rate"),
        ("mean_total_reward",  "Mean total reward"),
        ("mean_episode_length","Mean episode length"),
        ("infection_containment_rate", "Infection containment"),
        ("infection_isolation_rate",   "Infection isolation"),
        ("parse_failure_rate", "Parse failure rate"),
    ]
    for k, label in keys_to_print:
        bv = baseline_agg.get(k)
        tv = trained_agg.get(k)
        bs = f"{bv:.4f}" if isinstance(bv, (int, float)) else str(bv)
        ts = f"{tv:.4f}" if isinstance(tv, (int, float)) else str(tv)
        print(f"{label:<32} {bs:>15} {ts:>15}")
    print("=" * 70)
    print(f"Trained policy: {'REAL LoRA' if trained_is_real else 'random fallback'}")
    print(f"Files: {out_json}, {out_bars}, {os.path.join(args.output_dir, 'eval_history.png')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
