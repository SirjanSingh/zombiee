"""SurviveCity v2 simulator — rich text-mode visualizer of one episode.

Runs a single episode and pretty-prints every step: full grid, all agent
states, zombie positions, action chosen, reward delta, broadcasts, plus
phase-change banners (waves, votes, day/night, infection reveals, bites).

Useful for: debugging the env, recording demo footage, and showing
"everything that's happening according to v2" in a self-contained transcript.

Usage:
    # Random policy (default)
    python -m training.simulator --seed 42

    # LLM-driven (loads a LoRA from local dir or HF Hub)
    python -m training.simulator --lora-path ./checkpoints/checkpoint-100 --seed 42

    # Save full transcript to file
    python -m training.simulator --seed 42 --output ./results/transcripts/sim42.txt

    # No colour (for piping to a file or non-tty)
    python -m training.simulator --seed 42 --no-color
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
from typing import Optional, TextIO

logging.basicConfig(level=logging.WARNING)


# ANSI colour codes — graceful no-op via _ANSI dict when --no-color
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg_dark": "\033[40m",
    "bg_dim": "\033[48;5;236m",
}


def _c(text: str, *codes: str, color: bool = True) -> str:
    if not color:
        return text
    prefix = "".join(_ANSI[k] for k in codes if k in _ANSI)
    return f"{prefix}{text}{_ANSI['reset']}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-actions", type=int, default=600)
    p.add_argument("--policy", choices=["random", "llm"], default="random")
    p.add_argument("--lora-path", default=None,
                   help="Local checkpoint dir or HF Hub repo id; "
                        "if given, --policy is forced to llm.")
    p.add_argument("--revision", default=None)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour output.")
    p.add_argument("--output", default=None,
                   help="Write transcript to this file (in addition to stdout).")
    p.add_argument("--rewards-table-every", type=int, default=10,
                   help="Print a per-rubric reward breakdown every N steps.")
    return p.parse_args()


def render_grid_pretty(grid: list[list[str]], color: bool) -> str:
    """Render the 15x15 grid as fixed-width with colour for cell types."""
    out_lines = []
    for row in grid:
        cells = []
        for cell in row:
            if cell == ".":
                cells.append(_c(". ", "dim", color=color))
            elif cell == "#":
                cells.append(_c("# ", "white", color=color))
            elif cell == "F":
                cells.append(_c("F ", "yellow", color=color))
            elif cell == "W":
                cells.append(_c("W ", "cyan", color=color))
            elif cell == "M":
                cells.append(_c("M ", "magenta", "bold", color=color))
            elif cell == "S":
                cells.append(_c("S ", "green", color=color))
            elif cell == "Z":
                cells.append(_c("Z ", "red", "bold", color=color))
            elif cell.startswith("A"):
                # A0-A4 — colour per agent (cyan family)
                cells.append(_c(f"{cell} "[:2], "blue", "bold", color=color))
            else:
                cells.append(f"{cell} ")
        out_lines.append("".join(cells))
    return "\n".join(out_lines)


def fmt_agent_line(agent: dict, infected_role: Optional[str] = None, color: bool = True) -> str:
    """One-line agent summary."""
    if not agent.get("is_alive", True):
        return f"  A{agent['agent_id']}: " + _c("DEAD", "red", "dim", color=color)
    inf = agent.get("infection_state", "none")
    inf_tag = ""
    if inf == "latent":
        inf_tag = _c(" [LATENT]", "magenta", "bold", color=color)
    elif inf == "revealed":
        role = infected_role or agent.get("infection_role") or "?"
        inf_tag = _c(f" [REVEALED-{role}]", "red", "bold", color=color)
    locked = _c(" [LOCKED]", "yellow", "bold", color=color) if agent.get("locked_out") else ""
    inv = agent.get("inventory") or []
    inv_str = "[" + ",".join(i[0].upper() for i in inv) + "_" * (3 - len(inv)) + "]"
    return (
        f"  A{agent['agent_id']}: "
        f"({agent['row']:>2},{agent['col']:>2}) "
        f"hp={agent['hp']} hgr={agent['hunger']:>2} thr={agent['thirst']:>2} "
        f"inv={inv_str}{inf_tag}{locked}"
    )


def diff_state(prev: dict | None, curr: dict) -> list[str]:
    """Return human-readable phase-change banners that fired this step."""
    banners: list[str] = []
    if prev is None:
        return banners
    pmeta = prev.get("metadata", {})
    cmeta = curr.get("metadata", {})

    # Wave spawned (zombie count went up)
    if cmeta.get("n_zombies", 0) > pmeta.get("n_zombies", 0):
        if curr.get("step_count", 0) in (25, 50, 75):
            delta = cmeta["n_zombies"] - pmeta.get("n_zombies", 0)
            banners.append(f"🧟 ZOMBIE WAVE @ step {curr['step_count']}: +{delta} zombies "
                           f"(now {cmeta['n_zombies']} on map)")

    # Day/night flip
    if cmeta.get("day_phase") != pmeta.get("day_phase"):
        emoji = "🌙" if cmeta.get("day_phase") == "night" else "☀"
        banners.append(f"{emoji} DAY/NIGHT FLIP: {pmeta.get('day_phase')} -> {cmeta.get('day_phase')}")

    # New bites
    new_bites = len(cmeta.get("bite_history", [])) - len(pmeta.get("bite_history", []))
    if new_bites > 0:
        for b in cmeta["bite_history"][-new_bites:]:
            banners.append(f"🦷 BITE: A{b['biter_id']} bit A{b['victim_id']} at step {b['step']}")

    # New postmortems (deaths)
    new_pms = len(cmeta.get("postmortems", [])) - len(pmeta.get("postmortems", []))
    if new_pms > 0:
        for pm in cmeta["postmortems"][-new_pms:]:
            banners.append(f"💀 DEATH: {pm}")

    # Vote resolution
    new_lockouts = set(cmeta.get("lockout_results", {}).keys()) - set(pmeta.get("lockout_results", {}).keys())
    for vstep in sorted(new_lockouts):
        target = cmeta["lockout_results"].get(vstep)
        banners.append(f"🗳 VOTE PHASE @ step {vstep}: lockout result -> "
                       f"{f'A{target} locked out' if target is not None else 'no plurality'}")

    return banners


def step_banner(step: int, day_phase: str, color: bool) -> str:
    icon = "☀" if day_phase == "day" else "🌙"
    return _c(f"\n=== Step {step:>3} | {day_phase.upper()} {icon} ===", "bold", color=color)


def run_simulation(args, out: TextIO):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from survivecity_v2_env.env import SurviveCityV2Env
    from training.inference import random_action

    color = not args.no_color and out.isatty() if hasattr(out, "isatty") else not args.no_color

    # Build the action_fn
    action_fn = None
    policy_label = "RANDOM"
    if args.lora_path or args.policy == "llm":
        try:
            from training.eval import load_trained_model
            from training.inference import make_llm_action_fn
            model, tokenizer = load_trained_model(args.model_name, args.lora_path, args.revision)
            if model is None:
                print(_c("⚠ LoRA load failed — falling back to random.", "yellow", color=color), file=out)
            else:
                action_fn = make_llm_action_fn(model, tokenizer, max_new_tokens=args.max_new_tokens)
                policy_label = f"LLM ({args.lora_path or args.model_name})"
        except Exception as e:
            print(_c(f"⚠ LoRA load error: {e} — falling back to random.", "yellow", color=color), file=out)

    if action_fn is None:
        rng = random.Random(args.seed)
        action_fn = lambda aid, obs, _r=rng: random_action(aid, obs, rng=_r)

    print(_c("=" * 78, "bold", color=color), file=out)
    print(_c(f"  SurviveCity v2 simulator — seed={args.seed} policy={policy_label}", "bold", color=color), file=out)
    print(_c("=" * 78, "bold", color=color), file=out)

    env = SurviveCityV2Env()
    obs = env.reset(seed=args.seed)

    starting_infected = obs["metadata"]["starting_infected"]
    print(file=out)
    print(_c(f"Starting setup:", "bold", color=color), file=out)
    print(f"  Episode seed: {args.seed}", file=out)
    print(f"  Agents: 5 (A0..A4) at safehouse spawn", file=out)
    print(f"  Starting infected: {starting_infected} (one biter, one saboteur — roles hidden in masked obs)", file=out)
    print(f"  Initial zombies: {obs['metadata']['n_zombies']}", file=out)
    print(f"  Resources: F=8, W=4, M=2 cells. Inventory cap=3 per agent.", file=out)
    print(f"  Wave schedule: t=25 (+2), t=50 (+3), t=75 (+3). Zombie cap=12.", file=out)
    print(f"  Vote phases: t=30, 60, 90.", file=out)
    print(f"  Day/night: 0-24 day, 25-49 night, 50-74 day, 75-99 night.", file=out)
    print(file=out)

    prev_obs: Optional[dict] = None
    last_step_seen = -1
    actions_taken = 0
    cum_total = {i: 0.0 for i in range(5)}

    while not obs.get("done", False) and actions_taken < args.max_actions and obs.get("step_count", 0) <= args.max_steps:
        step = obs.get("step_count", 0)
        agent_id = obs["metadata"]["current_agent_id"]
        day_phase = obs["metadata"].get("day_phase", "day")

        # Print step banner once per step (first agent's turn)
        if step != last_step_seen:
            print(step_banner(step, day_phase, color), file=out)
            print(_c("Grid:", "dim", color=color), file=out)
            print(render_grid_pretty(obs["grid"], color), file=out)
            print(_c("Agents (only own infection_state shown):", "dim", color=color), file=out)
            for a in obs["agents"]:
                print(fmt_agent_line(a, color=color), file=out)
            zombies = obs["zombies"]
            if zombies:
                z_str = ", ".join(f"Z{z['zombie_id']}({z['row']},{z['col']})" for z in zombies)
                print(_c(f"Zombies ({len(zombies)}): {z_str}", "dim", color=color), file=out)
            print(_c(
                f"Noise meter: {obs['metadata'].get('noise_meter', 0)}/"
                f"{obs['metadata'].get('noise_threshold', 3)}", "dim", color=color
            ), file=out)
            last_step_seen = step

        # Diff banners (waves, deaths, votes resolved, day/night flip, bites)
        for banner in diff_state(prev_obs, obs):
            print(_c("  ⚑ " + banner, "yellow", color=color), file=out)

        # Get & apply the action for current agent
        try:
            action = action_fn(agent_id, obs)
        except Exception as e:
            print(_c(f"  action_fn error for A{agent_id}: {e} -> falling back to wait", "red", color=color), file=out)
            action = {"agent_id": agent_id, "action_type": "wait"}

        # Render the action choice nicely
        atype = action.get("action_type", "?")
        suffix_parts = []
        for k in ("vote_target", "scan_target", "inject_target", "gift_target", "item_slot", "item_type"):
            v = action.get(k)
            if v is not None:
                suffix_parts.append(f"{k}={v}")
        if action.get("message"):
            suffix_parts.append(f"msg={action['message'][:40]!r}")
        suffix = (" " + ", ".join(suffix_parts)) if suffix_parts else ""

        print(
            _c(f"  -> A{agent_id}", "blue", "bold", color=color)
            + " chose " + _c(atype, "cyan", color=color) + suffix,
            file=out,
        )

        prev_obs = obs
        obs = env.step(action)
        actions_taken += 1

        # Reward delta
        last_raw = obs.get("metadata", {}).get("raw_reward", 0.0)
        cum_total[agent_id] = cum_total.get(agent_id, 0.0) + last_raw
        delta_color = "green" if last_raw >= 0 else "red"
        print(
            f"     reward={obs.get('reward'):.4f} (raw={_c(f'{last_raw:+.4f}', delta_color, color=color)}) "
            f"running_total_A{agent_id}={cum_total[agent_id]:+.4f}",
            file=out,
        )

        # Inject result side-channel
        inj = obs["metadata"].get("last_inject_result")
        if inj:
            tag = "green" if inj in {"self_cured", "other_cured"} else "yellow"
            print(_c(f"     inject -> {inj}", tag, color=color), file=out)

        # Optional per-rubric breakdown periodically
        if args.rewards_table_every > 0 and step != prev_obs.get("step_count") and step % args.rewards_table_every == 0:
            br = obs.get("metadata", {}).get("rubric_breakdown")
            if br:
                print(_c(f"  per-rubric breakdown (A{agent_id}):", "dim", color=color), file=out)
                for k, v in br.items():
                    if abs(v) > 1e-9:
                        print(f"      {k:<32} {v:+.4f}", file=out)

    # Episode end summary
    print(file=out)
    print(_c("=" * 78, "bold", color=color), file=out)
    print(_c(f"  EPISODE END  step={obs.get('step_count', 0)}  done={obs.get('done', False)}", "bold", color=color), file=out)
    print(_c("=" * 78, "bold", color=color), file=out)
    meta = obs.get("metadata", {})
    print(f"  Final alive: {meta.get('n_alive', 0)} / 5  "
          f"(healthy: {meta.get('n_healthy_alive', 0)})", file=out)
    print(f"  Total bites this episode: {len(meta.get('bite_history', []))}", file=out)
    print(f"  Lockout results: {meta.get('lockout_results', {})}", file=out)
    print(f"  Zombies on map at end: {meta.get('n_zombies', 0)}", file=out)
    print(f"  Cumulative raw reward per agent:", file=out)
    for aid in range(5):
        print(f"    A{aid}: {cum_total.get(aid, 0.0):+.4f}", file=out)

    # Print all post-mortems (the cross-episode learning content)
    pms = meta.get("postmortems", [])
    if pms:
        print(_c("\n  POST-MORTEMS (would be prepended to next episode's prompts):",
                 "bold", color=color), file=out)
        for pm in pms:
            print(f"    {pm}", file=out)
    print(_c("=" * 78, "bold", color=color), file=out)


def main():
    args = parse_args()
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        # Tee: write to both stdout and file. Use a simple wrapper.
        f = open(args.output, "w")

        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, s):
                for st in self.streams:
                    st.write(s)
            def flush(self):
                for st in self.streams:
                    st.flush()
            def isatty(self):
                return False  # never colorise when teeing

        run_simulation(args, Tee(sys.stdout, f))
        f.close()
        print(f"\nTranscript saved to {args.output}")
    else:
        run_simulation(args, sys.stdout)


if __name__ == "__main__":
    main()
