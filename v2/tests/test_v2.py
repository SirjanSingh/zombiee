"""Tests for SurviveCity v2 mechanics.

Run with:
    cd v2 && pytest -q
"""

from __future__ import annotations

import random

import pytest

from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.game import (
    create_episode,
    apply_agent_action,
    advance_step,
    advance_zombies,
)
from survivecity_v2_env.infection import should_bite, _hash01, P_BITE
from survivecity_v2_env import inventory as inv
from survivecity_v2_env import spawn as spawn_mod


# ---------------------------------------------------------------------------
# Reset / step basics
# ---------------------------------------------------------------------------

def test_reset_basics():
    env = SurviveCityV2Env()
    obs = env.reset(seed=1)
    assert len(obs["agents"]) == 5
    assert len(obs["zombies"]) == 3
    assert obs["step_count"] == 0
    # Two starting infected (one biter, one saboteur)
    starting = obs["metadata"]["starting_infected"]
    assert len(starting) == 2
    assert obs["metadata"]["phase"] in {"pre_biter_reveal", "post_biter_reveal", "mid_episode", "post_saboteur_reveal"}


def test_reward_in_open_unit_interval():
    """OpenEnv contract: every observed reward must be strictly in (0, 1)."""
    env = SurviveCityV2Env()
    obs = env.reset(seed=2)
    rng = random.Random(2)
    actions = ["move_up", "move_down", "eat", "wait", "drink", "pickup"]
    for _ in range(100):
        if obs["done"]:
            break
        aid = obs["metadata"]["current_agent_id"]
        a = {"agent_id": aid, "action_type": rng.choice(actions)}
        obs = env.step(a)
        r = obs["reward"]
        assert 0.0 < r < 1.0, f"reward {r} not in (0,1)"


def test_action_space_includes_v1_and_v2():
    """All v1 + v2 action types must validate without errors."""
    env = SurviveCityV2Env()
    obs = env.reset(seed=3)
    every_action = [
        "move_up", "move_down", "move_left", "move_right",
        "eat", "wait", "vote_lockout", "broadcast",
        "drink", "scan", "pickup", "drop", "give", "inject",
    ]
    for atype in every_action:
        if obs["done"]:
            break
        aid = obs["metadata"]["current_agent_id"]
        action = {
            "agent_id": aid,
            "action_type": atype,
            "vote_target": 1,
            "message": "hi",
            "scan_target": 2,
            "inject_target": 0,
            "gift_target": 1,
            "item_slot": 0,
            "item_type": "food",
        }
        obs = env.step(action)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _run_random_episode(seed: int, n_actions: int = 50) -> list[tuple]:
    env = SurviveCityV2Env()
    obs = env.reset(seed=seed)
    rng = random.Random(seed)
    history = []
    actions_pool = ["move_up", "move_down", "move_left", "move_right", "eat", "drink", "wait"]
    for _ in range(n_actions):
        if obs["done"]:
            break
        aid = obs["metadata"]["current_agent_id"]
        action = {"agent_id": aid, "action_type": rng.choice(actions_pool)}
        obs = env.step(action)
        history.append((
            obs["step_count"],
            obs["reward"],
            obs["metadata"]["n_alive"],
            obs["metadata"]["n_zombies"],
            len(obs["metadata"]["bite_history"]),
        ))
    return history


def test_episode_determinism():
    """Same seed should produce identical state trajectories."""
    h1 = _run_random_episode(42)
    h2 = _run_random_episode(42)
    assert h1 == h2


# ---------------------------------------------------------------------------
# Bite RNG
# ---------------------------------------------------------------------------

def test_bite_rng_is_deterministic():
    a = should_bite(123, 25, 0, 1)
    b = should_bite(123, 25, 0, 1)
    assert a == b


def test_bite_rng_distribution_matches_p_bite():
    """Empirical bite frequency over 5000 samples should be close to P_BITE."""
    samples = [should_bite(i, 25, 0, 1) for i in range(5000)]
    freq = sum(samples) / len(samples)
    assert abs(freq - P_BITE) < 0.03, f"bite freq {freq} != {P_BITE} (±0.03)"


def test_hash01_is_uniform():
    samples = [_hash01(i) for i in range(5000)]
    mean = sum(samples) / len(samples)
    assert 0.45 < mean < 0.55


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def test_inventory_cap():
    items: list[str] = []
    assert inv.add_item(items, "food")
    assert inv.add_item(items, "water")
    assert inv.add_item(items, "medicine")
    assert not inv.add_item(items, "food")  # full
    assert len(items) == 3


def test_inventory_remove_at():
    items = ["food", "water", "medicine"]
    assert inv.remove_at(items, 1) == "water"
    assert items == ["food", "medicine"]
    # Out of range
    assert inv.remove_at(items, 99) is None
    assert inv.remove_at(items, None) is None


def test_inventory_remove_first():
    items = ["food", "food", "medicine"]
    assert inv.remove_first(items, "food")
    assert items == ["food", "medicine"]
    assert not inv.remove_first(items, "water")  # no water


# ---------------------------------------------------------------------------
# Wave spawning
# ---------------------------------------------------------------------------

def test_wave_step_lookup():
    assert spawn_mod.is_wave_step(25)
    assert spawn_mod.is_wave_step(50)
    assert spawn_mod.is_wave_step(75)
    assert not spawn_mod.is_wave_step(0)
    assert not spawn_mod.is_wave_step(99)


def test_wave_spawn_count_capped():
    rng = random.Random(0)
    # If already at the cap, pick_wave_spawn_cells returns []
    cells = spawn_mod.pick_wave_spawn_cells(50, rng, occupied=[], current_zombie_count=12)
    assert cells == []


def test_wave_spawn_excludes_occupied():
    rng = random.Random(0)
    pool_size = len(spawn_mod.WAVE_SPAWN_POOL)
    # Occupy every pool cell -> no spawn possible
    cells = spawn_mod.pick_wave_spawn_cells(50, rng, occupied=spawn_mod.WAVE_SPAWN_POOL, current_zombie_count=0)
    assert cells == []
    # Otherwise we should get up to WAVE_SCHEDULE[50] cells
    rng2 = random.Random(0)
    cells2 = spawn_mod.pick_wave_spawn_cells(50, rng2, occupied=[], current_zombie_count=0)
    assert len(cells2) == spawn_mod.WAVE_SCHEDULE[50]
    # All chosen cells are inside the pool
    for c in cells2:
        assert c in spawn_mod.WAVE_SPAWN_POOL


# ---------------------------------------------------------------------------
# Vote resolution (manually driven)
# ---------------------------------------------------------------------------

def test_vote_resolution_manual():
    ep = create_episode(seed=0)
    ep.step_count = 30
    for aid in range(5):
        apply_agent_action(ep, aid, "vote_lockout", vote_target=4)
    advance_step(ep)
    assert ep.lockout_results.get(30) == 4
    assert ep.agents[4].locked_out
    assert 30 in ep.vote_resolved_steps


def test_vote_no_plurality():
    ep = create_episode(seed=0)
    ep.step_count = 30
    # 2 vote for A1, 2 vote for A2, 1 abstain
    apply_agent_action(ep, 0, "vote_lockout", vote_target=1)
    apply_agent_action(ep, 1, "vote_lockout", vote_target=1)
    apply_agent_action(ep, 2, "vote_lockout", vote_target=2)
    apply_agent_action(ep, 3, "vote_lockout", vote_target=2)
    apply_agent_action(ep, 4, "vote_lockout", vote_target=None)
    advance_step(ep)
    assert ep.lockout_results.get(30) is None
    assert not any(a.locked_out for a in ep.agents)


# ---------------------------------------------------------------------------
# Inject / medicine
# ---------------------------------------------------------------------------

def test_inject_self_cures_latent():
    ep = create_episode(seed=0)
    # Force agent 1 to be latent-bitten
    ep.agents[1].infection_state = "latent"
    ep.agents[1].infection_role = "biter"
    ep.agents[1].bite_at_step = 5
    ep.agents[1].inventory = ["medicine"]
    apply_agent_action(ep, 1, "inject", inject_target=1, item_slot=0)
    assert ep.agents[1].infection_state == "none"
    assert ep.agents[1].inventory == []
    assert ep.last_inject_result.get(1) == "self_cured"


def test_inject_wasted_on_healthy():
    ep = create_episode(seed=0)
    ep.agents[2].inventory = ["medicine"]
    # Agent 2 is healthy by default (unless they were one of the two starting
    # infected). Force to none for determinism.
    ep.agents[2].infection_state = "none"
    ep.agents[2].infection_role = None
    apply_agent_action(ep, 2, "inject", inject_target=2, item_slot=0)
    assert ep.agents[2].inventory == []
    assert ep.last_inject_result.get(2) == "wasted_on_healthy"


# ---------------------------------------------------------------------------
# Day/night
# ---------------------------------------------------------------------------

def test_day_night_phase_advancement():
    ep = create_episode(seed=0)
    ep.step_count = -1  # so first advance_step lands on 0
    advance_step(ep)
    assert ep.day_phase == "day"
    ep.step_count = 24
    advance_step(ep)
    assert ep.day_phase == "night"
    ep.step_count = 49
    advance_step(ep)
    assert ep.day_phase == "day"
    ep.step_count = 74
    advance_step(ep)
    assert ep.day_phase == "night"
