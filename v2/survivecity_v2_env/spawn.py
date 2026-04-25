"""Zombie spawn-wave scheduler.

Zombie spawns happen at deterministic step boundaries:
    t = 25  → +2 zombies
    t = 50  → +3 zombies
    t = 75  → +3 zombies

Spawn positions are sampled (without replacement) from a pre-computed pool
of inner-grid cells using the episode RNG, with cells currently occupied
by an agent or another zombie excluded. Capped at MAX_ZOMBIES total.
"""

from __future__ import annotations

import random
from typing import Iterable

from survivecity_v2_env.layout import WAVE_SPAWN_POOL


WAVE_SCHEDULE: dict[int, int] = {
    25: 2,
    50: 3,
    75: 3,
}

MAX_ZOMBIES = 12


def is_wave_step(step: int) -> bool:
    return step in WAVE_SCHEDULE


def pick_wave_spawn_cells(
    step: int,
    rng: random.Random,
    occupied: Iterable[tuple[int, int]],
    current_zombie_count: int,
) -> list[tuple[int, int]]:
    """Select cells to spawn zombies at for the given wave step.

    Args:
        step: current step number (must be in WAVE_SCHEDULE).
        rng: episode RNG (deterministic given the seed).
        occupied: cells currently containing an agent or zombie.
        current_zombie_count: how many zombies are alive right now.

    Returns:
        A list of (row, col) cells. Length is min(WAVE_SCHEDULE[step],
        MAX_ZOMBIES - current_zombie_count, len(available_pool)).
    """
    wanted = WAVE_SCHEDULE.get(step, 0)
    headroom = max(0, MAX_ZOMBIES - current_zombie_count)
    n = min(wanted, headroom)
    if n <= 0:
        return []

    occupied_set = set(occupied)
    available = [c for c in WAVE_SPAWN_POOL if c not in occupied_set]
    if not available:
        return []

    n = min(n, len(available))
    # Use rng.sample to keep determinism across runs with the same seed
    return rng.sample(available, n)
