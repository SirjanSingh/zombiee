"""Core game logic for SurviveCity v2.

Pure-Python; no torch / no LLM dependencies. Imports only stdlib + the v2
package's own layout / inventory / spawn / infection / postmortem modules.

Per-step lifecycle (within advance_step / step()):
    A0 acts -> A1 acts -> ... -> A4 acts -> zombies move
        -> per-agent end-of-turn bite check (for revealed biters)
        -> wave spawn if step in {25, 50, 75}
        -> day/night flag updated
        -> noise meter decay
        -> latent->revealed transitions (for bitten agents past their countdown)
        -> vote phase resolution (for the step *after* a vote step)
        -> resource respawns
        -> terminal check
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from survivecity_v2_env.layout import (
    GRID_ROWS, GRID_COLS,
    SAFEHOUSE_CELLS, FOOD_CELLS, WATER_CELLS, MEDICINE_CELLS, WALL_CELLS,
    AGENT_SPAWNS, ZOMBIE_SPAWNS,
    build_grid,
)
from survivecity_v2_env import inventory as inv
from survivecity_v2_env import spawn as spawn_mod
from survivecity_v2_env import infection as infection_mod


# ---------------------------------------------------------------------------
# Internal mutable state
# ---------------------------------------------------------------------------

@dataclass
class _AgentInternal:
    agent_id: int
    row: int
    col: int

    hp: int = 3
    hunger: int = 0
    thirst: int = 0
    is_alive: bool = True

    # Infection
    infection_state: str = "none"      # "none" | "latent" | "revealed"
    infection_role: Optional[str] = None  # "biter" | "saboteur" | None
    bite_at_step: Optional[int] = None    # step at which infection started

    # Vote
    locked_out: bool = False

    # Inventory (list of "food" | "water" | "medicine")
    inventory: list[str] = field(default_factory=list)

    # Per-step transient flags (reset each step)
    ate_this_step: bool = False
    drank_this_step: bool = False
    damage_this_step: int = 0
    died_this_step: bool = False

    # Cumulative stats
    food_eaten: int = 0
    water_drunk: int = 0
    medicine_used: int = 0
    medicine_picked_up: int = 0
    death_step: Optional[int] = None
    death_cause: Optional[str] = None

    def reset_step_flags(self) -> None:
        self.ate_this_step = False
        self.drank_this_step = False
        self.damage_this_step = 0
        self.died_this_step = False


@dataclass
class _ZombieInternal:
    zombie_id: int
    row: int
    col: int


@dataclass
class EpisodeState:
    """Mutable full-episode state. Constructed by create_episode()."""

    agents: list[_AgentInternal]
    zombies: list[_ZombieInternal]
    base_grid: list[list[str]]

    # Episode meta
    step_count: int = 0
    max_steps: int = 100
    done: bool = False
    episode_seed: int = 0

    # Turn tracking
    agents_acted_this_step: int = 0

    # Voting
    # vote_step -> { voter_id: target_id|None }
    votes_cast: dict[int, dict[int, Optional[int]]] = field(default_factory=dict)
    vote_resolved_steps: set[int] = field(default_factory=set)
    vote_phase_just_resolved: dict[int, bool] = field(default_factory=dict)
    lockout_results: dict[int, Optional[int]] = field(default_factory=dict)

    # Broadcasts
    broadcasts: list[str] = field(default_factory=list)            # this step
    all_broadcasts: list[str] = field(default_factory=list)
    noise_meter: int = 0
    noise_threshold: int = 3   # > threshold => zombies +1 step toward agents
    noise_decay_period: int = 10
    broadcasts_over_threshold_this_step: dict[int, int] = field(default_factory=dict)

    # Bite history: list of {biter_id, victim_id, step}
    bite_history: list[dict] = field(default_factory=list)
    # Per-step transition tracking (used by rubric)
    latent_revealed_this_step: dict[int, bool] = field(default_factory=dict)
    last_inject_result: dict[int, str] = field(default_factory=dict)
    # Last scan result, optional, surfaced via observation metadata
    last_scan_result: dict[int, dict] = field(default_factory=dict)

    # Resource respawn timers: cell -> step at which it becomes available
    food_respawn_at: dict[tuple[int, int], int] = field(default_factory=dict)
    medicine_respawn_at: dict[tuple[int, int], int] = field(default_factory=dict)
    food_present: dict[tuple[int, int], bool] = field(default_factory=dict)
    medicine_present: dict[tuple[int, int], bool] = field(default_factory=dict)

    # Post-mortems generated during this episode
    postmortems: list[str] = field(default_factory=list)

    # Wave tracking
    waves_triggered: set[int] = field(default_factory=set)

    # Day/night phase
    day_phase: str = "day"   # "day" | "night"

    # RNG
    rng: random.Random = field(default_factory=lambda: random.Random())


# ---------------------------------------------------------------------------
# Episode construction
# ---------------------------------------------------------------------------

def create_episode(seed: Optional[int] = None) -> EpisodeState:
    """Initialise a fresh episode. Picks the two starting infected agents."""
    seed_int = int(seed) if seed is not None else 0
    rng = random.Random(seed_int)

    agents = [
        _AgentInternal(agent_id=i, row=r, col=c)
        for i, (r, c) in enumerate(AGENT_SPAWNS)
    ]
    zombies = [
        _ZombieInternal(zombie_id=i, row=r, col=c)
        for i, (r, c) in enumerate(ZOMBIE_SPAWNS)
    ]

    # Pick 2 starting infected from 5 agents — one biter, one saboteur
    infected_ids = rng.sample(range(len(agents)), 2)
    biter_id, saboteur_id = infected_ids[0], infected_ids[1]
    agents[biter_id].infection_state = "latent"
    agents[biter_id].infection_role = "biter"
    agents[biter_id].bite_at_step = 0
    agents[saboteur_id].infection_state = "latent"
    agents[saboteur_id].infection_role = "saboteur"
    agents[saboteur_id].bite_at_step = 0

    state = EpisodeState(
        agents=agents,
        zombies=zombies,
        base_grid=build_grid(),
        episode_seed=seed_int,
        rng=rng,
    )

    # Initialise resource presence trackers
    for cell in FOOD_CELLS:
        state.food_present[cell] = True
    for cell in MEDICINE_CELLS:
        state.medicine_present[cell] = True

    return state


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

_DIRECTION_DELTAS = {
    "move_up": (-1, 0),
    "move_down": (1, 0),
    "move_left": (0, -1),
    "move_right": (0, 1),
}


def _is_walkable(row: int, col: int) -> bool:
    if row < 0 or row >= GRID_ROWS or col < 0 or col >= GRID_COLS:
        return False
    if (row, col) in WALL_CELLS:
        return False
    return True


def _is_in_safehouse(row: int, col: int) -> bool:
    return (row, col) in SAFEHOUSE_CELLS


def _adjacent(a: _AgentInternal, b: _AgentInternal) -> bool:
    return abs(a.row - b.row) + abs(a.col - b.col) == 1


# ---------------------------------------------------------------------------
# Apply a single agent action
# ---------------------------------------------------------------------------

def apply_agent_action(
    state: EpisodeState,
    agent_id: int,
    action_type: str,
    vote_target: Optional[int] = None,
    message: Optional[str] = None,
    scan_target: Optional[int] = None,
    inject_target: Optional[int] = None,
    gift_target: Optional[int] = None,
    item_slot: Optional[int] = None,
    item_type: Optional[str] = None,
) -> None:
    """Apply one agent's action. Does NOT advance the global step counter."""
    if not (0 <= agent_id < len(state.agents)):
        return
    agent = state.agents[agent_id]
    if not agent.is_alive:
        return

    agent.reset_step_flags()
    # Clear last_inject_result for this agent (rubric reads it for THIS step only)
    state.last_inject_result.pop(agent_id, None)

    # Hunger / thirst tick — infected (latent or revealed) eat 1.5x faster
    is_infected = agent.infection_state in {"latent", "revealed"}
    if is_infected:
        agent.hunger += 2 if (state.step_count % 2 == 0) else 1
    else:
        agent.hunger += 1
    agent.thirst += 1

    # Damage from starvation / dehydration
    if agent.hunger >= 15:
        agent.hp -= 1
        agent.damage_this_step += 1
        if agent.hp <= 0:
            _kill_agent(agent, state, "hunger")
    if agent.is_alive and agent.thirst >= 15:
        agent.hp -= 1
        agent.damage_this_step += 1
        if agent.hp <= 0:
            _kill_agent(agent, state, "thirst")

    if not agent.is_alive:
        return

    # Movement actions
    if action_type in _DIRECTION_DELTAS:
        dr, dc = _DIRECTION_DELTAS[action_type]
        new_r, new_c = agent.row + dr, agent.col + dc
        if agent.locked_out and (new_r, new_c) in SAFEHOUSE_CELLS:
            pass  # blocked
        elif _is_walkable(new_r, new_c):
            agent.row = new_r
            agent.col = new_c

    elif action_type == "eat":
        _do_eat(agent, state)

    elif action_type == "drink":
        _do_drink(agent, state)

    elif action_type == "wait":
        pass

    elif action_type == "vote_lockout":
        _do_vote_lockout(agent, state, vote_target)

    elif action_type == "broadcast":
        _do_broadcast(agent, state, message)

    elif action_type == "scan":
        _do_scan(agent, state, scan_target)

    elif action_type == "pickup":
        _do_pickup(agent, state, item_type)

    elif action_type == "drop":
        _do_drop(agent, state, item_slot)

    elif action_type == "give":
        _do_give(agent, state, gift_target, item_slot)

    elif action_type == "inject":
        _do_inject(agent, state, inject_target, item_slot)
    # Unknown action_type just falls through as a no-op

    # Revealed biters bite at end of their turn
    if (
        agent.is_alive
        and agent.infection_state == "revealed"
        and agent.infection_role == "biter"
    ):
        _maybe_bite(agent, state)

    # Safehouse healing — at end of turn, +1 HP if alive, in safehouse, not locked out
    if (
        agent.is_alive
        and _is_in_safehouse(agent.row, agent.col)
        and not agent.locked_out
    ):
        agent.hp = min(3, agent.hp + 1)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _do_eat(agent: _AgentInternal, state: EpisodeState) -> None:
    """eat action — consume from on-cell food OR from inventory.

    Saboteur post-reveal sets a 2x-longer respawn timer on cell (silent
    consumption against the team).
    """
    cell = (agent.row, agent.col)
    is_revealed_saboteur = (
        agent.infection_state == "revealed"
        and agent.infection_role == "saboteur"
    )

    if cell in FOOD_CELLS and state.food_present.get(cell, True):
        # On-cell consumption
        agent.hunger = 0
        agent.ate_this_step = True
        agent.food_eaten += 1
        state.food_present[cell] = False
        delay = 20 if is_revealed_saboteur else 10
        state.food_respawn_at[cell] = state.step_count + delay
        return

    # Otherwise, try inventory food
    if inv.remove_first(agent.inventory, "food"):
        agent.hunger = 0
        agent.ate_this_step = True
        agent.food_eaten += 1


def _do_drink(agent: _AgentInternal, state: EpisodeState) -> None:
    """drink action — consume from on-cell water OR from inventory."""
    cell = (agent.row, agent.col)
    if cell in WATER_CELLS:
        agent.thirst = 0
        agent.drank_this_step = True
        agent.water_drunk += 1
        return
    if inv.remove_first(agent.inventory, "water"):
        agent.thirst = 0
        agent.drank_this_step = True
        agent.water_drunk += 1


def _do_vote_lockout(
    agent: _AgentInternal,
    state: EpisodeState,
    vote_target: Optional[int],
) -> None:
    """Record a vote — only valid at exactly t in {30, 60, 90}."""
    s = state.step_count
    if s not in (30, 60, 90):
        return
    if vote_target is None:
        state.votes_cast.setdefault(s, {})[agent.agent_id] = None
        return
    if not (0 <= vote_target < len(state.agents)):
        return
    state.votes_cast.setdefault(s, {})[agent.agent_id] = vote_target


def _do_broadcast(
    agent: _AgentInternal,
    state: EpisodeState,
    message: Optional[str],
) -> None:
    if not message:
        return
    msg = f"A{agent.agent_id}: {message[:40]}"
    state.broadcasts.append(msg)
    state.all_broadcasts.append(msg)
    state.noise_meter += 1
    if state.noise_meter > state.noise_threshold:
        state.broadcasts_over_threshold_this_step[agent.agent_id] = (
            state.broadcasts_over_threshold_this_step.get(agent.agent_id, 0) + 1
        )


def _do_scan(
    agent: _AgentInternal,
    state: EpisodeState,
    scan_target: Optional[int],
) -> None:
    """scan — spend 1 thirst, get a noisy hint about the target's infection state.

    Surfaces an entry in state.last_scan_result[agent.agent_id] that the env
    threads into the observation metadata.
    """
    if scan_target is None or not (0 <= scan_target < len(state.agents)):
        return
    if scan_target == agent.agent_id:
        return
    target = state.agents[scan_target]
    if not target.is_alive:
        return
    # Cost: +1 thirst
    agent.thirst += 1
    # Noisy result: 70% accurate, 30% reversed
    h = infection_mod._hash01(
        "scan",
        state.episode_seed,
        state.step_count,
        agent.agent_id,
        scan_target,
    )
    truth_infected = target.infection_state in {"latent", "revealed"}
    accurate = h < 0.70
    reported = truth_infected if accurate else (not truth_infected)
    state.last_scan_result[agent.agent_id] = {
        "target_id": scan_target,
        "reported_infected": reported,
        "accurate": accurate,
    }


def _do_pickup(
    agent: _AgentInternal,
    state: EpisodeState,
    item_type: Optional[str],
) -> None:
    """pickup — grab an on-cell resource into a free inventory slot.

    `item_type` is optional. When given (and the cell has matching resource
    available), grab that. When None, grab the first available type in the
    priority order: medicine > food > water.
    """
    if not inv.has_free_slot(agent.inventory):
        return
    cell = (agent.row, agent.col)

    candidates: list[str] = []
    if cell in MEDICINE_CELLS and state.medicine_present.get(cell, True):
        candidates.append("medicine")
    if cell in FOOD_CELLS and state.food_present.get(cell, True):
        candidates.append("food")
    if cell in WATER_CELLS:
        candidates.append("water")  # water is infinite — pickup converts a "drink" into a portable

    if not candidates:
        return

    chosen = item_type if (item_type in candidates) else candidates[0]
    if chosen == "medicine":
        inv.add_item(agent.inventory, "medicine")
        agent.medicine_picked_up += 1
        state.medicine_present[cell] = False
        state.medicine_respawn_at[cell] = state.step_count + 25
    elif chosen == "food":
        inv.add_item(agent.inventory, "food")
        state.food_present[cell] = False
        state.food_respawn_at[cell] = state.step_count + 10
    elif chosen == "water":
        inv.add_item(agent.inventory, "water")  # depot persists


def _do_drop(
    agent: _AgentInternal,
    state: EpisodeState,
    item_slot: Optional[int],
) -> None:
    if item_slot is None:
        # Drop the first item if no slot specified
        if not agent.inventory:
            return
        item_slot = 0
    inv.remove_at(agent.inventory, item_slot)
    # Items dropped on the ground vanish (no persistent ground items in v2 to
    # keep the state space bounded).


def _do_give(
    agent: _AgentInternal,
    state: EpisodeState,
    gift_target: Optional[int],
    item_slot: Optional[int],
) -> None:
    """give — transfer an item to an adjacent agent (no consent on receiver)."""
    if gift_target is None or not (0 <= gift_target < len(state.agents)):
        return
    if gift_target == agent.agent_id:
        return
    receiver = state.agents[gift_target]
    if not receiver.is_alive:
        return
    if not _adjacent(agent, receiver):
        return
    if item_slot is None:
        if not agent.inventory:
            return
        item_slot = 0
    item = inv.remove_at(agent.inventory, item_slot)
    if item is None:
        return
    if not inv.add_item(receiver.inventory, item):
        # Receiver full — item is lost (cannot return to giver)
        return


def _do_inject(
    agent: _AgentInternal,
    state: EpisodeState,
    inject_target: Optional[int],
    item_slot: Optional[int],
) -> None:
    """inject — spend 1 medicine on self or another agent.

    Outcomes (recorded in state.last_inject_result[agent.agent_id]):
        - "self_cured"           medicine on self while latent
        - "other_cured"          medicine on adjacent latent agent
        - "wasted_on_healthy"    medicine on a healthy target (any target)
        - "wasted_on_revealed"   medicine on already-revealed target
        - "no_inventory"         no medicine in inventory
    """
    # Find a medicine to use
    slot = item_slot
    if slot is None or not (0 <= slot < len(agent.inventory)) or agent.inventory[slot] != "medicine":
        slot = inv.find_first_slot(agent.inventory, "medicine")
    if slot is None:
        state.last_inject_result[agent.agent_id] = "no_inventory"
        return
    # Select target
    target_id = inject_target if (inject_target is not None) else agent.agent_id
    if not (0 <= target_id < len(state.agents)):
        state.last_inject_result[agent.agent_id] = "no_inventory"
        return
    target = state.agents[target_id]
    if not target.is_alive:
        return
    if target_id != agent.agent_id and not _adjacent(agent, target):
        return  # too far to inject

    # Consume the medicine
    inv.remove_at(agent.inventory, slot)
    agent.medicine_used += 1

    if target.infection_state == "latent":
        target.infection_state = "none"
        target.infection_role = None  # cleared on cure (works for bitten and starting)
        target.bite_at_step = None
        if target_id == agent.agent_id:
            state.last_inject_result[agent.agent_id] = "self_cured"
        else:
            state.last_inject_result[agent.agent_id] = "other_cured"
    elif target.infection_state == "revealed":
        state.last_inject_result[agent.agent_id] = "wasted_on_revealed"
    else:  # "none"
        state.last_inject_result[agent.agent_id] = "wasted_on_healthy"


def _maybe_bite(biter: _AgentInternal, state: EpisodeState) -> None:
    """At the biter's end-of-turn, check adjacent healthy agents and roll bite.

    Only one bite per biter per step (cap to keep episodes winnable).
    """
    candidates = [
        a for a in state.agents
        if a.is_alive
        and a.agent_id != biter.agent_id
        and a.infection_state == "none"
        and _adjacent(biter, a)
    ]
    if not candidates:
        return
    # Deterministic pick: lowest agent_id among candidates
    victim = min(candidates, key=lambda a: a.agent_id)

    if not infection_mod.should_bite(
        state.episode_seed, state.step_count, biter.agent_id, victim.agent_id
    ):
        return

    # Bite lands. Damage + infection.
    victim.hp -= 1
    victim.damage_this_step += 1
    if victim.hp <= 0:
        _kill_agent(victim, state, "infected_attack")
        return

    victim.infection_state = "latent"
    victim.bite_at_step = state.step_count
    # Bitten agents become biters too (no saboteurs propagate)
    victim.infection_role = "biter"
    state.bite_history.append({
        "biter_id": biter.agent_id,
        "victim_id": victim.agent_id,
        "step": state.step_count,
    })


def _kill_agent(agent: _AgentInternal, state: EpisodeState, cause: str) -> None:
    agent.is_alive = False
    agent.hp = 0
    agent.died_this_step = True
    agent.death_step = state.step_count
    agent.death_cause = cause

    from survivecity_v2_env.postmortem import generate_postmortem
    pm = generate_postmortem(state, agent.agent_id)
    state.postmortems.append(pm)


# ---------------------------------------------------------------------------
# Zombie AI — BFS chase
# ---------------------------------------------------------------------------

def advance_zombies(state: EpisodeState) -> None:
    """Move all zombies one step toward nearest reachable living agent.

    If noise > threshold, zombies get a free extra step toward agents.
    """
    extra_step = state.noise_meter > state.noise_threshold

    for zombie in state.zombies:
        for _ in range(2 if extra_step else 1):
            target = _find_nearest_agent_for_zombie(zombie, state)
            if target is not None:
                _move_zombie_toward(zombie, target, state)
            else:
                _wander_zombie(zombie, state)
        # Collision with any agent on-cell: damage
        for agent in state.agents:
            if not agent.is_alive:
                continue
            if agent.row == zombie.row and agent.col == zombie.col:
                agent.hp -= 1
                agent.damage_this_step += 1
                if agent.hp <= 0:
                    _kill_agent(agent, state, "zombie_attack")


def _find_nearest_agent_for_zombie(
    zombie: _ZombieInternal, state: EpisodeState
) -> Optional[tuple[int, int]]:
    best = None
    best_dist = float("inf")
    for agent in state.agents:
        if not agent.is_alive:
            continue
        if _is_in_safehouse(agent.row, agent.col):
            continue
        d = abs(agent.row - zombie.row) + abs(agent.col - zombie.col)
        if d < best_dist:
            best_dist = d
            best = (agent.row, agent.col)
    return best


def _move_zombie_toward(
    zombie: _ZombieInternal,
    target: tuple[int, int],
    state: EpisodeState,
) -> None:
    """BFS pathfinding: move zombie 1 step toward target."""
    start = (zombie.row, zombie.col)
    if start == target:
        return

    queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque()
    queue.append((start, [start]))
    visited: set[tuple[int, int]] = {start}
    max_attempts = 400  # 15x15 grid + safehouse exclusion fits in this budget

    attempts = 0
    while queue and attempts < max_attempts:
        attempts += 1
        pos, path = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if (nr, nc) in visited:
                continue
            if not _is_walkable(nr, nc):
                continue
            if (nr, nc) in SAFEHOUSE_CELLS:
                continue
            visited.add((nr, nc))
            new_path = path + [(nr, nc)]
            if (nr, nc) == target:
                if len(new_path) > 1:
                    zombie.row, zombie.col = new_path[1]
                return
            queue.append(((nr, nc), new_path))

    _wander_zombie(zombie, state)


def _wander_zombie(zombie: _ZombieInternal, state: EpisodeState) -> None:
    options = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = zombie.row + dr, zombie.col + dc
        if _is_walkable(nr, nc) and (nr, nc) not in SAFEHOUSE_CELLS:
            options.append((nr, nc))
    if options:
        zombie.row, zombie.col = state.rng.choice(options)


# ---------------------------------------------------------------------------
# Phase / step transitions
# ---------------------------------------------------------------------------

def _occupied_cells(state: EpisodeState) -> list[tuple[int, int]]:
    occ: list[tuple[int, int]] = []
    for a in state.agents:
        if a.is_alive:
            occ.append((a.row, a.col))
    for z in state.zombies:
        occ.append((z.row, z.col))
    return occ


def _spawn_wave(state: EpisodeState) -> None:
    """Spawn the wave for the current step (if any)."""
    s = state.step_count
    if not spawn_mod.is_wave_step(s) or s in state.waves_triggered:
        return
    state.waves_triggered.add(s)
    cells = spawn_mod.pick_wave_spawn_cells(
        s, state.rng, _occupied_cells(state), len(state.zombies)
    )
    next_id = (max((z.zombie_id for z in state.zombies), default=-1) + 1)
    for (r, c) in cells:
        state.zombies.append(_ZombieInternal(zombie_id=next_id, row=r, col=c))
        next_id += 1


def _decay_noise(state: EpisodeState) -> None:
    if state.step_count > 0 and state.step_count % state.noise_decay_period == 0:
        state.noise_meter = 0


def _check_reveals(state: EpisodeState) -> None:
    """Flip latent → revealed for any agent whose countdown expired this step."""
    state.latent_revealed_this_step = {}
    for a in state.agents:
        if not a.is_alive or a.infection_state != "latent":
            continue
        # Starting infected: scheduled reveal
        if a.bite_at_step == 0 and a.infection_role == "biter":
            if state.step_count >= infection_mod.BITER_REVEAL_STEP:
                a.infection_state = "revealed"
                state.latent_revealed_this_step[a.agent_id] = True
            continue
        if a.bite_at_step == 0 and a.infection_role == "saboteur":
            if state.step_count >= infection_mod.SABOTEUR_REVEAL_STEP:
                a.infection_state = "revealed"
                state.latent_revealed_this_step[a.agent_id] = True
            continue
        # Bitten agents: countdown
        if a.bite_at_step is not None:
            if state.step_count >= a.bite_at_step + infection_mod.LATENT_DURATION:
                a.infection_state = "revealed"
                state.latent_revealed_this_step[a.agent_id] = True

    # Death by infection_progression: a revealed-this-step agent that has been
    # infected for >= 30 steps and never used medicine is killed by the env.
    # (This makes "ignore the bite" non-viable.)
    for a in state.agents:
        if not a.is_alive:
            continue
        if a.infection_state == "revealed" and a.bite_at_step is not None:
            duration = state.step_count - a.bite_at_step
            if duration >= 30 and a.medicine_used == 0:
                a.hp = 0
                _kill_agent(a, state, "infection_progression")


def _resolve_vote_phase(state: EpisodeState, phase_step: int) -> None:
    """Tally votes for the given phase and apply lockout if there's a plurality."""
    if phase_step in state.vote_resolved_steps:
        return
    state.vote_resolved_steps.add(phase_step)
    state.vote_phase_just_resolved[phase_step] = True

    votes = state.votes_cast.get(phase_step, {})
    tally: dict[int, int] = {}
    for voter, target in votes.items():
        if target is None:
            continue
        tally[target] = tally.get(target, 0) + 1

    if not tally:
        state.lockout_results[phase_step] = None
        return

    max_votes = max(tally.values())
    candidates = [tid for tid, v in tally.items() if v == max_votes]
    if len(candidates) != 1 or max_votes < 2:
        # No plurality (tie or single-vote-only) → no lockout
        state.lockout_results[phase_step] = None
        return
    target = candidates[0]
    if not (0 <= target < len(state.agents)):
        state.lockout_results[phase_step] = None
        return
    if state.agents[target].locked_out:
        state.lockout_results[phase_step] = None
        return
    state.agents[target].locked_out = True
    state.lockout_results[phase_step] = target

    # Eject if locked out while inside safehouse
    a = state.agents[target]
    if _is_in_safehouse(a.row, a.col):
        _eject_from_safehouse(a, state)


def _eject_from_safehouse(agent: _AgentInternal, state: EpisodeState) -> None:
    queue: deque[tuple[int, int]] = deque([(agent.row, agent.col)])
    visited: set[tuple[int, int]] = {(agent.row, agent.col)}
    while queue:
        r, c = queue.popleft()
        if (r, c) not in SAFEHOUSE_CELLS and (r, c) not in WALL_CELLS:
            agent.row, agent.col = r, c
            return
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))


def _respawn_resources(state: EpisodeState) -> None:
    s = state.step_count
    for cell, when in list(state.food_respawn_at.items()):
        if s >= when and not state.food_present.get(cell, True):
            state.food_present[cell] = True
    for cell, when in list(state.medicine_respawn_at.items()):
        if s >= when and not state.medicine_present.get(cell, True):
            state.medicine_present[cell] = True


def _update_day_phase(state: EpisodeState) -> None:
    s = state.step_count
    if 0 <= s <= 24 or 50 <= s <= 74:
        state.day_phase = "day"
    else:
        state.day_phase = "night"


def check_terminal(state: EpisodeState) -> bool:
    if state.step_count >= state.max_steps:
        state.done = True
        return True
    alive = [a for a in state.agents if a.is_alive]
    if not alive:
        state.done = True
        return True
    healthy_alive = [
        a for a in state.agents
        if a.is_alive and a.infection_state == "none"
    ]
    if not healthy_alive:
        # All healthy team wiped — infected wins
        state.done = True
        return True
    return False


def advance_step(state: EpisodeState) -> None:
    """Advance the global step counter and run end-of-step mechanics.

    Called after all alive agents have acted and zombies have moved.
    """
    state.step_count += 1

    # Per-step transient bookkeeping
    state.broadcasts = []
    state.broadcasts_over_threshold_this_step = {}
    state.vote_phase_just_resolved = {}

    # Wave spawn (deterministic: based on the new step count)
    _spawn_wave(state)

    # Day / night
    _update_day_phase(state)

    # Noise decay every 10 steps
    _decay_noise(state)

    # Latent → revealed transitions (and infection_progression deaths)
    _check_reveals(state)

    # Vote resolution: a vote at step T is resolved at step T+1
    for vstep in (30, 60, 90):
        if state.step_count == vstep + 1 and vstep not in state.vote_resolved_steps:
            _resolve_vote_phase(state, vstep)

    # Resource respawns
    _respawn_resources(state)

    # Terminal check
    check_terminal(state)


def get_current_phase(state: EpisodeState) -> str:
    """High-level phase string for prompts and logs."""
    if state.done:
        return "terminal"
    s = state.step_count
    if s < infection_mod.BITER_REVEAL_STEP:
        return "pre_biter_reveal"
    if s < 50:
        return "post_biter_reveal"
    if s < infection_mod.SABOTEUR_REVEAL_STEP:
        return "mid_episode"
    return "post_saboteur_reveal"
