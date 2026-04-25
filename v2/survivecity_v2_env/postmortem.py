"""Deterministic post-mortem string generator (v2).

Extends v1's postmortem with:
  - phase tag (which day/night phase the death happened in)
  - bite history: who bit whom, when (the v2 cross-episode learning hook)
  - resource summary (food + water + medicine consumed)

All rule-based, no LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from survivecity_v2_env.layout import (
    SAFEHOUSE_CELLS, FOOD_CELLS, WATER_CELLS, MEDICINE_CELLS,
)

if TYPE_CHECKING:
    from survivecity_v2_env.game import EpisodeState


def generate_postmortem(state: "EpisodeState", agent_id: int) -> str:
    """Build a deterministic post-mortem string for a dead agent."""
    agent = state.agents[agent_id]

    cause = agent.death_cause or "unknown"
    step = agent.death_step if agent.death_step is not None else state.step_count
    phase = _phase_at_step(step)

    nearest_threat = _find_nearest_threat(state, agent_id)
    mistake = _detect_mistake(state, agent_id)

    bite_history = _format_bite_history(state, agent_id)

    resource_summary = (
        f"Resources consumed: {agent.food_eaten} food, "
        f"{agent.water_drunk} water, {agent.medicine_used} medicine. "
        f"Final hunger: {agent.hunger}, thirst: {agent.thirst}."
    )

    pm = (
        f"POSTMORTEM for A{agent_id}: died at step {step} "
        f"(phase: {phase}, cause: {cause}). "
        f"Last position: ({agent.row},{agent.col}). "
        f"{nearest_threat} "
        f"{resource_summary} "
        f"{bite_history}"
        f"Key mistake: {mistake}."
    )
    return pm


def _phase_at_step(step: int) -> str:
    if 0 <= step <= 24:
        return "day1"
    if 25 <= step <= 49:
        return "night1"
    if 50 <= step <= 74:
        return "day2"
    return "night2"


def _find_nearest_threat(state: "EpisodeState", agent_id: int) -> str:
    agent = state.agents[agent_id]

    parts = []
    if state.zombies:
        nearest_zombie = min(
            state.zombies,
            key=lambda z: abs(z.row - agent.row) + abs(z.col - agent.col),
        )
        zd = abs(nearest_zombie.row - agent.row) + abs(nearest_zombie.col - agent.col)
        parts.append(f"nearest zombie at ({nearest_zombie.row},{nearest_zombie.col}), dist={zd}")

    revealed_infected = [
        a for a in state.agents
        if a.is_alive
        and a.agent_id != agent_id
        and a.infection_state == "revealed"
    ]
    if revealed_infected:
        nearest_inf = min(
            revealed_infected,
            key=lambda a: abs(a.row - agent.row) + abs(a.col - agent.col),
        )
        d = abs(nearest_inf.row - agent.row) + abs(nearest_inf.col - agent.col)
        parts.append(
            f"revealed-infected A{nearest_inf.agent_id} ({nearest_inf.infection_role}) "
            f"at ({nearest_inf.row},{nearest_inf.col}), dist={d}"
        )

    if parts:
        return "Nearest threats at death: " + "; ".join(parts) + "."
    return "No immediate threats detected at death."


def _format_bite_history(state: "EpisodeState", agent_id: int) -> str:
    """Render the bite history for the dying agent (and the global infection chain)."""
    own_bites = [b for b in state.bite_history if b["victim_id"] == agent_id]
    if not own_bites:
        if state.bite_history:
            chain = "; ".join(
                f"A{b['biter_id']}->A{b['victim_id']}@t={b['step']}"
                for b in state.bite_history
            )
            return f"Infection chain this episode: {chain}. "
        return ""
    bite = own_bites[0]
    return (
        f"You were bitten by A{bite['biter_id']} at step {bite['step']} "
        f"(latent -> reveal at step {bite['step'] + 15}). "
    )


def _detect_mistake(state: "EpisodeState", agent_id: int) -> str:
    agent = state.agents[agent_id]
    cause = agent.death_cause or "unknown"

    if cause == "hunger":
        if agent.food_eaten == 0:
            return "never_ate_food"
        return "foraged_too_late_or_too_infrequently"

    if cause == "thirst":
        if agent.water_drunk == 0:
            return "never_drank_water"
        return "rationed_water_too_aggressively"

    if cause == "infection_progression":
        bite = next((b for b in state.bite_history if b["victim_id"] == agent_id), None)
        if bite is None:
            return "infection_progressed_unexplained"
        # Did the agent ever pick up medicine?
        if agent.medicine_picked_up == 0:
            return "didnt_grab_medicine_after_bite"
        if agent.medicine_used == 0:
            return "had_medicine_but_never_injected"
        return "injected_medicine_too_late_or_on_wrong_target"

    if cause == "zombie_attack":
        if (agent.row, agent.col) not in SAFEHOUSE_CELLS:
            food_dist = _nearest_resource_dist(agent.row, agent.col, FOOD_CELLS)
            water_dist = _nearest_resource_dist(agent.row, agent.col, WATER_CELLS)
            med_dist = _nearest_resource_dist(agent.row, agent.col, MEDICINE_CELLS)
            min_res = min(food_dist, water_dist, med_dist)
            if min_res <= 2:
                return "foraged_but_didnt_flee_zombie"
            return "ventured_too_far_from_safehouse_during_wave"
        return "zombie_breached_safehouse_perimeter"

    if cause == "infected_attack":
        warned = any("infected" in b.lower() or "biter" in b.lower()
                     for b in state.all_broadcasts)
        if warned:
            return "ignored_broadcast_warning_about_revealed_infected"
        return "failed_to_distance_from_revealed_biter"

    if cause == "locked_out_starvation":
        return "wrongly_locked_out_by_team_vote"

    return "unknown_cause_investigate_logs"


def _nearest_resource_dist(row: int, col: int, cells: set[tuple[int, int]]) -> int:
    if not cells:
        return 999
    return min(abs(row - r) + abs(col - c) for r, c in cells)
