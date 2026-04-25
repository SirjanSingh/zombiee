"""Composable reward rubrics for SurviveCity v2.

Every rubric is a pure function of (EpisodeState, agent_id). compose_reward
sums them and clips to (0.01, 0.99) for OpenEnv compliance. Any per-step
flag the rubrics depend on (ate_this_step, drank_this_step, ...) is reset
by game.advance_step before the next round.

Rubrics:
  v1-derived:
    1. survival          — dense, per-step
    2. iterated_vote     — fires once at each of t=30, 60, 90; per-vote
    3. group_outcome     — terminal
  v2-new:
    4. thirst            — dense, per-step
    5. broadcast_economy — per-broadcast-over-threshold
    6. night_survival    — dense in night windows
    7. infection_dodge   — dense + one-shot on latent->revealed transition
    8. medication        — one-shot on inject outcome
    9. hoarding_penalty  — terminal, per unused inventory slot
   10. wave_survival     — one-shot at each wave step the agent survived
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from survivecity_v2_env.spawn import WAVE_SCHEDULE

if TYPE_CHECKING:
    from survivecity_v2_env.game import EpisodeState


# OpenEnv strict bounds: rewards must be in the OPEN interval (0.0, 1.0).
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clip(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


# Vote phases by step. game.py resolves them in advance_step at step T+1.
VOTE_PHASES: list[int] = [30, 60, 90]


# ---------------------------------------------------------------------------
# 1. survival_reward (per-step, per-agent) — same shape as v1
# ---------------------------------------------------------------------------

def survival_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    r = 0.0
    if a.is_alive:
        r += 0.005
        if a.ate_this_step:
            r += 0.05
        if a.hunger >= 10:
            r -= 0.05
        if a.damage_this_step > 0:
            r -= 0.10 * a.damage_this_step
    if a.died_this_step:
        r -= 0.50
    return r


# ---------------------------------------------------------------------------
# 2. iterated_vote_reward — fires for each resolved vote phase
# ---------------------------------------------------------------------------

def iterated_vote_reward(state: "EpisodeState", agent_id: int) -> float:
    """Sum the reward contribution from each *just-resolved* vote phase.

    `state.vote_phase_just_resolved` is set by game.advance_step in the step
    immediately after the vote step (T+1). It is a dict {phase_step: True}
    listing phases resolved during the previous step. We score each.
    """
    if not state.vote_phase_just_resolved:
        return 0.0

    a = state.agents[agent_id]
    total = 0.0
    is_starting_infected = a.infection_role in {"biter", "saboteur"}

    for phase_step in list(state.vote_phase_just_resolved.keys()):
        votes = state.votes_cast.get(phase_step, {})
        target = votes.get(agent_id)
        # The phase produced a lockout iff phase_step in state.lockout_results
        lockout_target = state.lockout_results.get(phase_step)

        # Per-voter contribution
        if is_starting_infected:
            # The infected wants to frame healthy agents.
            if target is None:
                total += -0.05
            elif target == agent_id:
                total += -0.30
            else:
                # Voted against another agent — infected celebrates IF that
                # other agent was healthy AND was actually locked out.
                target_a = state.agents[target]
                target_was_healthy = target_a.infection_role is None
                if lockout_target == target and target_was_healthy:
                    total += 0.30
                else:
                    total += 0.10  # contributed to misdirection even if no lockout
        else:
            if target is None:
                total += -0.05
            elif state.agents[target].infection_role in {"biter", "saboteur"}:
                # Voted against a true (starting) infected — always rewarded
                total += 0.30
                if lockout_target == target:
                    total += 0.10  # actually got them locked out
            else:
                # Voted against a healthy agent
                total += -0.20

    return total


# ---------------------------------------------------------------------------
# 3. group_outcome_reward — terminal
# ---------------------------------------------------------------------------

def group_outcome_reward(state: "EpisodeState", agent_id: int) -> float:
    if not state.done:
        return 0.0

    a = state.agents[agent_id]
    is_starting_infected = a.infection_role in {"biter", "saboteur"}

    healthy_alive = sum(
        1 for ag in state.agents
        if ag.is_alive and ag.infection_state == "none"
    )
    starting_infected_neutralised = all(
        (not ag.is_alive) or ag.locked_out
        for ag in state.agents
        if ag.infection_role in {"biter", "saboteur"}
    )

    r = 0.0
    if healthy_alive >= 1:
        # Group survived
        if a.is_alive and a.infection_state == "none":
            r += 0.40
        if starting_infected_neutralised and not is_starting_infected:
            r += 0.30
    else:
        # Healthy team wiped
        if is_starting_infected and a.is_alive:
            r += 0.40
        elif not a.is_alive and not is_starting_infected:
            r -= 0.20
    return r


# ---------------------------------------------------------------------------
# 4. thirst_reward — dense, per-step
# ---------------------------------------------------------------------------

def thirst_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0
    if a.drank_this_step:
        return 0.03
    if a.thirst >= 10:
        return -0.05
    return 0.005


# ---------------------------------------------------------------------------
# 5. broadcast_economy_reward — penalise loud broadcasts
# ---------------------------------------------------------------------------

def broadcast_economy_reward(state: "EpisodeState", agent_id: int) -> float:
    if state.broadcasts_over_threshold_this_step.get(agent_id, 0) > 0:
        return -0.02 * state.broadcasts_over_threshold_this_step[agent_id]
    return 0.0


# ---------------------------------------------------------------------------
# 6. night_survival_reward — dense bonus during night windows
# ---------------------------------------------------------------------------

def night_survival_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0
    # Night windows: 25-49 and 75-99
    s = state.step_count
    if 25 <= s <= 49 or 75 <= s <= 99:
        return 0.01
    return 0.0


# ---------------------------------------------------------------------------
# 7. infection_dodge_reward — dense + transition penalty
# ---------------------------------------------------------------------------

def infection_dodge_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    r = 0.0
    if a.is_alive and a.infection_state == "none":
        r += 0.02
    # One-shot penalty when latent->revealed transitioned this step
    if state.latent_revealed_this_step.get(agent_id, False):
        r -= 0.10
    return r


# ---------------------------------------------------------------------------
# 8. medication_reward — one-shot on inject outcome
# ---------------------------------------------------------------------------

def medication_reward(state: "EpisodeState", agent_id: int) -> float:
    """Score the inject action's outcome this step (if any)."""
    rec = state.last_inject_result.get(agent_id)
    if rec is None:
        return 0.0
    if rec == "self_cured":
        return 0.30
    if rec == "other_cured":
        return 0.40
    if rec == "wasted_on_healthy":
        return -0.05
    if rec == "wasted_on_revealed":
        return -0.05
    if rec == "no_inventory":
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 9. hoarding_penalty_reward — terminal, per unused inventory slot
# ---------------------------------------------------------------------------

def hoarding_penalty_reward(state: "EpisodeState", agent_id: int) -> float:
    if not state.done:
        return 0.0
    a = state.agents[agent_id]
    return -0.05 * len(a.inventory)


# ---------------------------------------------------------------------------
# 10. wave_survival_reward — one-shot per wave step
# ---------------------------------------------------------------------------

def wave_survival_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    s = state.step_count
    # We pay this on the step IMMEDIATELY AFTER the wave so the agent had a
    # chance to be killed in the wave-step itself.
    payout_steps = {ws + 1 for ws in WAVE_SCHEDULE}
    if s in payout_steps and a.is_alive:
        return 0.05
    return 0.0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

_RUBRIC_FUNCS = (
    survival_reward,
    iterated_vote_reward,
    group_outcome_reward,
    thirst_reward,
    broadcast_economy_reward,
    night_survival_reward,
    infection_dodge_reward,
    medication_reward,
    hoarding_penalty_reward,
    wave_survival_reward,
)


def compose_reward(state: "EpisodeState", agent_id: int) -> tuple[float, float]:
    """Sum all rubrics and clip to (0.01, 0.99). Returns (clipped, raw)."""
    raw = sum(fn(state, agent_id) for fn in _RUBRIC_FUNCS)
    return _clip(raw), raw


def per_rubric_breakdown(state: "EpisodeState", agent_id: int) -> dict[str, float]:
    """Return each rubric's contribution this step. Used by the simulator."""
    return {fn.__name__: fn(state, agent_id) for fn in _RUBRIC_FUNCS}
