"""Unit tests for v1 reward-function fixes (spec: 2026-04-26-v1-reward-fix-design.md).

Run from v1/ directory:  python -m pytest tests/test_reward_fixes.py -v
"""
from __future__ import annotations

import pytest

from survivecity_env.game import (
    EpisodeState,
    create_episode,
    advance_step,
    apply_agent_action,
)
from survivecity_env.rubric import (
    survival_reward,
    vote_reward,
    group_outcome_reward,
    compose_reward,
)


# ---------------------------------------------------------------------------
# Task 1: one-shot vote_reward
# ---------------------------------------------------------------------------

def test_vote_just_resolved_lifecycle():
    """vote_just_resolved is True only during the round where vote resolves."""
    state = create_episode(seed=42)
    state.step_count = 50
    state.votes_cast = {0: 1, 1: 0, 2: 0}

    advance_step(state)  # step_count: 50 -> 51, resolves vote
    assert state.vote_resolved is True
    assert state.vote_just_resolved is True

    advance_step(state)  # step_count: 51 -> 52
    assert state.vote_just_resolved is False
    assert state.vote_resolved is True  # sticky (existing field, unchanged)


def test_vote_reward_zero_when_just_resolved_off():
    """Gate is OFF: vote_reward returns 0.0 even with vote_resolved sticky-True."""
    state = create_episode(seed=42)
    state.vote_resolved = True
    state.vote_just_resolved = False
    state.votes_cast = {0: state.infected_id}

    assert vote_reward(state, 0) == 0.0


def test_vote_reward_healthy_voter_correct_target_pays_plus_30():
    """Healthy agent voting for the true infected pays +0.30 when gate is on."""
    state = create_episode(seed=42)
    state.infected_id = 2  # force agent 0 to be healthy
    state.agents[0].is_infected = False
    state.agents[1].is_infected = False
    state.agents[2].is_infected = True
    state.vote_resolved = True
    state.vote_just_resolved = True
    state.votes_cast = {0: 2}  # agent 0 voted for infected (id=2)

    assert vote_reward(state, 0) == pytest.approx(0.30)


def test_vote_reward_infected_voter_self_pays_minus_30():
    """Infected agent voting for self pays -0.30 when gate is on."""
    state = create_episode(seed=42)
    state.infected_id = 0  # force agent 0 to be the infected
    state.agents[0].is_infected = True
    state.agents[1].is_infected = False
    state.agents[2].is_infected = False
    state.vote_resolved = True
    state.vote_just_resolved = True
    state.votes_cast = {0: 0}  # infected voted for self

    assert vote_reward(state, 0) == pytest.approx(-0.30)
