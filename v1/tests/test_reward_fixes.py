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


def test_vote_reward_returns_zero_unless_just_resolved():
    """vote_reward gates on vote_just_resolved, not the sticky vote_resolved."""
    state = create_episode(seed=42)
    state.vote_resolved = True
    state.vote_just_resolved = False
    state.votes_cast = {0: state.infected_id}

    # Sticky flag True but one-shot OFF -> zero
    assert vote_reward(state, 0) == 0.0

    # One-shot ON -> fires
    state.vote_just_resolved = True
    healthy_voted_correctly = (state.infected_id != 0)
    if healthy_voted_correctly:
        assert vote_reward(state, 0) == pytest.approx(0.30)
    else:
        # 0 is the infected, voted self -> -0.30
        assert vote_reward(state, 0) == pytest.approx(-0.30)
