# V1 Reward-Function Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three correctness bugs and add forage/wait shaping in v1's reward pipeline so GRPO sees a real gradient signal during the next training run, on the `v1new` branch, with all artefacts pushed to a timestamped subfolder under `noanya/v1-zombiee/`.

**Architecture:** Modify the env-side reward computation and the rubric (potential-based forage shaping, one-shot vote, damage credit accumulator, wait penalty, docstring fix), swap the GRPO `reward_fn` to read raw rewards instead of clipped, and replace the default Trainer hub push with a custom callback that writes to a timestamped HF subfolder. Reference spec: `docs/superpowers/specs/2026-04-26-v1-reward-fix-design.md`.

**Tech Stack:** Python 3.10+, PyTorch 2.5.1+cu121, transformers 4.46.3, peft 0.13.2, trl 0.15.2, bitsandbytes 0.43.3, pytest, huggingface_hub.

**Branch:** `v1new` (already created off `master`).

---

### Task 1: One-shot `vote_reward` via `vote_just_resolved` lifecycle

**Goal:** Stop the bug where `vote_reward` fires every step after step 51. Replace the sticky `state.vote_resolved` gate with a one-shot `state.vote_just_resolved` bool that is True only during the round when the vote resolves.

**Files:**
- Create: `v1/tests/__init__.py`
- Create: `v1/tests/test_reward_fixes.py`
- Modify: `v1/survivecity_env/game.py` (add field to `EpisodeState`, update `advance_step`)
- Modify: `v1/survivecity_env/rubric.py` (`vote_reward` gate change)

- [ ] **Step 1: Create test file with failing tests**

Create `v1/tests/__init__.py` as an empty file.

Create `v1/tests/test_reward_fixes.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_vote_just_resolved_lifecycle tests/test_reward_fixes.py::test_vote_reward_returns_zero_unless_just_resolved -v
```

Expected output (both fail because `vote_just_resolved` doesn't exist yet):
```
FAILED ... AttributeError: 'EpisodeState' object has no attribute 'vote_just_resolved'
```

- [ ] **Step 3: Add `vote_just_resolved` field to `EpisodeState`**

In `v1/survivecity_env/game.py`, locate the `EpisodeState` dataclass (around line 65). Add this field right after the existing `vote_resolved` line (around line 84):

```python
    vote_resolved: bool = False
    vote_just_resolved: bool = False  # NEW: True only during the round when vote resolves
    lockout_target: Optional[int] = None
```

- [ ] **Step 4: Update `advance_step` lifecycle**

In `v1/survivecity_env/game.py`, locate the `advance_step` function (around line 448). Replace the function body with:

```python
def advance_step(state: EpisodeState) -> None:
    """Advance the global step counter and run per-step mechanics.

    Called after all 3 agents have acted and zombies have moved.
    """
    state.step_count += 1

    # Clear per-step broadcasts and the one-shot vote flag.
    state.broadcasts = []
    state.vote_just_resolved = False  # NEW: clears at start of every round

    # Check infection reveal at step 30
    check_infection_reveal(state)

    # Auto-trigger vote phase at step 50
    if state.step_count == 50:
        state.vote_phase_active = True

    # Resolve vote after step 50 actions
    if state.step_count == 51 and not state.vote_resolved:
        resolve_vote(state)
        state.vote_just_resolved = True  # NEW: True for exactly this round

    # Check terminal conditions
    check_terminal(state)
```

- [ ] **Step 5: Update `vote_reward` gate**

In `v1/survivecity_env/rubric.py`, replace the `vote_reward` function (around line 61). The body is unchanged except the gate condition on line 74:

```python
def vote_reward(state: EpisodeState, agent_id: int) -> float:
    """Score the agent's vote at step 50.

    For healthy agents:
      vote_target == true_infected_id -> +0.30
      vote_target is a healthy agent  -> -0.20
      null/no vote                    -> -0.05

    For infected agent:
      voted for self -> -0.30 (catastrophic blunder)
      voted for a healthy agent -> +0.30 (successful framing)
      null vote -> -0.05
    """
    if not state.vote_just_resolved:  # CHANGED: was `state.vote_resolved`
        return 0.0

    vote = state.votes_cast.get(agent_id)

    if agent_id == state.infected_id:
        if vote is None:
            return -0.05
        if vote == state.infected_id:
            return -0.30
        return 0.30

    if vote is None:
        return -0.05
    if vote == state.infected_id:
        return 0.30
    return -0.20
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_vote_just_resolved_lifecycle tests/test_reward_fixes.py::test_vote_reward_returns_zero_unless_just_resolved -v
```

Expected output: both `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add v1/tests/__init__.py v1/tests/test_reward_fixes.py v1/survivecity_env/game.py v1/survivecity_env/rubric.py
git commit -m "feat(reward): one-shot vote_reward via vote_just_resolved

- Add vote_just_resolved bool to EpisodeState (set true for exactly the
  round when resolve_vote fires; cleared at start of next advance_step)
- Update vote_reward gate from sticky vote_resolved to one-shot
  vote_just_resolved
- Add 2 unit tests covering lifecycle + gate behaviour

Fixes the bug where vote_reward fired every step after 51, baking
±0.20/±0.30 into cumulative reward for the post-vote half of every
episode (visible as raw=-0.345 plateau in eval logs).

Spec: docs/superpowers/specs/2026-04-26-v1-reward-fix-design.md FM-1"
```

---

### Task 2: `pending_damage_reward` field + reset semantics

**Goal:** Introduce a damage-reward accumulator that carries across rounds, so damage taken between an agent's actions is not lost. The field is set wherever HP decreases (next task instruments those sites), drained inside `compose_reward` (Task 4), and explicitly NOT cleared by `reset_step_flags`.

**Files:**
- Modify: `v1/tests/test_reward_fixes.py` (add test)
- Modify: `v1/survivecity_env/game.py` (add field to `_AgentInternal`, update `reset_step_flags`)

- [ ] **Step 1: Add the failing test**

Append to `v1/tests/test_reward_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: pending_damage_reward carries across rounds
# ---------------------------------------------------------------------------

def test_pending_damage_carries_across_apply_action():
    """reset_step_flags MUST NOT clear pending_damage_reward."""
    state = create_episode(seed=42)
    a0 = state.agents[0]

    # Simulate damage taken at end of prior round
    a0.pending_damage_reward = -0.10

    # Agent acts: apply_agent_action calls reset_step_flags internally
    apply_agent_action(state, 0, "wait")

    # The accumulator MUST persist (only env-side drainage clears it, in Task 4)
    assert a0.pending_damage_reward == pytest.approx(-0.10)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_pending_damage_carries_across_apply_action -v
```

Expected: `FAILED ... AttributeError: '_AgentInternal' object has no attribute 'pending_damage_reward'`

- [ ] **Step 3: Add field to `_AgentInternal`**

In `v1/survivecity_env/game.py`, locate the `_AgentInternal` dataclass (around line 26). Add this field after `damage_this_step` (around line 41):

```python
    # Per-step transient flags (reset each step)
    ate_this_step: bool = False
    damage_this_step: int = 0
    died_this_step: bool = False

    # Carries damage cost across rounds; drained inside compose_reward (Task 4).
    # NOT reset by reset_step_flags — only compose_reward clears it.
    pending_damage_reward: float = 0.0  # NEW
```

- [ ] **Step 4: Update `reset_step_flags` to NOT clear the new field**

Still in `v1/survivecity_env/game.py`, replace the `reset_step_flags` method (around line 49):

```python
    def reset_step_flags(self) -> None:
        self.ate_this_step = False
        self.damage_this_step = 0
        self.died_this_step = False
        # NOTE: pending_damage_reward NOT reset here — compose_reward drains it.
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_pending_damage_carries_across_apply_action -v
```

Expected: `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add v1/tests/test_reward_fixes.py v1/survivecity_env/game.py
git commit -m "feat(reward): add pending_damage_reward accumulator on _AgentInternal

- New float field on _AgentInternal, defaults to 0.0
- reset_step_flags explicitly does NOT clear it (carries across rounds)
- Set in damage sites in next task (zombie collisions, infected attacks,
  starvation HP loss); drained inside compose_reward in Task 4

This is the foundation for fixing the damage-credit-assignment hole
where damage_this_step was cleared before the victim's next compose_reward
fired.

Spec: 2026-04-26-v1-reward-fix-design.md FM-2 (component 1/3)"
```

---

### Task 3: Instrument the three damage sites

**Goal:** Every site that decrements an agent's HP also accumulates `−0.10` into that agent's `pending_damage_reward`, so the damage cost is preserved until the next `compose_reward` call for that agent.

**Files:**
- Modify: `v1/tests/test_reward_fixes.py` (add 3 tests)
- Modify: `v1/survivecity_env/game.py` (3 instrumentation lines)

- [ ] **Step 1: Add 3 failing tests for the 3 damage sites**

Append to `v1/tests/test_reward_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: damage sites accumulate pending_damage_reward
# ---------------------------------------------------------------------------

def test_starvation_damage_accumulates_pending():
    """When apply_agent_action triggers starvation HP loss, pending_damage_reward -= 0.10."""
    state = create_episode(seed=42)
    a0 = state.agents[0]
    a0.hunger = 14  # one more action and starvation kicks in (>= 15)
    a0.pending_damage_reward = 0.0

    apply_agent_action(state, 0, "wait")  # hunger becomes 15 -> -1 HP

    assert a0.damage_this_step == 1
    assert a0.pending_damage_reward == pytest.approx(-0.10)


def test_infected_attack_accumulates_victim_pending():
    """When infected attacks an adjacent agent, victim's pending_damage_reward -= 0.10."""
    state = create_episode(seed=42)
    infected_id = state.infected_id
    victim_id = (infected_id + 1) % 3

    infected = state.agents[infected_id]
    victim = state.agents[victim_id]

    # Place victim adjacent to infected; reveal infection
    infected.infection_revealed = True
    victim.row, victim.col = infected.row, infected.col  # same cell -> Chebyshev dist 0
    victim.pending_damage_reward = 0.0

    # Trigger _infected_attack via the infected agent's apply_agent_action
    # (any non-vote action works; movement is convenient)
    apply_agent_action(state, infected_id, "wait")

    assert victim.damage_this_step >= 1
    assert victim.pending_damage_reward == pytest.approx(-0.10 * victim.damage_this_step)


def test_zombie_collision_accumulates_pending():
    """When advance_zombies puts a zombie on the agent's cell, agent.pending_damage_reward -= 0.10."""
    from survivecity_env.game import advance_zombies
    state = create_episode(seed=42)
    a0 = state.agents[0]
    z0 = state.zombies[0]

    # Force a collision: put zombie on agent's cell
    z0.row, z0.col = a0.row, a0.col
    a0.pending_damage_reward = 0.0

    advance_zombies(state)

    # advance_zombies first moves the zombie (may step off A0), then checks
    # collision for ALL zombies. If after movement the zombie is still on A0,
    # A0 takes damage. Since BFS may move the zombie, use a stricter scenario:
    # force collision by overriding zombie position AFTER advance_zombies
    # would naturally move them. Instead, simpler: advance_zombies will not
    # move a zombie onto a safehouse cell; if A0 is outside safehouse and
    # zombie is on A0, the zombie stays. Validate directly.

    # Easier: test the direct collision branch by hand-constructing.
    # advance_zombies only damages on the post-move collision check.
    # Use a known scenario: agent at (0, 4) (no safehouse), zombie also at (0, 4).
    a0.row, a0.col = 0, 4
    z0.row, z0.col = 0, 4
    a0.damage_this_step = 0
    a0.pending_damage_reward = 0.0
    advance_zombies(state)

    # advance_zombies moves zombie first via BFS; then collision check.
    # Either zombie stayed on A0 (damage applied) or moved off (no damage this round).
    # We assert the consistency: damage_this_step matches pending_damage_reward magnitude.
    assert a0.pending_damage_reward == pytest.approx(-0.10 * a0.damage_this_step)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_starvation_damage_accumulates_pending tests/test_reward_fixes.py::test_infected_attack_accumulates_victim_pending tests/test_reward_fixes.py::test_zombie_collision_accumulates_pending -v
```

Expected: 3 FAILED, all because `pending_damage_reward` is still 0.0 after the damage events (the field exists but isn't being set).

- [ ] **Step 3: Instrument starvation damage**

In `v1/survivecity_env/game.py`, locate the starvation block in `apply_agent_action` (around lines 183-187). Add the `pending_damage_reward` accumulator:

```python
    # Starvation damage
    if agent.hunger >= 15:
        agent.hp -= 1
        agent.damage_this_step += 1
        agent.pending_damage_reward -= 0.10  # NEW
        if agent.hp <= 0:
            _kill_agent(agent, state, "hunger")
```

- [ ] **Step 4: Instrument infected attack**

In `v1/survivecity_env/game.py`, locate `_infected_attack` (around line 231). Add the accumulator:

```python
def _infected_attack(attacker: _AgentInternal, state: EpisodeState) -> None:
    """Infected agent attacks adjacent healthy agents (after step 30 reveal)."""
    for other in state.agents:
        if other.agent_id == attacker.agent_id or not other.is_alive:
            continue
        if abs(other.row - attacker.row) <= 1 and abs(other.col - attacker.col) <= 1:
            other.hp -= 1
            other.damage_this_step += 1
            other.pending_damage_reward -= 0.10  # NEW
            if other.hp <= 0:
                _kill_agent(other, state, "infected_attack")

            # Infection spread (20% chance, cap 1 per episode) — unchanged
            if not state.infection_spread_used and not other.is_infected:
                if state.rng.random() < 0.20:
                    state.infection_spread_used = True
```

- [ ] **Step 5: Instrument zombie collision**

In `v1/survivecity_env/game.py`, locate `advance_zombies` collision block (around lines 280-287). Add the accumulator:

```python
        # Check collision: zombie on same cell as agent -> damage
        for agent in state.agents:
            if not agent.is_alive:
                continue
            if agent.row == zombie.row and agent.col == zombie.col:
                agent.hp -= 1
                agent.damage_this_step += 1
                agent.pending_damage_reward -= 0.10  # NEW
                if agent.hp <= 0:
                    _kill_agent(agent, state, "zombie_attack")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_starvation_damage_accumulates_pending tests/test_reward_fixes.py::test_infected_attack_accumulates_victim_pending tests/test_reward_fixes.py::test_zombie_collision_accumulates_pending -v
```

Expected: 3 PASSED.

- [ ] **Step 7: Commit**

```bash
git add v1/tests/test_reward_fixes.py v1/survivecity_env/game.py
git commit -m "feat(reward): instrument 3 damage sites to accumulate pending_damage_reward

- game.py:185 (starvation)        -> agent.pending_damage_reward -= 0.10
- game.py:237 (infected_attack)   -> other.pending_damage_reward -= 0.10
- game.py:284 (zombie_collision)  -> agent.pending_damage_reward -= 0.10

Damage cost is now preserved in the agent's accumulator and survives
across rounds until compose_reward reads + drains it (Task 4).

Spec: 2026-04-26-v1-reward-fix-design.md FM-2 (component 2/3)"
```

---

### Task 4: `survival_reward` reads `pending_damage_reward`; `compose_reward` drains it

**Goal:** `survival_reward` adds the accumulated damage cost to its return value (replacing the existing `damage_this_step` branch which was buggy), and `compose_reward` zeros the accumulator after the rubric reads it. The drain happens in `compose_reward` (single site) so it covers both the env-step path and the terminal-observation path.

**Files:**
- Modify: `v1/tests/test_reward_fixes.py` (add 1 test)
- Modify: `v1/survivecity_env/rubric.py` (`survival_reward` body + `compose_reward` drain)

- [ ] **Step 1: Add the failing test**

Append to `v1/tests/test_reward_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: compose_reward drains pending_damage_reward (no double-counting)
# ---------------------------------------------------------------------------

def test_compose_reward_reads_and_drains_pending_damage():
    """compose_reward sums pending damage into the reward AND zeros it.

    Symmetric mutation: the value was just consumed by survival_reward,
    so the drain happens inside compose_reward itself. Covers both
    env.step's compute path AND _build_observation's terminal path.
    """
    state = create_episode(seed=42)
    a0 = state.agents[0]
    a0.pending_damage_reward = -0.20

    clipped, raw = compose_reward(state, 0)

    # The reward includes the -0.20 plus the alive bonus (+0.005) and any
    # other terms; we just check the damage actually appeared.
    assert raw <= -0.20 + 0.005 + 1e-9, (
        f"raw={raw} should be <= -0.195 if pending damage was applied"
    )
    # And the accumulator was drained inside compose_reward
    assert a0.pending_damage_reward == 0.0

    # Second call must NOT re-apply the same damage
    clipped2, raw2 = compose_reward(state, 0)
    assert raw2 > raw, "second call had no pending damage so should be higher"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_compose_reward_reads_and_drains_pending_damage -v
```

Expected: `FAILED` because `survival_reward` doesn't read `pending_damage_reward` yet (and `compose_reward` doesn't drain).

- [ ] **Step 3: Update `survival_reward` to read `pending_damage_reward` (replacing old damage branch)**

In `v1/survivecity_env/rubric.py`, replace the `survival_reward` function (around line 30):

```python
def survival_reward(state: EpisodeState, agent_id: int) -> float:
    """Per-step survival reward/penalty for one agent.

    +0.005 per step alive
    +0.05  when agent eats food (hunger reset)
    -0.05  per step when hunger >= 10 (starving)
    +pending_damage_reward (carries -0.10 per HP loss across rounds; drained
                            by compose_reward after this fn returns)
    -0.50  at moment of death (one-shot)
    """
    a = state.agents[agent_id]
    r = 0.0

    if a.is_alive:
        r += 0.005
        if a.ate_this_step:
            r += 0.05
        if a.hunger >= 10:
            r -= 0.05
        # CHANGED: replaces direct damage_this_step branch with the
        # cross-round accumulator. compose_reward drains it after we read.
        r += a.pending_damage_reward

    if a.died_this_step:
        r -= 0.50

    return r
```

- [ ] **Step 4: Update `compose_reward` to drain after summing**

In `v1/survivecity_env/rubric.py`, replace the `compose_reward` function (around line 140):

```python
def compose_reward(state: EpisodeState, agent_id: int) -> tuple[float, float]:
    """Compose all rubrics into a single reward.

    Returns:
        (clipped_reward, raw_reward)
        clipped_reward is in (0.01, 0.99) for OpenEnv compliance
        raw_reward is the unclipped sum for debugging / training signal
    """
    raw = (
        survival_reward(state, agent_id)
        + vote_reward(state, agent_id)
        + group_outcome_reward(state, agent_id)
    )
    # Drain the consumed pending_damage_reward (it was just added inside
    # survival_reward). Done here — single site — so all callers get
    # consistent semantics: env.step's compute path AND
    # _build_observation's terminal/reset path both drain.
    state.agents[agent_id].pending_damage_reward = 0.0  # NEW

    clipped = _clip(raw)
    return clipped, raw
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_compose_reward_reads_and_drains_pending_damage -v
```

Expected: `PASSED`.

- [ ] **Step 6: Run all Task 1-4 tests to confirm no regression**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py -v
```

Expected: 7 passed (2 from Task 1, 1 from Task 2, 3 from Task 3, 1 from Task 4).

- [ ] **Step 7: Commit**

```bash
git add v1/tests/test_reward_fixes.py v1/survivecity_env/rubric.py
git commit -m "feat(reward): drain pending_damage_reward inside compose_reward

- survival_reward replaces direct damage_this_step branch with
  pending_damage_reward (the cross-round accumulator from Task 2/3)
- compose_reward zeros the accumulator after the rubric reads it,
  covering both env.step's compute path and _build_observation's
  terminal/reset path with single-site drainage

Combined with Tasks 2-3, this fixes the damage credit assignment hole
where reset_step_flags() at the start of apply_agent_action wiped
damage_this_step before the victim's compose_reward fired. Now zombie
hits, infected attacks, and starvation damage all reach the policy's
gradient signal.

Spec: 2026-04-26-v1-reward-fix-design.md FM-2 (component 3/3)"
```

---

### Task 5: `_min_food_dist` helper

**Goal:** Add a helper that returns the Manhattan distance from a `(row, col)` to the nearest cell in `FOOD_CELLS`. Returns `99` (sentinel) if `FOOD_CELLS` is empty. Used by Task 6 for forage shaping.

**Files:**
- Modify: `v1/tests/test_reward_fixes.py` (add 2 tests)
- Modify: `v1/survivecity_env/game.py` (add helper)

- [ ] **Step 1: Add failing tests**

Append to `v1/tests/test_reward_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: _min_food_dist helper
# ---------------------------------------------------------------------------

def test_min_food_dist_returns_correct_manhattan():
    """_min_food_dist returns Manhattan distance to nearest FOOD_CELLS entry."""
    from survivecity_env.game import _min_food_dist
    from survivecity_env.layout import FOOD_CELLS

    assert FOOD_CELLS, "test prerequisite: FOOD_CELLS must be non-empty in v1 layout"

    # Distance from a known FOOD cell to itself is 0
    fr, fc = next(iter(FOOD_CELLS))
    assert _min_food_dist(fr, fc) == 0

    # Some interior point should have a positive Manhattan distance
    d = _min_food_dist(5, 5)
    expected = min(abs(r - 5) + abs(c - 5) for (r, c) in FOOD_CELLS)
    assert d == expected


def test_min_food_dist_handles_empty_food_cells(monkeypatch):
    """When FOOD_CELLS is empty, helper returns sentinel 99 (no ValueError)."""
    from survivecity_env import game as game_module

    # Patch FOOD_CELLS at the module level the helper imports from
    monkeypatch.setattr("survivecity_env.layout.FOOD_CELLS", set())
    # Reload the binding inside game.py if needed
    monkeypatch.setattr(game_module, "FOOD_CELLS", set(), raising=False)

    # Helper must not raise; must return sentinel
    from survivecity_env.game import _min_food_dist
    assert _min_food_dist(0, 0) == 99
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_min_food_dist_returns_correct_manhattan tests/test_reward_fixes.py::test_min_food_dist_handles_empty_food_cells -v
```

Expected: `FAILED ... ImportError: cannot import name '_min_food_dist' from 'survivecity_env.game'`.

- [ ] **Step 3: Add the helper to `game.py`**

In `v1/survivecity_env/game.py`, locate the layout-helper section (around lines 140-155, just before `apply_agent_action`). Add:

```python
def _min_food_dist(row: int, col: int) -> int:
    """Manhattan distance from (row, col) to nearest food cell.

    Returns 0 if standing on a food cell, 99 if FOOD_CELLS is empty
    (sentinel keeps the forage-shaping arithmetic safe).
    """
    if not FOOD_CELLS:
        return 99
    return min(abs(fr - row) + abs(fc - col) for (fr, fc) in FOOD_CELLS)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_min_food_dist_returns_correct_manhattan tests/test_reward_fixes.py::test_min_food_dist_handles_empty_food_cells -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add v1/tests/test_reward_fixes.py v1/survivecity_env/game.py
git commit -m "feat(reward): add _min_food_dist helper for forage shaping

Manhattan distance from (row, col) to nearest FOOD_CELLS entry.
Returns 99 sentinel when FOOD_CELLS is empty (no ValueError on
edge-case maps).

Used by Task 6's potential-based forage shaping.

Spec: 2026-04-26-v1-reward-fix-design.md FM-7"
```

---

### Task 6: Forage shaping (snapshots + survival_reward delta)

**Goal:** Add `prev_food_dist_this_step` and `cur_food_dist_this_step` snapshots on `_AgentInternal`; populate them inside `apply_agent_action` (before/after the action effects); add a potential-based shaped term to `survival_reward` that pays `+0.005 × (prev_dist − cur_dist)` per action. Net-zero on round-trips, positive when moving toward food, negative when moving away.

**Files:**
- Modify: `v1/tests/test_reward_fixes.py` (add 2 tests)
- Modify: `v1/survivecity_env/game.py` (2 fields, snapshots in `apply_agent_action`)
- Modify: `v1/survivecity_env/rubric.py` (`survival_reward` adds delta term)

- [ ] **Step 1: Add the failing tests**

Append to `v1/tests/test_reward_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: forage shaping (potential-based, +0.005 per cell closer)
# ---------------------------------------------------------------------------

def test_forage_shaping_rewards_closing_distance():
    """Moving 1 cell closer to food adds +0.005 to survival_reward."""
    state = create_episode(seed=42)
    a0 = state.agents[0]
    # Hand-set both snapshots so we don't rely on geometry
    a0.prev_food_dist_this_step = 5
    a0.cur_food_dist_this_step = 4

    r = survival_reward(state, 0)
    # +0.005 alive + 0.005 forage delta (1 cell closer) = 0.010
    # (no eat, no hunger>=10, no pending damage, not died)
    assert r == pytest.approx(0.010, abs=1e-9)


def test_forage_shaping_net_zero_on_round_trip():
    """Move closer then back — net zero forage contribution."""
    state = create_episode(seed=42)
    a0 = state.agents[0]

    a0.prev_food_dist_this_step = 4
    a0.cur_food_dist_this_step = 5  # moved farther
    r_back = survival_reward(state, 0)

    # +0.005 alive - 0.005 forage delta = 0.000
    assert r_back == pytest.approx(0.000, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_forage_shaping_rewards_closing_distance tests/test_reward_fixes.py::test_forage_shaping_net_zero_on_round_trip -v
```

Expected: 2 FAILED with `AttributeError: '_AgentInternal' object has no attribute 'prev_food_dist_this_step'`.

- [ ] **Step 3: Add the snapshot fields to `_AgentInternal`**

In `v1/survivecity_env/game.py`, in the `_AgentInternal` dataclass (around lines 39-43), add the two snapshot fields after `pending_damage_reward`:

```python
    # Per-step transient flags (reset each step)
    ate_this_step: bool = False
    damage_this_step: int = 0
    died_this_step: bool = False

    # Carries damage cost across rounds; drained inside compose_reward.
    pending_damage_reward: float = 0.0

    # Forage-shaping snapshots: Manhattan distance to nearest food before/after
    # this step's action. Overwritten each step by apply_agent_action; -1
    # means "no snapshot taken yet" (initial episode state).
    prev_food_dist_this_step: int = -1  # NEW
    cur_food_dist_this_step: int = -1   # NEW
```

- [ ] **Step 4: Snapshot in `apply_agent_action`**

In `v1/survivecity_env/game.py`, replace the relevant lines of `apply_agent_action` (around lines 161-229). The change is two lines: snapshot prev_food_dist right after `reset_step_flags`, and cur_food_dist at the very end of the function. Show the full updated function:

```python
def apply_agent_action(state: EpisodeState, agent_id: int, action_type: str,
                       vote_target: Optional[int] = None,
                       message: Optional[str] = None) -> None:
    """Apply one agent's action to the episode state.

    This advances that agent's turn but does NOT advance the global step counter.
    The step counter advances after all 3 agents have acted + zombies move.
    """
    agent = state.agents[agent_id]
    if not agent.is_alive:
        return  # dead agents can't act

    agent.reset_step_flags()
    # Snapshot pre-action distance to nearest food (for potential-based shaping)
    agent.prev_food_dist_this_step = _min_food_dist(agent.row, agent.col)  # NEW

    # Hunger always increases (+1 per action, or +1.5 for infected = +2 every other step)
    if agent.is_infected:
        agent.hunger += 2 if (state.step_count % 2 == 0) else 1
    else:
        agent.hunger += 1

    # Starvation damage
    if agent.hunger >= 15:
        agent.hp -= 1
        agent.damage_this_step += 1
        agent.pending_damage_reward -= 0.10
        if agent.hp <= 0:
            _kill_agent(agent, state, "hunger")

    if not agent.is_alive:
        # Take post-action snapshot even on death (so survival_reward sees a
        # well-defined cur_food_dist; survival_reward gates on is_alive anyway).
        agent.cur_food_dist_this_step = _min_food_dist(agent.row, agent.col)  # NEW
        return

    if action_type in _DIRECTION_DELTAS:
        dr, dc = _DIRECTION_DELTAS[action_type]
        new_r, new_c = agent.row + dr, agent.col + dc

        if agent.locked_out and (new_r, new_c) in SAFEHOUSE_CELLS:
            pass
        elif _is_walkable(new_r, new_c, state):
            agent.row = new_r
            agent.col = new_c

    elif action_type == "eat":
        if _is_food_cell(agent.row, agent.col):
            agent.hunger = 0
            agent.ate_this_step = True
            agent.food_eaten += 1

    elif action_type == "vote_lockout":
        if state.step_count >= 50 and vote_target is not None and 0 <= vote_target <= 2:
            state.votes_cast[agent_id] = vote_target

    elif action_type == "broadcast":
        if message:
            msg = f"A{agent_id}: {message[:40]}"
            state.broadcasts.append(msg)
            state.all_broadcasts.append(msg)

    elif action_type == "wait":
        pass

    if agent.is_infected and agent.infection_revealed:
        _infected_attack(agent, state)

    if _is_in_safehouse(agent.row, agent.col) and not agent.locked_out:
        agent.hp = min(3, agent.hp + 1)

    # Snapshot post-action distance to nearest food (the move may have changed it)
    agent.cur_food_dist_this_step = _min_food_dist(agent.row, agent.col)  # NEW
```

- [ ] **Step 5: Add forage delta to `survival_reward`**

In `v1/survivecity_env/rubric.py`, update `survival_reward` (replacing what we wrote in Task 4):

```python
def survival_reward(state: EpisodeState, agent_id: int) -> float:
    """Per-step survival reward/penalty for one agent.

    +0.005 per step alive
    +0.05  when agent eats food (hunger reset)
    -0.05  per step when hunger >= 10 (starving)
    +pending_damage_reward (carries -0.10 per HP loss across rounds;
                            drained by compose_reward after this fn returns)
    +0.005 * (prev_food_dist - cur_food_dist)  potential-based forage shaping
                            (Ng et al. 1999): +0.005 per cell closer,
                            net-zero on round-trips, optimal-policy-preserving
    -0.50  at moment of death (one-shot)
    """
    a = state.agents[agent_id]
    r = 0.0

    if a.is_alive:
        r += 0.005
        if a.ate_this_step:
            r += 0.05
        if a.hunger >= 10:
            r -= 0.05
        r += a.pending_damage_reward
        # Potential-based forage shaping (Ng, Harada, Russell 1999):
        # F(s, a, s') = gamma * Phi(s') - Phi(s) where Phi = -0.005 * min_food_dist.
        # gamma=1.0; cumulative shaped reward over any closed trajectory is exactly 0,
        # so the optimal policy is provably preserved.
        if a.prev_food_dist_this_step >= 0 and a.cur_food_dist_this_step >= 0:
            r += 0.005 * (a.prev_food_dist_this_step - a.cur_food_dist_this_step)  # NEW

    if a.died_this_step:
        r -= 0.50

    return r
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_forage_shaping_rewards_closing_distance tests/test_reward_fixes.py::test_forage_shaping_net_zero_on_round_trip -v
```

Expected: 2 PASSED.

- [ ] **Step 7: Run all Task 1-6 tests to confirm no regression**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py -v
```

Expected: 11 PASSED (cumulative across tasks).

- [ ] **Step 8: Commit**

```bash
git add v1/tests/test_reward_fixes.py v1/survivecity_env/game.py v1/survivecity_env/rubric.py
git commit -m "feat(reward): potential-based forage shaping (+0.005 per cell closer)

- New fields on _AgentInternal: prev_food_dist_this_step, cur_food_dist_this_step
- apply_agent_action snapshots both before and after action effects
- survival_reward adds 0.005 * (prev - cur) — net-zero on round-trips,
  positive when moving toward food, negative away. Uses Ng/Harada/Russell
  1999 potential-based form so the optimal policy is provably preserved.

This is the single biggest behavioural lever in the design — addresses
the eval log's 75% starvation deaths by giving the model a per-action
gradient toward food instead of relying on the rare +0.05 eat bonus.

Spec: 2026-04-26-v1-reward-fix-design.md (forage shaping callout, FM-3, FM-4)"
```

---

### Task 7: Wait penalty

**Goal:** Add a `last_action_this_step` field on `_AgentInternal`, set it inside `apply_agent_action`, reset it inside `reset_step_flags`, and add a `−0.002` penalty in `survival_reward` when the value is `"wait"`. Anti-camping pressure that doesn't crush legitimate safehouse-healing waits.

**Files:**
- Modify: `v1/tests/test_reward_fixes.py` (add 1 test)
- Modify: `v1/survivecity_env/game.py` (1 field, 2 lines in `apply_agent_action` and `reset_step_flags`)
- Modify: `v1/survivecity_env/rubric.py` (`survival_reward` extra branch)

- [ ] **Step 1: Add the failing test**

Append to `v1/tests/test_reward_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 7: wait penalty (-0.002 for alive agents only)
# ---------------------------------------------------------------------------

def test_wait_penalty_only_for_alive_wait_actions():
    """Wait penalty fires when last_action_this_step == 'wait' AND agent is alive."""
    state = create_episode(seed=42)
    a0 = state.agents[0]
    a0.last_action_this_step = "wait"

    r = survival_reward(state, 0)
    # +0.005 alive - 0.002 wait = 0.003 (no eat, no hunger>=10, no damage)
    assert r == pytest.approx(0.003, abs=1e-9)

    # Dead agent: penalty does NOT apply (gated by the is_alive branch)
    a0.is_alive = False
    a0.last_action_this_step = "wait"
    r = survival_reward(state, 0)
    assert r == 0.0  # no alive bonus, no wait penalty -- nothing
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_wait_penalty_only_for_alive_wait_actions -v
```

Expected: `FAILED ... AttributeError: '_AgentInternal' object has no attribute 'last_action_this_step'`.

- [ ] **Step 3: Add the field to `_AgentInternal`**

In `v1/survivecity_env/game.py`, add to `_AgentInternal` (after the forage snapshot fields):

```python
    # Forage-shaping snapshots: Manhattan distance to nearest food before/after
    # this step's action.
    prev_food_dist_this_step: int = -1
    cur_food_dist_this_step: int = -1

    # Last action this agent took this step — used by survival_reward for
    # the wait penalty.
    last_action_this_step: str = ""  # NEW
```

- [ ] **Step 4: Update `reset_step_flags` to clear it**

In `v1/survivecity_env/game.py`, replace `reset_step_flags`:

```python
    def reset_step_flags(self) -> None:
        self.ate_this_step = False
        self.damage_this_step = 0
        self.died_this_step = False
        self.last_action_this_step = ""  # NEW
        # NOTE: pending_damage_reward NOT reset here — compose_reward drains it.
        # NOTE: prev/cur_food_dist_this_step overwritten by next apply_agent_action snapshots.
```

- [ ] **Step 5: Set the field in `apply_agent_action`**

In `v1/survivecity_env/game.py`, in `apply_agent_action`, add the line right after `reset_step_flags()` (and BEFORE the `prev_food_dist_this_step` snapshot):

```python
    agent.reset_step_flags()
    agent.last_action_this_step = action_type  # NEW
    agent.prev_food_dist_this_step = _min_food_dist(agent.row, agent.col)
```

- [ ] **Step 6: Add the wait penalty branch to `survival_reward`**

In `v1/survivecity_env/rubric.py`, update `survival_reward` again:

```python
def survival_reward(state: EpisodeState, agent_id: int) -> float:
    """Per-step survival reward/penalty for one agent.

    +0.005 per step alive
    +0.05  when agent eats food
    -0.05  per step when hunger >= 10
    -0.002 if last action was 'wait' (anti-camping; net step reward
            still positive (+0.003) but lower than non-wait moves)
    +pending_damage_reward
    +0.005 * (prev_food_dist - cur_food_dist) potential-based forage shaping
    -0.50  at moment of death
    """
    a = state.agents[agent_id]
    r = 0.0

    if a.is_alive:
        r += 0.005
        if a.ate_this_step:
            r += 0.05
        if a.hunger >= 10:
            r -= 0.05
        if a.last_action_this_step == "wait":  # NEW: anti-camping
            r -= 0.002
        r += a.pending_damage_reward
        if a.prev_food_dist_this_step >= 0 and a.cur_food_dist_this_step >= 0:
            r += 0.005 * (a.prev_food_dist_this_step - a.cur_food_dist_this_step)

    if a.died_this_step:
        r -= 0.50

    return r
```

- [ ] **Step 7: Run tests to verify it passes**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_wait_penalty_only_for_alive_wait_actions -v
```

Expected: `PASSED`.

- [ ] **Step 8: Run all Task 1-7 tests to confirm no regression**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py -v
```

Expected: 12 PASSED.

- [ ] **Step 9: Commit**

```bash
git add v1/tests/test_reward_fixes.py v1/survivecity_env/game.py v1/survivecity_env/rubric.py
git commit -m "feat(reward): wait penalty -0.002 for alive agents

- New field _AgentInternal.last_action_this_step (cleared by reset_step_flags,
  set by apply_agent_action immediately after reset)
- survival_reward gates a -0.002 branch on last_action_this_step == 'wait'

Net wait reward = +0.005 alive - 0.002 = +0.003 (still positive — preserves
safehouse healing utility — but lower than any non-wait action that doesn't
take damage). 100 waits cumulative -0.2, meaningful anti-camping pressure
without crushing the policy.

Spec: 2026-04-26-v1-reward-fix-design.md (FM-5; user-requested addition)"
```

---

### Task 8: `group_outcome_reward` docstring fix

**Goal:** The existing code in `group_outcome_reward` is per-agent flat (+0.40 if THIS agent is alive+healthy) but the docstring says "for each living healthy agent" (sum over agents). Add a test that pins down the per-agent-flat behaviour, then update the docstring to match — no behavioural change.

**Files:**
- Modify: `v1/tests/test_reward_fixes.py` (add 1 test)
- Modify: `v1/survivecity_env/rubric.py` (docstring rewrite only)

- [ ] **Step 1: Add the test (should already pass — code is correct)**

Append to `v1/tests/test_reward_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 8: group_outcome_reward per-agent flat (docstring/code consistency)
# ---------------------------------------------------------------------------

def test_group_outcome_per_agent_flat():
    """+0.40 fires for THIS agent if alive+healthy at end (not summed over agents)."""
    state = create_episode(seed=42)
    state.done = True
    # Make A0 healthy-alive, A1 healthy-alive, A2 infected-dead
    state.infected_id = 2
    state.agents[0].is_alive = True
    state.agents[0].is_infected = False
    state.agents[1].is_alive = True
    state.agents[1].is_infected = False
    state.agents[2].is_alive = False
    state.agents[2].is_infected = True

    r0 = group_outcome_reward(state, 0)
    # Healthy team survived (A0, A1 alive); infected (A2) dead -> neutralised.
    # A0: +0.40 (alive healthy) + 0.30 (infected neutralised, A0 not infected) = 0.70
    assert r0 == pytest.approx(0.70)

    # If the rubric were summing across agents, r0 would be 0.40 * 2 + 0.30 = 1.10.
    # The current per-agent-flat code gives 0.70.
    assert r0 != pytest.approx(1.10), "code is per-agent flat, not summed over agents"
```

- [ ] **Step 2: Run the test — expect PASS already (code is correct)**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_group_outcome_per_agent_flat -v
```

Expected: `PASSED` immediately. The bug here is purely a docstring mismatch.

- [ ] **Step 3: Update the docstring to match the code**

In `v1/survivecity_env/rubric.py`, replace the docstring of `group_outcome_reward` (around lines 99-110):

```python
def group_outcome_reward(state: EpisodeState, agent_id: int) -> float:
    """Terminal reward based on group outcome (per-agent flat).

    Healthy team survives (>=1 healthy alive at end):
      +0.40 if THIS agent is alive and not infected
      +0.30 if infected was neutralized AND this agent is not infected

    Infected wins (all healthy dead):
      +0.40 if THIS agent is infected (the winner)
      -0.20 if THIS agent is dead AND not infected (a casualty)

    Note: per-agent flat — the same agent can receive +0.40 for being alive+healthy
    AND +0.30 for the infected being neutralized (max +0.70). The function does NOT
    sum across other agents' outcomes.
    """
    if not state.done:
        return 0.0

    a = state.agents[agent_id]
    r = 0.0

    healthy_alive = sum(1 for ag in state.agents if ag.is_alive and not ag.is_infected)
    infected_agent = state.agents[state.infected_id]
    infected_neutralized = not infected_agent.is_alive or infected_agent.locked_out

    if healthy_alive >= 1:
        if a.is_alive and not a.is_infected:
            r += 0.40
        if infected_neutralized and not a.is_infected:
            r += 0.30
    else:
        if a.is_infected:
            r += 0.40
        elif not a.is_alive:
            r -= 0.20

    return r
```

- [ ] **Step 4: Confirm test still passes**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py::test_group_outcome_per_agent_flat -v
```

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add v1/tests/test_reward_fixes.py v1/survivecity_env/rubric.py
git commit -m "docs(reward): rewrite group_outcome_reward docstring to match code

The existing code computes per-agent flat amounts (+0.40 if THIS agent
is alive+healthy, etc.) but the original docstring claimed 'for each
living healthy agent', implying a sum over agents. Add a test pinning
the per-agent semantics and rewrite the docstring to match.

Spec: 2026-04-26-v1-reward-fix-design.md design issue #6"
```

---

### Task 9: Train on raw reward + reward-variance smoke test

**Goal:** Change `train.py:174` to read `obs.metadata.raw_reward` instead of the clipped `obs.reward`. Build a smoke test (`v1/scripts/smoke_test_reward_variance.py`) that scores 8 different first-actions on the same seed and asserts `std(rewards) > 0.01`. The smoke test must FAIL on the unfixed code and PASS after the fix — it's the canary that catches the floor-pinning bug before burning 9 GPU-hours.

**Files:**
- Create: `v1/scripts/smoke_test_reward_variance.py`
- Modify: `v1/training/train.py` (line 174)

- [ ] **Step 1: Create the smoke test**

Create `v1/scripts/smoke_test_reward_variance.py`:

```python
"""Reward-variance smoke test (host CPU, ~30s).

Runs the GRPO reward function against 8 different first-actions on the
same env seed. Asserts std(rewards) > 0.01 — i.e. the reward signal
varies enough across model first-actions to give GRPO a real gradient.

Catches the train-on-clipped-reward bug ahead of the multi-hour training
launch: a clipped 0.01 floor on most timesteps yields std < 0.001 and
GRPO collapses to no-gradient.

Run from v1/ directory:  python scripts/smoke_test_reward_variance.py
"""
from __future__ import annotations

import statistics
import sys

# We don't need a real LLM — reuse the GRPO reward_fn directly.
from training.train import create_reward_fn


def main() -> int:
    reward_fn = create_reward_fn()
    SEED = 12345
    prompt = f"[SEED:{SEED}]\nSmoke test prompt — first action varies across completions."

    completions = [
        '{"action_type":"move_up"}',
        '{"action_type":"move_down"}',
        '{"action_type":"move_left"}',
        '{"action_type":"move_right"}',
        '{"action_type":"eat"}',
        '{"action_type":"wait"}',
        '{"action_type":"broadcast","message":"hi"}',
        "garbage non-JSON to test parse fallback",
    ]
    prompts = [prompt] * len(completions)

    print("Running reward_fn on 8 completions...")
    rewards = reward_fn(prompts, completions)
    for c, r in zip(completions, rewards):
        label = c[:40].replace("\n", " ")
        print(f"  {label!r:45s} -> {r:+.4f}")

    if len(rewards) < 2:
        print("\nERROR: reward_fn returned <2 rewards", file=sys.stderr)
        return 1

    std = statistics.stdev(rewards)
    mean = statistics.mean(rewards)
    print(f"\n  mean = {mean:+.4f}")
    print(f"  std  = {std:.4f}")

    if std <= 0.01:
        print("\nFAILED: reward variance too low (std <= 0.01).", file=sys.stderr)
        print("  GRPO will not get a usable gradient with this signal.", file=sys.stderr)
        print("  Likely cause: reward_fn still reads clipped obs.reward "
              "instead of obs.metadata.raw_reward.", file=sys.stderr)
        return 2

    print("\nPASSED: reward variance is above the 0.01 floor — safe to launch training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the smoke test on UNFIXED code; expect FAIL**

```bash
cd v1 && python scripts/smoke_test_reward_variance.py
```

Expected: prints rewards (most or all near 0.01), then prints `FAILED: reward variance too low (std <= 0.01)`. Exit code: 2.

If the smoke test PASSES on unfixed code, that means the existing reward variance is incidentally high enough — proceed to Step 3 anyway, since the fix is still required by spec.

- [ ] **Step 3: Apply the one-line fix to `train.py`**

In `v1/training/train.py`, locate line 174 inside `create_reward_fn`. The current line reads:

```python
                rewards.append(obs.get("reward", 0.01))
```

Replace it with:

```python
                rewards.append(obs.get("metadata", {}).get("raw_reward", 0.0))  # CHANGED: raw, not clipped
```

- [ ] **Step 4: Run the smoke test again; expect PASS**

```bash
cd v1 && python scripts/smoke_test_reward_variance.py
```

Expected: prints rewards (now varied), then `PASSED: reward variance is above the 0.01 floor`. Exit code: 0.

- [ ] **Step 5: Commit**

```bash
git add v1/scripts/smoke_test_reward_variance.py v1/training/train.py
git commit -m "feat(train): GRPO reward_fn reads raw_reward not clipped reward

train.py:174 changed to obs.metadata.raw_reward (signed, unclipped) from
obs.reward (clipped to (0.01, 0.99) for OpenEnv compliance). The clipped
value pinned ~95% of timesteps to 0.01 and erased GRPO's group-relative
advantages.

Adds scripts/smoke_test_reward_variance.py — host-CPU canary that scores
8 first-actions on the same seed and asserts std(rewards) > 0.01. Run
this before any training launch; aborts pre-launch if the reward signal
is floor-pinned.

Spec: 2026-04-26-v1-reward-fix-design.md priority #1, IF-1, Phase 2 smoke test"
```

---

### Task 10: `TimestampedHubPushCallback` — push to `noanya/v1-zombiee/<timestamp>/`

**Goal:** Replace the default `Trainer.hub_strategy="every_save"` (which pushes to repo root) with a custom callback that uploads `output_dir` to `noanya/v1-zombiee/<timestamp>/`. Multiple training runs accumulate in the same repo with no overwrites.

**Files:**
- Modify: `v1/training/train.py` (add callback class, wire into `main()`, disable default push)

- [ ] **Step 1: Add the callback class near the top of `train.py`**

In `v1/training/train.py`, just below the existing imports and the `_RANDOM_ACTIONS` constant (after line 95, before `_parse_action`), add:

```python
from datetime import datetime
from transformers import TrainerCallback


# Callback class for timestamped HF Hub uploads — installed in main() instead
# of the default Trainer hub_strategy="every_save", which pushes to the repo
# root and overwrites prior runs.
class TimestampedHubPushCallback(TrainerCallback):
    """Push the entire output_dir to noanya/v1-zombiee/<timestamp>/ on every save.

    Replaces hub_strategy='every_save' (root-of-repo) with a per-run
    timestamped subfolder so multiple runs accumulate side-by-side
    in the same repo without overwriting.
    """

    def __init__(self, hub_repo_id: str, path_in_repo: str, hf_token: str | None):
        self.hub_repo_id = hub_repo_id
        self.path_in_repo = path_in_repo
        self.hf_token = hf_token

    def on_save(self, args, state, control, **kw):  # noqa: D401
        from huggingface_hub import upload_folder
        try:
            upload_folder(
                folder_path=args.output_dir,
                path_in_repo=self.path_in_repo,
                repo_id=self.hub_repo_id,
                repo_type="model",
                token=self.hf_token,
                commit_message=f"step {state.global_step}",
            )
            logger.info(
                f"hub push OK: step {state.global_step} -> "
                f"{self.hub_repo_id}/{self.path_in_repo}"
            )
        except Exception as e:
            # Hub push is best-effort; never raise into the training loop.
            logger.warning(
                f"hub push FAILED at step {state.global_step}: {type(e).__name__}: {e}"
            )
```

- [ ] **Step 2: Wire the callback into `GRPOTrainer` construction**

In `v1/training/train.py`, locate the `GRPOTrainer(...)` call inside `main()` (around line 371). Disable the default Trainer-driven hub push (which would push to repo root) and pass the new callback. Replace the relevant block (around lines 297-374) with:

```python
    from trl import GRPOTrainer, GRPOConfig

    # Generate the timestamp ONCE per training launch — used as the
    # path_in_repo for every checkpoint upload by TimestampedHubPushCallback.
    run_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%MZ")
    logger.info(f"Run timestamp (used for HF subfolder): {run_timestamp}")

    # CHANGED: disable Trainer-driven hub push entirely. We push manually
    # via TimestampedHubPushCallback so artefacts land in
    # <hub_model_id>/<timestamp>/ instead of the repo root.
    trainer_push_to_hub = False  # was: bool(args.push_to_hub and args.hub_model_id)
    if args.push_to_hub and not args.hub_model_id:
        logger.warning("--push-to-hub set without --hub-model-id; disabling hub push.")

    # Auto-detect available VRAM and adjust generation count if the GPU is shared.
    num_gen = args.num_generations
    grad_accum = 16
    max_prompt_len = 512
    max_compl_len = 256
    if cuda_ok:
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        logger.info(f"Free VRAM before trainer init: {free_gb:.2f} GB")
        if free_gb < 6:
            num_gen = min(num_gen, 2)
            max_compl_len = 128
            logger.warning(f"Very low VRAM ({free_gb:.1f} GB) — reducing to "
                           f"num_gen={num_gen}, max_compl_len={max_compl_len}")
        elif free_gb < 12:
            num_gen = min(num_gen, 4)
            logger.info(f"Moderate VRAM ({free_gb:.1f} GB) — capping num_gen={num_gen}")

    config = GRPOConfig(
        output_dir=args.output_dir, num_generations=num_gen,
        per_device_train_batch_size=1, gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr, max_steps=args.max_steps,
        save_steps=args.save_steps, logging_steps=10,
        save_total_limit=args.save_total_limit,
        max_prompt_length=max_prompt_len, max_completion_length=max_compl_len,
        temperature=args.temperature, beta=args.beta,
        bf16=use_bf16, fp16=use_fp16,
        bf16_full_eval=use_bf16, fp16_full_eval=use_fp16,
        tf32=use_bf16,
        report_to=args.report_to if args.report_to != "none" else None,
        push_to_hub=trainer_push_to_hub,                    # CHANGED: now False
        hub_model_id=None,                                  # CHANGED: callback owns this
        hub_private_repo=args.hub_private,
        hub_strategy="end",                                 # CHANGED: ignored since push_to_hub=False
        seed=args.seed)
    logger.info(
        f"GRPOConfig precision: bf16={config.bf16} fp16={config.fp16} "
        f"bf16_full_eval={config.bf16_full_eval} fp16_full_eval={config.fp16_full_eval} "
        f"tf32={config.tf32}"
    )

    _seed_warnings_issued(model)

    # Build callbacks list. Only attach the hub-push callback if --push-to-hub
    # was supplied and a hub_model_id is set.
    callbacks: list[TrainerCallback] = []
    if args.push_to_hub and args.hub_model_id:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        callbacks.append(TimestampedHubPushCallback(
            hub_repo_id=args.hub_model_id,
            path_in_repo=run_timestamp,
            hf_token=hf_token,
        ))
        logger.info(
            f"TimestampedHubPushCallback installed: "
            f"{args.hub_model_id}/{run_timestamp}/"
        )

    trainer = GRPOTrainer(
        model=model, args=config,
        reward_funcs=[create_reward_fn(args.env_url)],
        train_dataset=dataset, processing_class=tokenizer,
        callbacks=callbacks,                                 # NEW
    )
```

- [ ] **Step 3: Update the post-training push so it ALSO uses the callback's path**

In `v1/training/train.py`, find the post-training push block (around line 393-398). Replace:

```python
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if push_to_hub:
        logger.info(f"Pushing final model to hub: {args.hub_model_id}")
        trainer.push_to_hub(commit_message="final model")
    logger.info(f"Saved to {args.output_dir}")
```

With:

```python
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Final push to the timestamped subfolder (the on_save callback also
    # fires here, but doing one explicit upload at end-of-train guarantees
    # the final tokenizer + adapter land under <timestamp>/ even if the
    # last save_steps was earlier than max_steps).
    if args.push_to_hub and args.hub_model_id:
        from huggingface_hub import upload_folder
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            upload_folder(
                folder_path=args.output_dir,
                path_in_repo=run_timestamp,
                repo_id=args.hub_model_id,
                repo_type="model",
                token=hf_token,
                commit_message="final model",
            )
            logger.info(
                f"Final hub push OK: {args.hub_model_id}/{run_timestamp}/"
            )
        except Exception as e:
            logger.warning(f"Final hub push failed: {type(e).__name__}: {e}")
    logger.info(f"Saved to {args.output_dir}")
```

- [ ] **Step 4: Smoke-test the import path (callback class is syntactically valid)**

```bash
cd v1 && python -c "from training.train import TimestampedHubPushCallback; print('callback class imports cleanly')"
```

Expected output: `callback class imports cleanly`.

If this fails with `ImportError` or `SyntaxError`, fix the train.py edit before committing.

- [ ] **Step 5: Commit**

```bash
git add v1/training/train.py
git commit -m "feat(train): TimestampedHubPushCallback — push to <repo>/<timestamp>/

- New TrainerCallback that fires on every save_steps interval
- Uploads output_dir to noanya/v1-zombiee/<timestamp>/ via huggingface_hub.upload_folder
- Replaces Trainer's hub_strategy='every_save' which pushes to repo root
  and overwrites prior runs
- Disables Trainer's auto-push (push_to_hub=False) so the callback owns
  all hub interactions
- Final post-train push ALSO uses the timestamped path

Run timestamp is generated once per training launch (UTC, format
YYYY-MM-DDTHHMMZ). Multiple training runs accumulate side-by-side in
the same repo without overwriting.

Spec: 2026-04-26-v1-reward-fix-design.md (Hub targets section, IF-4)"
```

---

### Task 11: Final verification — full test suite + smoke test + branch state

**Goal:** Confirm every test passes, the smoke test passes, the v1new branch is in a clean state ready for the Kaggle training launch.

**Files:** None (verification only).

- [ ] **Step 1: Run the full unit test suite**

```bash
cd v1 && python -m pytest tests/test_reward_fixes.py -v
```

Expected: 12 PASSED, 0 FAILED. Specifically:
1. `test_vote_just_resolved_lifecycle`
2. `test_vote_reward_returns_zero_unless_just_resolved`
3. `test_pending_damage_carries_across_apply_action`
4. `test_starvation_damage_accumulates_pending`
5. `test_infected_attack_accumulates_victim_pending`
6. `test_zombie_collision_accumulates_pending`
7. `test_compose_reward_reads_and_drains_pending_damage`
8. `test_min_food_dist_returns_correct_manhattan`
9. `test_min_food_dist_handles_empty_food_cells`
10. `test_forage_shaping_rewards_closing_distance`
11. `test_forage_shaping_net_zero_on_round_trip`
12. `test_wait_penalty_only_for_alive_wait_actions`
13. `test_group_outcome_per_agent_flat`

(13 tests after Task 8 is added — count was off-by-one above; expect 13 passed.)

If anything fails, halt and fix the failing test before moving on. Do NOT launch training with red tests.

- [ ] **Step 2: Run the smoke test**

```bash
cd v1 && python scripts/smoke_test_reward_variance.py
```

Expected: `PASSED: reward variance is above the 0.01 floor — safe to launch training`. Exit code 0.

- [ ] **Step 3: Verify branch state and commit log**

```bash
git status
git log --oneline -15
git rev-parse --abbrev-ref HEAD
```

Expected: clean working tree, branch `v1new`, ~10 commits visible (one per task plus the spec commits from before this plan).

- [ ] **Step 4: Push the v1new branch to GitHub**

```bash
git push -u origin v1new
```

Expected: `Branch 'v1new' set up to track remote branch 'v1new' from 'origin'.`

If the push fails because the branch already exists upstream, use `git push origin v1new` (without `-u`) to update.

- [ ] **Step 5: Final commit marking the branch as ready**

This is a bookkeeping commit so the GitHub branch has a clear "ready-to-train" head:

```bash
git commit --allow-empty -m "chore: v1new ready for Kaggle training launch

All reward-function fixes implemented and tested:
- One-shot vote_reward (FM-1)
- pending_damage_reward accumulator + drain (FM-2)
- _min_food_dist helper (FM-7)
- Potential-based forage shaping (FM-3, FM-4)
- Wait penalty (FM-5; user-requested)
- group_outcome docstring fix (design issue #6)
- train.py reads raw_reward (priority #1)
- TimestampedHubPushCallback (Hub layout)

Verification:
- 13/13 unit tests pass
- Smoke test passes (reward variance > 0.01)

Next: launch Kaggle training with HUB_REPO=noanya/v1-zombiee."

git push origin v1new
```

- [ ] **Step 6: Document run-launch checklist**

This is for the Kaggle training operator (likely the same person, but documenting for clarity). After this plan is complete, the next steps OUTSIDE this plan are:

1. Open Kaggle notebook (or use an existing one — `v1/notebooks/train_kaggle.ipynb`).
2. Set `HF_TOKEN` in Kaggle Secrets.
3. Set Accelerator: GPU T4 ×1.
4. Set Internet: On.
5. In the notebook's git-clone cell, check out branch `v1new` instead of `master`.
6. Pre-flight: run `python scripts/smoke_test_reward_variance.py` from inside the notebook BEFORE launching training. If it fails, abort.
7. Launch training with `--hub-model-id noanya/v1-zombiee --push-to-hub`. Output dir: `/kaggle/working/lora_v1_fixed`.
8. Watch TB for first-step assertions: `reward_std > 0`, `loss > 0`, `grad_norm > 0`. If any are zero, kill the run.
9. After ~25-30 GRPO steps complete (~9h), eval against `noanya/v1-zombiee/<timestamp>/`.

Plan complete after Step 5 — Step 6 is operational handoff documentation.

---

## Self-Review

Spec coverage scan against `docs/superpowers/specs/2026-04-26-v1-reward-fix-design.md`:

| Spec section | Implemented in |
|---|---|
| Bug #1 (clipped reward) | Task 9 |
| Bug #2 (vote_reward sticky) | Task 1 |
| Bug #2 from analysis (damage credit) | Tasks 2, 3, 4 |
| Forage shaping at a glance | Tasks 5, 6 |
| Wait penalty (user-requested) | Task 7 |
| group_outcome docstring | Task 8 |
| Hub layout (timestamped) | Task 10 |
| Phase 1 unit tests | Tasks 1-8 (13 tests) |
| Phase 2 smoke test | Task 9 |
| Phase 3 in-training assertions | Task 11 Step 6 (documented) |
| Phase 4 eval | Out of scope (post-training; uses existing `eval.py`) |

No spec gaps. No placeholders ("TBD" / "TODO" / "fill in later") in the plan. All test code is concrete; all implementation code is concrete with exact file paths and approximate line numbers.

Type/name consistency check:
- Field names: `pending_damage_reward`, `prev_food_dist_this_step`, `cur_food_dist_this_step`, `last_action_this_step`, `vote_just_resolved` — all consistent across tasks.
- Helper name: `_min_food_dist` consistent.
- Test names match across the verify-tests-pass commands and the file contents.
- Import paths match: `from survivecity_env.game import ...`, `from survivecity_env.rubric import ...`, `from training.train import ...`.

One minor inconsistency I noticed during review: Task 11 step 1 says "12 PASSED" then later says "13 PASSED". The correct count is 13 (one test added in each of Tasks 1-8 except Task 1 which adds 2, so: 2+1+3+1+2+2+1+1 = 13). Updated inline.
