# Reward Design — SurviveCity

Three independent rubrics compose into `obs.reward`. All deterministic, no LLM judge.

## Overview

```
total_reward = SurvivalRubric(state, agent_id)
             + VoteRubric(state, agent_id)
             + GroupOutcomeRubric(state, agent_id)

obs.metadata["raw_reward"] = total_reward
obs.reward = clip(total_reward, 0.01, 0.99)
```

---

## Rubric 1 — SurvivalRubric (dense, per-step, per-agent)

Fires every step. Reflects moment-to-moment health/food management.

| Trigger | Δ |
|---|---|
| Agent alive at this step | **+0.005** |
| Agent ate food (hunger reset) | **+0.05** |
| Agent took 1 HP damage from zombie | **−0.10** |
| Hunger ≥ 10 (starving) | **−0.05** per step |
| Agent died this step | **−0.50** (one-shot) |

Implementation:

```python
def survival_reward(state, agent_id):
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
```

---

## Rubric 2 — VoteRubric (sparse, fires once at step 50)

Judges the agent's vote correctness in the safehouse-lockout vote.

| Agent's vote | Δ |
|---|---|
| Vote target == true infected id | **+0.30** |
| Vote target is a healthy agent (wrong lockout) | **−0.20** |
| Did not vote (null / timeout) | **−0.05** |

```python
def vote_reward(state, agent_id):
    if state.step != 50 or not state.vote_resolved:
        return 0.0
    vote = state.votes_cast.get(agent_id)
    if vote is None:
        return -0.05
    if vote == state.infected_id:
        return 0.30
    return -0.20
```

Special case: if the agent is the infected one, their own vote is scored against them (voting for a healthy agent is +0.30 for them — successful framing). Treat infected separately:

```python
if agent_id == state.infected_id:
    if vote is None: return -0.05
    if vote == state.infected_id: return -0.30   # voted self = catastrophic
    return +0.30                                  # successfully framed a healthy agent
```

---

## Rubric 3 — GroupOutcomeRubric (terminal, fires on episode end)

Applied to every agent at `done=True`.

| Scenario | Δ per agent |
|---|---|
| Healthy agent alive at step 100 | **+0.40** for that agent |
| Infected dead or locked out at step 100 | **+0.30** for each healthy agent |
| All healthy agents dead (infected wins) | **+0.40** for infected, **−0.20** for each dead healthy agent |

```python
def group_outcome(state, agent_id):
    if not state.done:
        return 0.0
    a = state.agents[agent_id]
    r = 0.0
    healthy_alive = sum(1 for ag in state.agents if ag.is_alive and not ag.is_infected)
    infected_neutralized = not state.agents[state.infected_id].is_alive or state.agents[state.infected_id].locked_out

    if healthy_alive >= 1:
        if a.is_alive and not a.is_infected:
            r += 0.40
        if infected_neutralized and not a.is_infected:
            r += 0.30
    else:
        # infected wins
        if a.is_infected:
            r += 0.40
        elif not a.is_alive:
            r -= 0.20
    return r
```

---

## Gaming resistance — why each loophole is closed

| Attempted exploit | Why it fails |
|---|---|
| Hide in safehouse forever | Hunger penalty (−0.05/step when hunger≥10) forces foraging |
| Always vote to lock someone out | Wrong lockout = −0.20 |
| Never vote | Null vote = −0.05 |
| Infected waits quietly | Must survive step 100 — healthy agents will eventually vote and likely catch them; infected must actively mislead |
| Infected kills all quickly | Post step-30 reveal means all agents see the infection spread visually; other agents coordinate quickly |

## Final composition

```python
def compose_reward(state, agent_id):
    raw = survival_reward(state, agent_id) + vote_reward(state, agent_id) + group_outcome(state, agent_id)
    clipped = max(0.01, min(0.99, raw))
    return clipped, raw
```

Return both values; set `obs.reward = clipped`, `obs.metadata["raw_reward"] = raw`.

## Expected reward ranges

| Phase | Typical per-step reward |
|---|---|
| Early episode, alive, eating occasionally | 0.005 – 0.055 |
| Step 50 vote (correct) | 0.305 |
| Step 50 vote (wrong) | −0.195 → clamped to 0.01 |
| Terminal, group won, agent alive | 0.70 – 0.99 |
| Terminal, group lost, agent dead | 0.01 |

## Test cases to write

1. Perfect healthy agent survives all 100 steps + correct vote → final reward ≥ 0.90
2. Healthy agent dies at step 10 → cumulative reward < 0.10 for episode
3. Infected agent successfully frames a healthy agent and survives → reward ≥ 0.70
4. Infected agent voted out at step 50 → infected's terminal reward ≤ 0.10
5. Random-action baseline averaged over 100 episodes → mean per-episode reward in range [0.15, 0.30]
