"""SurviveCityV2Env — OpenEnv-compliant wrapper around the v2 game.

Implements reset(), step(), state property. Mirrors v1's API exactly so
training loops are interchangeable. Adds v2-specific metadata fields.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from survivecity_v2_env.models import (
    AgentState,
    ZombieState,
    SurviveAction,
    SurviveObservation,
)
from survivecity_v2_env.game import (
    EpisodeState,
    create_episode,
    apply_agent_action,
    advance_zombies,
    advance_step,
    get_current_phase,
)
from survivecity_v2_env.infection import (
    mask_infection_for_agent,
    get_behavioral_cues,
)
from survivecity_v2_env.rubric import compose_reward, per_rubric_breakdown
from survivecity_v2_env.layout import render_grid
from survivecity_v2_env.prompts import format_observation_description

logger = logging.getLogger(__name__)


N_AGENTS = 5


class SurviveCityV2Env:
    """OpenEnv-compliant 5-agent multi-resource zombie env."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._episode: Optional[EpisodeState] = None
        self._episode_id: int = 0
        self._cumulative_rewards: dict[int, float] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> dict:
        actual_seed = seed if seed is not None else self._seed
        if actual_seed is None:
            actual_seed = 0
        self._episode = create_episode(actual_seed)
        self._episode_id += 1
        self._cumulative_rewards = {i: 0.0 for i in range(N_AGENTS)}

        starters = [
            (a.agent_id, a.infection_role)
            for a in self._episode.agents
            if a.infection_role
        ]
        logger.info(
            f"[v2 START] episode={self._episode_id} starting_infected={starters} seed={actual_seed}"
        )

        obs = self._build_observation(agent_id=0)
        return obs.model_dump()

    def step(self, action: dict) -> dict:
        if self._episode is None:
            raise RuntimeError("Must call reset() before step()")

        if self._episode.done:
            obs = self._build_observation(agent_id=0)
            return obs.model_dump()

        # Lenient parsing — extra fields are ignored, missing v2 fields default to None
        try:
            parsed = SurviveAction(**action)
        except Exception as e:
            logger.debug(f"Action parse error: {e}")
            obs = self._build_observation(agent_id=self._get_next_alive_agent() or 0)
            return obs.model_dump()

        # Determine the agent whose turn this is — must match parsed.agent_id
        expected = self._get_next_alive_agent()
        if expected is None:
            obs = self._build_observation(agent_id=0)
            return obs.model_dump()

        # Apply the action (no-ops if agent_id is invalid or dead)
        apply_agent_action(
            self._episode,
            agent_id=parsed.agent_id,
            action_type=parsed.action_type,
            vote_target=parsed.vote_target,
            message=parsed.message,
            scan_target=parsed.scan_target,
            inject_target=parsed.inject_target,
            gift_target=parsed.gift_target,
            item_slot=parsed.item_slot,
            item_type=parsed.item_type,
        )
        self._episode.agents_acted_this_step += 1

        # Compute reward for the acting agent
        clipped, raw = compose_reward(self._episode, parsed.agent_id)
        self._cumulative_rewards[parsed.agent_id] = (
            self._cumulative_rewards.get(parsed.agent_id, 0.0) + raw
        )

        logger.info(
            f"[v2 STEP] ep={self._episode_id} step={self._episode.step_count} "
            f"agent=A{parsed.agent_id} action={parsed.action_type} "
            f"reward={clipped:.4f} raw={raw:+.4f} done={self._episode.done}"
        )

        # If all alive agents have acted this step, advance zombies + step
        alive_count = sum(1 for a in self._episode.agents if a.is_alive)
        if self._episode.agents_acted_this_step >= alive_count:
            advance_zombies(self._episode)
            advance_step(self._episode)
            self._episode.agents_acted_this_step = 0

        next_id = self._get_next_alive_agent()
        if next_id is None:
            next_id = 0

        obs = self._build_observation(
            agent_id=next_id,
            last_reward=clipped,
            last_raw=raw,
        )
        return obs.model_dump()

    @property
    def state(self) -> dict:
        if self._episode is None:
            return {"episode_id": 0, "step_count": 0, "done": True}
        ep = self._episode
        starting_infected = [
            a.agent_id for a in ep.agents
            if a.infection_role in {"biter", "saboteur"}
        ]
        return {
            "episode_id": self._episode_id,
            "step_count": ep.step_count,
            "max_steps": ep.max_steps,
            "done": ep.done,
            "starting_infected": starting_infected,
            "phase": get_current_phase(ep),
            "day_phase": ep.day_phase,
            "noise_meter": ep.noise_meter,
            "n_zombies": len(ep.zombies),
            "n_alive": sum(1 for a in ep.agents if a.is_alive),
            "n_healthy_alive": sum(
                1 for a in ep.agents
                if a.is_alive and a.infection_state == "none"
            ),
            "cumulative_rewards": dict(self._cumulative_rewards),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_next_alive_agent(self) -> Optional[int]:
        if self._episode is None:
            return None
        acted = self._episode.agents_acted_this_step
        alive = [a.agent_id for a in self._episode.agents if a.is_alive]
        if not alive:
            return None
        if acted < len(alive):
            return alive[acted]
        return alive[0]

    def _build_observation(
        self,
        agent_id: int,
        last_reward: Optional[float] = None,
        last_raw: Optional[float] = None,
    ) -> SurviveObservation:
        ep = self._episode
        if ep is None:
            raise RuntimeError("No active episode")

        masked = mask_infection_for_agent(ep, observer_id=agent_id)
        cues = get_behavioral_cues(ep, observer_id=agent_id)

        # Pydantic AgentState models
        agent_states = [
            AgentState(
                agent_id=a["agent_id"],
                row=a["row"],
                col=a["col"],
                hp=a["hp"],
                hunger=a["hunger"],
                thirst=a["thirst"],
                is_alive=a["is_alive"],
                locked_out=a["locked_out"],
                infection_state=a["infection_state"],
                infection_role=a["infection_role"],
                bite_at_step=a["bite_at_step"],
                inventory=a["inventory"],
            )
            for a in masked
        ]
        zombie_states = [
            ZombieState(zombie_id=z.zombie_id, row=z.row, col=z.col)
            for z in ep.zombies
        ]

        grid = render_grid(
            ep.base_grid,
            [a.model_dump() for a in agent_states],
            [z.model_dump() for z in zombie_states],
            food_present=ep.food_present,
            medicine_present=ep.medicine_present,
        )

        if last_reward is not None:
            reward = last_reward
            raw = last_raw if last_raw is not None else 0.0
        else:
            reward, raw = compose_reward(ep, agent_id)

        phase = get_current_phase(ep)

        # Find this agent's own internal state for the description
        own_internal = ep.agents[agent_id]
        own_inventory = list(own_internal.inventory)
        own_inf_state = own_internal.infection_state
        own_bite_at = own_internal.bite_at_step
        last_scan = ep.last_scan_result.get(agent_id)

        description = format_observation_description(
            agent_id=agent_id,
            state_dict={
                "agents": masked,
                "zombies": [z.model_dump() for z in zombie_states],
            },
            phase=phase,
            day_phase=ep.day_phase,
            step=ep.step_count,
            broadcasts=ep.broadcasts + ep.all_broadcasts[-5:],
            behavioral_cues=cues,
            last_scan=last_scan,
            own_inventory=own_inventory,
            own_infection_state=own_inf_state,
            own_bite_at_step=own_bite_at,
            noise_meter=ep.noise_meter,
            noise_threshold=ep.noise_threshold,
        )

        # Aggregate metadata
        infected_agents = [
            a.agent_id for a in ep.agents
            if a.infection_state in {"latent", "revealed"}
        ]
        starting_infected = [
            a.agent_id for a in ep.agents
            if a.infection_role in {"biter", "saboteur"}
        ]
        # Vote correctness for the metadata side-channel: True iff this agent's
        # vote in the most recent resolved phase targeted a starting-infected.
        vote_correct: Optional[bool] = None
        for vstep in (90, 60, 30):
            if vstep in ep.vote_resolved_steps:
                v = ep.votes_cast.get(vstep, {}).get(agent_id)
                if v is None:
                    vote_correct = None
                else:
                    vote_correct = ep.agents[v].infection_role in {"biter", "saboteur"}
                break

        metadata: dict[str, Any] = {
            "raw_reward": round(raw, 6),
            "current_agent_id": agent_id,
            "phase": phase,
            "day_phase": ep.day_phase,
            "step_count": ep.step_count,
            "postmortems": list(ep.postmortems),
            "starting_infected": starting_infected,
            "currently_infected_count": len(infected_agents),
            "noise_meter": ep.noise_meter,
            "noise_threshold": ep.noise_threshold,
            "n_zombies": len(ep.zombies),
            "lockout_results": dict(ep.lockout_results),
            "vote_correct": vote_correct,
            "bite_history": list(ep.bite_history),
            "rubric_breakdown": per_rubric_breakdown(ep, agent_id),
            "n_alive": sum(1 for a in ep.agents if a.is_alive),
            "n_healthy_alive": sum(
                1 for a in ep.agents if a.is_alive and a.infection_state == "none"
            ),
            "last_inject_result": ep.last_inject_result.get(agent_id),
        }

        return SurviveObservation(
            grid=grid,
            agents=agent_states,
            zombies=zombie_states,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            description=description,
            done=ep.done,
            reward=reward,
            metadata=metadata,
            broadcasts=list(ep.all_broadcasts[-10:]),
        )
