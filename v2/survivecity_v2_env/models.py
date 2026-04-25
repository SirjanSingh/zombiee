"""Pydantic models — public API for SurviveCity v2 actions and observations.

Strict superset of v1: every v1 action_type still validates, every v1 field
still appears in observations. New action types and observation fields are
strictly additive.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Item type — used for inventory and resource depots
# ---------------------------------------------------------------------------

ItemType = Literal["food", "water", "medicine"]


# ---------------------------------------------------------------------------
# Agent state — observable, with infection masking applied at the env layer
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent_id: int
    row: int
    col: int
    hp: int = Field(default=3, ge=0, le=3)
    hunger: int = Field(default=0, ge=0)
    thirst: int = Field(default=0, ge=0)
    is_alive: bool = True
    locked_out: bool = False

    # Infection state visible to this agent's observer; masked for others
    # via env._build_observation. Values: "none" | "latent" | "revealed".
    infection_state: Literal["none", "latent", "revealed"] = "none"
    # Role (only meaningful when infection_state != "none"):
    # "biter" | "saboteur" | None
    infection_role: Optional[Literal["biter", "saboteur"]] = None
    # Step at which infection started (when bitten, or step 0 for starters)
    bite_at_step: Optional[int] = None

    # Inventory — a list of item types, max length 3 enforced by the env
    inventory: list[ItemType] = Field(default_factory=list)

    # Per-step transient flags — populated by the env during step()
    ate_this_step: bool = False
    drank_this_step: bool = False
    damage_this_step: int = 0
    died_this_step: bool = False


# ---------------------------------------------------------------------------
# Zombie state
# ---------------------------------------------------------------------------

class ZombieState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    zombie_id: int
    row: int
    col: int


# ---------------------------------------------------------------------------
# Action model — superset of v1 action_type
# ---------------------------------------------------------------------------

ACTION_TYPES = Literal[
    # v1 actions — UNCHANGED
    "move_up", "move_down", "move_left", "move_right",
    "eat", "wait", "vote_lockout", "broadcast",
    # v2 additions — strictly additive
    "drink",
    "scan",
    "pickup",
    "drop",
    "give",
    "inject",
]


class SurviveAction(BaseModel):
    """One agent's action for one step.

    Most fields are optional. Concrete validation rules:
      - vote_lockout    → vote_target in {0..4}
      - broadcast       → message length ≤ 40 (Pydantic enforces)
      - scan            → scan_target in {0..4}, must be != self
      - pickup          → item_type if cell has multiple resources stacked
      - drop / give     → item_slot in {0,1,2}
      - inject          → inject_target (None == self), item_slot in {0,1,2}
      - give            → gift_target adjacent agent_id

    Invalid combinations don't raise — the env interprets them as no-op so
    the action JSON is always parseable (matches v1's lenient behaviour).
    """

    model_config = ConfigDict(extra="ignore")

    agent_id: int
    action_type: ACTION_TYPES

    # v1 fields
    vote_target: Optional[int] = None
    message: Optional[str] = Field(default=None, max_length=40)

    # v2 fields — additive
    scan_target: Optional[int] = None
    inject_target: Optional[int] = None
    gift_target: Optional[int] = None
    item_slot: Optional[int] = None
    item_type: Optional[ItemType] = None


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class SurviveObservation(BaseModel):
    """Observation returned to the current agent after each step.

    OpenEnv contract pieces:
      - reward in (0.01, 0.99) — set on EVERY step
      - done flag
      - description: NL string for LLM prompting
      - metadata: raw_reward, current_agent_id, phase, postmortems, ...
    """

    model_config = ConfigDict(extra="ignore")

    grid: list[list[str]]
    agents: list[AgentState]
    zombies: list[ZombieState]
    step_count: int
    max_steps: int = 100
    task_id: str = "survivecity_v2"
    description: str = ""
    done: bool = False
    reward: float = Field(default=0.50, gt=0.0, lt=1.0)
    metadata: dict = Field(default_factory=dict)
    broadcasts: list[str] = Field(default_factory=list)
