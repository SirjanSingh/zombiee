"""Infection mechanics for v2 — masking, behavioural cues, and bite RNG.

The actual mutation of infection_state is in game.py. This module provides:

  1. mask_infection_for_agent: build the per-agent observation view
     respecting the rule "you only see your own infection_state".
  2. get_behavioral_cues: deterministic noisy hints other agents use to
     infer infection (30% false-positive on healthy, 30% miss on infected
     — exact rates honoured via deterministic per-step hashing).
  3. should_bite: hash-seeded, deterministic bite RNG. Compares against
     P_BITE without using random.random() so the outcome is fully
     reproducible from (episode_seed, step, biter_id, victim_id).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from survivecity_v2_env.game import EpisodeState


# Bite probability on adjacency. Calibrated so a balance sweep with random
# policy puts random survival in [0%, 5%] over 100 episodes (see
# notebooks/balance_v2_sweep.ipynb).
P_BITE = 0.35

# Latent-infection countdown: the bitten agent's infection_state flips from
# "latent" → "revealed" exactly LATENT_DURATION steps after the bite.
LATENT_DURATION = 15

# Reveal step for starting-infected agents (bypassing the bite mechanic):
#   biter starts revealed at step BITER_REVEAL_STEP
#   saboteur starts revealed at step SABOTEUR_REVEAL_STEP
BITER_REVEAL_STEP = 25
SABOTEUR_REVEAL_STEP = 60

# Behavioural-cue noise rates. Floats in [0, 1].
CUE_FALSE_POSITIVE_RATE = 0.30   # healthy agent flagged as suspicious
CUE_MISS_RATE = 0.30             # infected agent's cues suppressed


def _hash01(*parts: int | str) -> float:
    """Deterministic [0, 1) sample from arbitrary integer/string parts.

    Uses BLAKE2b-128 (cryptographic, fast, available in Python stdlib). We
    take the first 8 bytes as an unsigned 64-bit int and divide by 2**64.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "big") / (1 << 64)


def should_bite(
    episode_seed: int,
    step: int,
    biter_id: int,
    victim_id: int,
) -> bool:
    """Deterministically decide whether `biter_id` bites `victim_id` this step.

    OpenEnv compliance note: this does NOT advance any global RNG state,
    so the episode is reproducible regardless of how many bite checks
    are performed (which depends on adjacency, which depends on actions).
    """
    return _hash01("bite", episode_seed, step, biter_id, victim_id) < P_BITE


def cue_visible(
    episode_seed: int,
    step: int,
    observer_id: int,
    target_id: int,
    target_is_infected: bool,
) -> bool:
    """Deterministically decide whether the cue for target is shown to observer.

    Implements the noise rates:
      - target healthy: cue shown with prob CUE_FALSE_POSITIVE_RATE
      - target infected: cue shown with prob (1 - CUE_MISS_RATE)
    """
    h = _hash01("cue", episode_seed, step, observer_id, target_id)
    if target_is_infected:
        return h < (1.0 - CUE_MISS_RATE)
    return h < CUE_FALSE_POSITIVE_RATE


def mask_infection_for_agent(state: "EpisodeState", observer_id: int) -> list[dict]:
    """Build agent observation dicts with infection_state appropriately masked.

    Rules:
      - An observer NEVER sees another agent's infection_state.
      - An observer sees their own infection_state — including "latent",
        because the env explicitly leaks "You were bitten at step T." to
        bitten agents (lets them self-medicate). Starting infected agents
        see "latent" until their reveal step, then "revealed".
      - Inventory is also masked from observers (own-only).
    """
    masked: list[dict] = []
    for a in state.agents:
        is_self = a.agent_id == observer_id
        entry = {
            "agent_id": a.agent_id,
            "row": a.row,
            "col": a.col,
            "hp": a.hp,
            "hunger": a.hunger,
            "thirst": a.thirst,
            "is_alive": a.is_alive,
            "locked_out": a.locked_out,
            "infection_state": a.infection_state if is_self else "none",
            "infection_role": a.infection_role if is_self else None,
            "bite_at_step": a.bite_at_step if is_self else None,
            "inventory": list(a.inventory) if is_self else [],
        }
        masked.append(entry)
    return masked


def get_behavioral_cues(state: "EpisodeState", observer_id: int) -> list[str]:
    """Generate the noisy behavioural cues visible to `observer_id` this step.

    The cue text is deterministic given (episode_seed, step, observer_id,
    target_id) — agents who run the same episode twice see the exact same
    cue stream, which is required by OpenEnv's reproducibility guarantee.
    """
    cues: list[str] = []
    for a in state.agents:
        if a.agent_id == observer_id or not a.is_alive:
            continue
        infected = a.infection_state in {"latent", "revealed"}
        if not cue_visible(
            state.episode_seed, state.step_count, observer_id, a.agent_id, infected
        ):
            continue

        # Concrete cue text. We never reveal infection_role to other agents.
        if infected:
            if a.hunger > 8:
                cues.append(f"A{a.agent_id} seems unusually hungry (hunger={a.hunger})")
            elif state.step_count > 20:
                cues.append(f"A{a.agent_id}'s hunger is rising faster than expected")
            else:
                cues.append(f"A{a.agent_id} is acting strangely")
        else:
            if a.hunger > 10:
                cues.append(f"A{a.agent_id} is starving (hunger={a.hunger})")
            else:
                cues.append(f"A{a.agent_id} seems suspicious")  # false positive

    return cues
