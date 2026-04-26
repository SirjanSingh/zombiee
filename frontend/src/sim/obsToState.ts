// Convert a SurviveCity HF Space /reset or /step JSON observation
// into the frontend's local EpisodeState shape so the same UI components
// can render either backend without changes.

import { createEpisode } from "./engine";
import type { Agent, Broadcast, EpisodeState, Phase, Zombie } from "./types";

interface RemoteAgent {
  agent_id: number;
  row: number;
  col: number;
  hp: number;
  hunger: number;
  is_alive: boolean;
  is_infected: boolean;
  locked_out: boolean;
  ate_this_step?: boolean;
  damage_this_step?: number;
  died_this_step?: boolean;
}

interface RemoteZombie {
  zombie_id: number;
  row: number;
  col: number;
}

export interface RemoteObs {
  grid?: string[][];
  agents: RemoteAgent[];
  zombies: RemoteZombie[];
  step_count: number;
  max_steps?: number;
  done: boolean;
  reward?: number;
  description?: string;
  broadcasts?: (string | Broadcast)[];
  metadata?: {
    phase?: string;
    current_agent_id?: number;
    infected_id?: number;
    postmortem?: string;
    postmortem_agent_id?: number;
    [k: string]: unknown;
  };
}

export interface RemoteAdapterCarry {
  // Things the server doesn't return on every step but we want stable across ticks
  rngSeed: number;
  postmortems: string[];
  knownInfectedId: number | undefined;
  prevAgents?: Agent[];
}

function derivePhase(step: number, done: boolean, fromMeta?: string): Phase {
  if (done) return "terminal";
  const m = (fromMeta || "").toLowerCase();
  if (m === "pre_reveal" || m === "post_reveal" || m === "vote" || m === "post_vote" || m === "terminal") {
    return m as Phase;
  }
  if (step < 30) return "pre_reveal";
  if (step < 50) return "post_reveal";
  if (step === 50) return "vote";
  return "post_vote";
}

export function obsToState(obs: RemoteObs, carry: RemoteAdapterCarry): EpisodeState {
  // Resolve infected id: trust metadata first, then scan agents (only the
  // infected agent itself is unmasked after reveal so this can stay stale).
  let infectedId = carry.knownInfectedId;
  if (typeof obs.metadata?.infected_id === "number") {
    infectedId = obs.metadata.infected_id;
  } else {
    const flagged = obs.agents.find((a) => a.is_infected);
    if (flagged) infectedId = flagged.agent_id;
  }

  const agents: Agent[] = [0, 1, 2].map((id) => {
    const r = obs.agents.find((x) => x.agent_id === id);
    const prev = carry.prevAgents?.[id];
    if (!r) {
      // Defensive: keep previous snapshot if server omitted (shouldn't happen)
      return prev ?? {
        id, row: 0, col: 0, hp: 0, hunger: 0,
        alive: false, infected: id === infectedId,
        infectionRevealed: false, lockedOut: false,
        ate: false, damage: 0, died: false,
      };
    }
    const infectionRevealed =
      (r.is_infected && id === infectedId) || (infectedId !== undefined && obs.step_count >= 30);
    return {
      id,
      row: r.row,
      col: r.col,
      hp: r.hp,
      hunger: r.hunger,
      alive: r.is_alive,
      infected: id === infectedId,
      infectionRevealed,
      lockedOut: r.locked_out,
      ate: !!r.ate_this_step,
      damage: r.damage_this_step ?? 0,
      died: !!r.died_this_step,
      deathStep: r.died_this_step ? obs.step_count : prev?.deathStep,
      deathCause: r.died_this_step ? "remote" : prev?.deathCause,
    };
  });

  const zombies: Zombie[] = [0, 1, 2].map((id) => {
    const z = obs.zombies.find((x) => x.zombie_id === id);
    return z ? { id, row: z.row, col: z.col } : { id, row: 0, col: 0 };
  });

  const phase = derivePhase(obs.step_count, obs.done, obs.metadata?.phase);

  // Broadcasts: server may return strings (legacy) or objects
  const broadcasts: Broadcast[] = (obs.broadcasts || [])
    .map((b): Broadcast | null => {
      if (typeof b === "string") {
        return { step: obs.step_count, agentId: 0, text: b };
      }
      if (b && typeof b === "object" && "text" in b) {
        return {
          step: typeof b.step === "number" ? b.step : obs.step_count,
          agentId: typeof b.agentId === "number" ? b.agentId : 0,
          text: String(b.text || "").slice(0, 40),
        };
      }
      return null;
    })
    .filter((x): x is Broadcast => x !== null);

  // Synthesize pulses from per-step flags so animations still trigger
  const pulses: EpisodeState["pulses"] = [];
  for (const a of agents) {
    if (a.died) pulses.push({ kind: "death", row: a.row, col: a.col, step: obs.step_count, agentId: a.id });
    if (a.ate) pulses.push({ kind: "eat", row: a.row, col: a.col, step: obs.step_count, agentId: a.id });
    if (a.damage > 0 && !a.died) {
      pulses.push({ kind: "attack", row: a.row, col: a.col, step: obs.step_count, agentId: a.id });
    }
  }

  // Vote phase derivation: any locked_out agent → vote resolved
  const lockedAgent = agents.find((a) => a.lockedOut);

  // Postmortems: append any new ones from metadata
  let postmortems = carry.postmortems;
  if (obs.metadata?.postmortem && !postmortems.includes(obs.metadata.postmortem)) {
    postmortems = [...postmortems, obs.metadata.postmortem];
  }

  return {
    step: obs.step_count,
    maxSteps: obs.max_steps ?? 100,
    agents,
    zombies,
    infectedId: infectedId ?? 0,
    done: obs.done,
    phase,
    votes: {},               // server holds the truth; UI doesn't need per-agent tally for HF backend
    voteResolved: !!lockedAgent,
    lockoutTarget: lockedAgent?.id,
    broadcasts,
    rngSeed: carry.rngSeed,
    postmortems,
    pulses,
  };
}

export function placeholderState(seed: number): EpisodeState {
  // Used while waiting for the first /reset response so the UI doesn't crash.
  const s = createEpisode(seed);
  // Hide infected info — we don't know it yet
  s.agents.forEach((a) => { a.infected = false; a.infectionRevealed = false; });
  s.infectedId = 0;
  return s;
}
