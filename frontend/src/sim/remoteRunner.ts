// RemoteEpisodeRunner — drop-in replacement for EpisodeRunner that delegates
// the env to a deployed SurviveCity HF Space (POST /reset, POST /step).
// The local heuristic policy still picks actions client-side; only the
// environment state is computed on the server. This is what the BackendPicker
// flips between when the user selects an HF Space.

import { createMemory, decide } from "./policy";
import { RNG } from "./rng";
import { obsToState, placeholderState, type RemoteAdapterCarry, type RemoteObs } from "./obsToState";
import type { Action, EpisodeState } from "./types";

export interface RemoteRunnerOptions {
  baseUrl: string;
  seed?: number;
  stepsPerSecond?: number;
  onTick?: (state: EpisodeState) => void;
  onEnd?: (state: EpisodeState) => void;
  onError?: (err: Error) => void;
  onStatus?: (status: RemoteStatus) => void;
}

export interface RemoteStatus {
  kind: "idle" | "connecting" | "ready" | "stepping" | "error";
  message?: string;
  latencyMs?: number;
}

const ABORT_TIMEOUT_MS = 12_000;

async function fetchJson(url: string, init?: RequestInit): Promise<unknown> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ABORT_TIMEOUT_MS);
  try {
    const r = await fetch(url, { ...init, signal: ctrl.signal });
    if (!r.ok) {
      const body = await r.text().catch(() => "");
      throw new Error(`HTTP ${r.status} — ${body.slice(0, 160)}`);
    }
    return await r.json();
  } finally {
    clearTimeout(t);
  }
}

export class RemoteEpisodeRunner {
  state: EpisodeState;
  baseUrl: string;
  status: RemoteStatus = { kind: "idle" };
  private carry: RemoteAdapterCarry;
  private mem = createMemory();
  private rng: RNG;
  private timer: ReturnType<typeof setInterval> | null = null;
  private opts: RemoteRunnerOptions;
  private inFlight = false;
  private currentAgentId = 0;
  private destroyed = false;

  constructor(opts: RemoteRunnerOptions) {
    this.opts = { stepsPerSecond: 4, ...opts };
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    const seed = opts.seed ?? Math.floor(Math.random() * 1e9);
    this.rng = new RNG(seed);
    this.state = placeholderState(seed);
    this.carry = { rngSeed: seed, postmortems: [], knownInfectedId: undefined };
  }

  destroy() {
    this.destroyed = true;
    this.stop();
  }

  private setStatus(s: RemoteStatus) {
    this.status = s;
    this.opts.onStatus?.(s);
  }

  async reset(seed?: number) {
    this.stop();
    if (this.destroyed) return;
    const s = seed ?? Math.floor(Math.random() * 1e9);
    this.rng = new RNG(s);
    this.mem = createMemory();
    this.carry = { rngSeed: s, postmortems: [], knownInfectedId: undefined };
    this.currentAgentId = 0;
    this.setStatus({ kind: "connecting", message: "POST /reset…" });
    const t0 = performance.now();
    try {
      const obs = (await fetchJson(`${this.baseUrl}/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: s }),
      })) as RemoteObs;
      this.absorb(obs);
      const dt = performance.now() - t0;
      this.setStatus({ kind: "ready", message: `space online · ${Math.round(dt)}ms`, latencyMs: dt });
      this.opts.onTick?.(this.state);
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      this.setStatus({ kind: "error", message: e.message.slice(0, 120) });
      this.opts.onError?.(e);
    }
  }

  start() {
    if (this.timer || this.destroyed) return;
    const interval = 1000 / (this.opts.stepsPerSecond ?? 4);
    this.timer = setInterval(() => {
      // fire-and-forget; tick guards re-entrancy with inFlight
      this.tick();
    }, interval);
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  setSpeed(stepsPerSecond: number) {
    this.opts.stepsPerSecond = stepsPerSecond;
    if (this.timer) {
      this.stop();
      this.start();
    }
  }

  // One macro tick = enough /step calls for the server's step_count to advance
  // (i.e. all alive agents act once + zombies advance). Capped at 4 sub-calls
  // to prevent runaway loops if the env never advances.
  async tick() {
    if (this.destroyed || this.state.done || this.inFlight) return;
    this.inFlight = true;
    this.setStatus({ kind: "stepping", message: `POST /step (A${this.currentAgentId})…` });
    const startStep = this.state.step;
    const t0 = performance.now();

    try {
      for (let i = 0; i < 4; i++) {
        const action: Action = decide(this.state, this.currentAgentId, this.mem, this.rng);
        const obs = (await fetchJson(`${this.baseUrl}/step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            agent_id: this.currentAgentId,
            action_type: action.type,
            vote_target: action.voteTarget ?? null,
            message: action.message ?? null,
          }),
        })) as RemoteObs;
        this.absorb(obs);
        if (this.state.step > startStep || this.state.done) break;
      }
      const dt = performance.now() - t0;
      this.setStatus({
        kind: "ready",
        message: `step ${this.state.step}/${this.state.maxSteps} · ${Math.round(dt)}ms`,
        latencyMs: dt,
      });
      this.opts.onTick?.(this.state);
      if (this.state.done) {
        this.stop();
        this.opts.onEnd?.(this.state);
      }
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      this.setStatus({ kind: "error", message: e.message.slice(0, 120) });
      this.opts.onError?.(e);
      this.stop();
    } finally {
      this.inFlight = false;
    }
  }

  private absorb(obs: RemoteObs) {
    const prevAgents = this.state?.agents;
    this.state = obsToState(obs, { ...this.carry, prevAgents });
    this.carry = {
      rngSeed: this.carry.rngSeed,
      postmortems: this.state.postmortems,
      knownInfectedId:
        typeof obs.metadata?.infected_id === "number"
          ? obs.metadata.infected_id
          : this.carry.knownInfectedId,
    };
    this.currentAgentId = obs.metadata?.current_agent_id ?? 0;
  }
}
