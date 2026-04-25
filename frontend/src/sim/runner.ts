import { advanceStep, advanceZombies, applyAgentAction, createEpisode } from "./engine";
import { createMemory, decide } from "./policy";
import { RNG } from "./rng";
import type { EpisodeState } from "./types";

export interface RunnerOptions {
  seed?: number;
  stepsPerSecond?: number;
  onTick?: (state: EpisodeState) => void;
  onEnd?: (state: EpisodeState) => void;
}

export class EpisodeRunner {
  state: EpisodeState;
  private mem = createMemory();
  private rng: RNG;
  private timer: ReturnType<typeof setInterval> | null = null;
  private opts: RunnerOptions;
  private agentCursor = 0;

  constructor(opts: RunnerOptions = {}) {
    this.opts = { stepsPerSecond: 4, ...opts };
    const seed = opts.seed ?? Math.floor(Math.random() * 1e9);
    this.state = createEpisode(seed);
    this.rng = new RNG(seed);
  }

  reset(seed?: number) {
    this.stop();
    const s = seed ?? Math.floor(Math.random() * 1e9);
    this.state = createEpisode(s);
    this.mem = createMemory();
    this.rng = new RNG(s);
    this.agentCursor = 0;
    this.opts.onTick?.(this.state);
  }

  start() {
    if (this.timer) return;
    const interval = 1000 / (this.opts.stepsPerSecond ?? 4);
    this.timer = setInterval(() => this.tick(), interval);
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }

  setSpeed(stepsPerSecond: number) {
    this.opts.stepsPerSecond = stepsPerSecond;
    if (this.timer) { this.stop(); this.start(); }
  }

  // One full step = each living agent acts once + zombies advance + step++
  tick() {
    if (this.state.done) {
      this.stop();
      this.opts.onEnd?.(this.state);
      return;
    }

    // Clear pulses each macro tick
    this.state.pulses = [];

    for (let i = 0; i < 3; i++) {
      const a = this.state.agents[i];
      if (!a.alive) continue;
      const action = decide(this.state, i, this.mem, this.rng);
      applyAgentAction(this.state, action);
    }
    advanceZombies(this.state);
    advanceStep(this.state);
    this.opts.onTick?.(this.state);
  }
}
