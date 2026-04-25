import { useEffect, useMemo, useRef, useState } from "react";
import { EpisodeRunner } from "../sim/runner";
import { RemoteEpisodeRunner } from "../sim/remoteEngine";
import type { EpisodeState } from "../sim/types";

export type EpisodeMode = "local" | "remote";

export interface UseEpisodeOpts {
  seed?: number;
  speed?: number;
  autoStart?: boolean;
  loopOnEnd?: boolean;
  mode?: EpisodeMode;
}

type AnyRunner = EpisodeRunner | RemoteEpisodeRunner;

export function useEpisode(opts: UseEpisodeOpts = {}) {
  const { seed, speed = 4, autoStart = true, loopOnEnd = false, mode = "local" } = opts;
  // `tick` participates in the api memo deps so callers see fresh state when
  // the remote runner replaces `this.state` (the local runner mutates in
  // place; the remote one cannot — every /step response is a fresh object).
  const [tick, setTick] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlayingState] = useState(autoStart);
  const runnerRef = useRef<AnyRunner | null>(null);
  const lastModeRef = useRef<EpisodeMode | null>(null);

  const buildRunner = (m: EpisodeMode): AnyRunner => {
    const onTick = () => setTick(t => t + 1);
    const onEnd = () => {
      setPlayingState(false);
      if (loopOnEnd) {
        setTimeout(() => {
          void runnerRef.current?.reset();
          runnerRef.current?.start();
          setPlayingState(true);
        }, 1500);
      }
    };
    if (m === "remote") {
      return new RemoteEpisodeRunner({
        seed, stepsPerSecond: speed,
        onTick, onEnd,
        onError: (e) => setError(e.message),
      });
    }
    return new EpisodeRunner({
      seed, stepsPerSecond: speed,
      onTick, onEnd,
    });
  };

  // Synchronous lazy init — guarantees runnerRef.current.state is non-null
  // by the time the first render reads it. Without this, components like
  // <Grid> crash on `state.zombies` because the runner doesn't exist yet.
  if (!runnerRef.current) {
    runnerRef.current = buildRunner(mode);
    lastModeRef.current = mode;
  }

  // On mount: kick off auto-play. On unmount: stop the runner.
  useEffect(() => {
    if (autoStart) runnerRef.current?.start();
    return () => runnerRef.current?.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mode flip: tear down the old runner, build the new one, restart.
  useEffect(() => {
    if (lastModeRef.current === mode) return; // no-op on initial render
    runnerRef.current?.stop();
    setError(null);
    runnerRef.current = buildRunner(mode);
    if (autoStart) runnerRef.current.start();
    setPlayingState(autoStart);
    setTick(t => t + 1);
    lastModeRef.current = mode;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const api = useMemo(() => ({
    state: runnerRef.current!.state as EpisodeState,
    play: () => { runnerRef.current?.start(); setPlayingState(true); },
    pause: () => { runnerRef.current?.stop(); setPlayingState(false); },
    step: () => { void runnerRef.current?.tick(); },
    reset: (s?: number) => { void runnerRef.current?.reset(s); setTick(t => t + 1); },
    setSpeed: (sps: number) => runnerRef.current?.setSpeed(sps),
    playing,
    mode,
    error,
  }), [playing, mode, error, tick]);

  return api;
}
