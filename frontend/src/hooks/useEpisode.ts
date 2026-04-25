import { useEffect, useMemo, useRef, useState } from "react";
import { EpisodeRunner } from "../sim/runner";
import type { EpisodeState } from "../sim/types";

export interface UseEpisodeOpts {
  seed?: number;
  speed?: number;
  autoStart?: boolean;
  loopOnEnd?: boolean;
}

export function useEpisode(opts: UseEpisodeOpts = {}) {
  const { seed, speed = 4, autoStart = true, loopOnEnd = false } = opts;
  const [, setTick] = useState(0);
  const runnerRef = useRef<EpisodeRunner | null>(null);
  const playingRef = useRef(autoStart);
  const [playing, setPlayingState] = useState(autoStart);

  if (!runnerRef.current) {
    runnerRef.current = new EpisodeRunner({
      seed, stepsPerSecond: speed,
      onTick: () => setTick(t => t + 1),
      onEnd: () => {
        setPlayingState(false);
        playingRef.current = false;
        if (loopOnEnd) {
          setTimeout(() => {
            runnerRef.current?.reset();
            runnerRef.current?.start();
            playingRef.current = true;
            setPlayingState(true);
          }, 1500);
        }
      },
    });
  }

  useEffect(() => {
    if (autoStart) runnerRef.current?.start();
    return () => runnerRef.current?.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const api = useMemo(() => ({
    state: runnerRef.current!.state as EpisodeState,
    play: () => { runnerRef.current?.start(); playingRef.current = true; setPlayingState(true); },
    pause: () => { runnerRef.current?.stop(); playingRef.current = false; setPlayingState(false); },
    step: () => { runnerRef.current?.tick(); },
    reset: (s?: number) => { runnerRef.current?.reset(s); setTick(t => t + 1); },
    setSpeed: (sps: number) => runnerRef.current?.setSpeed(sps),
    playing,
  }), [playing]);
  return api;
}
