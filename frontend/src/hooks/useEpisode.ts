import { useEffect, useMemo, useRef, useState } from "react";
import { EpisodeRunner } from "../sim/runner";
import { RemoteEpisodeRunner, type RemoteStatus } from "../sim/remoteRunner";
import type { EpisodeState } from "../sim/types";

// HF Space deployments. Pick by key in BackendConfig.
export const HF_SPACE_ENDPOINTS: Record<string, { label: string; url: string; desc: string }> = {
  "zombiee": {
    label: "zombiee",
    url: "https://noanya-zombiee.hf.space",
    desc: "OpenEnv API · v1",
  },
  "zombiee-v1-extended": {
    label: "zombiee-v1-extended",
    url: "https://noanya-zombiee-v1-extended.hf.space",
    desc: "API + browser runner · v1",
  },
};

export type BackendKind = "local" | "hf";

export interface BackendConfig {
  kind: BackendKind;
  spaceKey?: string; // key into HF_SPACE_ENDPOINTS, required when kind === "hf"
}

export interface UseEpisodeOpts {
  seed?: number;
  speed?: number;
  autoStart?: boolean;
  loopOnEnd?: boolean;
  backend?: BackendConfig;
}

export interface UseEpisodeReturn {
  state: EpisodeState;
  play: () => void;
  pause: () => void;
  step: () => void;
  reset: (s?: number) => void;
  setSpeed: (sps: number) => void;
  playing: boolean;
  remoteStatus: RemoteStatus | null;
  remoteError: string | null;
  isRemote: boolean;
}

export function useEpisode(opts: UseEpisodeOpts = {}): UseEpisodeReturn {
  const { seed, speed = 4, autoStart = true, loopOnEnd = false } = opts;
  const backend: BackendConfig = opts.backend ?? { kind: "local" };

  const [, setTick] = useState(0);
  const [playing, setPlayingState] = useState(autoStart);
  const [remoteStatus, setRemoteStatus] = useState<RemoteStatus | null>(null);
  const [remoteError, setRemoteError] = useState<string | null>(null);
  const localRef = useRef<EpisodeRunner | null>(null);
  const remoteRef = useRef<RemoteEpisodeRunner | null>(null);

  // Lazily build the local runner once (used as both default & fallback for state shape)
  if (!localRef.current) {
    localRef.current = new EpisodeRunner({
      seed, stepsPerSecond: speed,
      onTick: () => setTick((t) => t + 1),
      onEnd: () => {
        if (backend.kind === "local") {
          setPlayingState(false);
          if (loopOnEnd) {
            setTimeout(() => {
              localRef.current?.reset();
              localRef.current?.start();
              setPlayingState(true);
            }, 1500);
          }
        }
      },
    });
  }

  // Spin up / tear down the remote runner when backend changes
  useEffect(() => {
    if (backend.kind !== "hf") {
      remoteRef.current?.destroy();
      remoteRef.current = null;
      setRemoteStatus(null);
      setRemoteError(null);
      // hand control back to the local runner
      if (autoStart) {
        localRef.current?.start();
        setPlayingState(true);
      }
      return;
    }

    // HF backend: pause local, build a remote runner pointing at the chosen Space
    localRef.current?.stop();
    const space = backend.spaceKey ? HF_SPACE_ENDPOINTS[backend.spaceKey] : undefined;
    if (!space) {
      setRemoteError(`Unknown space key: ${backend.spaceKey}`);
      return;
    }
    setRemoteError(null);
    const r = new RemoteEpisodeRunner({
      baseUrl: space.url,
      seed,
      stepsPerSecond: speed,
      onTick: () => setTick((t) => t + 1),
      onEnd: () => {
        setPlayingState(false);
        if (loopOnEnd) {
          setTimeout(() => {
            r.reset();
            if (autoStart) {
              r.start();
              setPlayingState(true);
            }
          }, 1500);
        }
      },
      onStatus: (s) => setRemoteStatus(s),
      onError: (e) => setRemoteError(e.message),
    });
    remoteRef.current = r;
    // Kick off a fresh episode against the Space
    (async () => {
      await r.reset(seed);
      if (autoStart) {
        r.start();
        setPlayingState(true);
      } else {
        setPlayingState(false);
      }
    })();

    return () => {
      r.destroy();
      if (remoteRef.current === r) remoteRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backend.kind, backend.spaceKey]);

  // Mount once: start local if that's the initial backend
  useEffect(() => {
    if (backend.kind === "local" && autoStart) localRef.current?.start();
    return () => {
      localRef.current?.stop();
      remoteRef.current?.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isRemote = backend.kind === "hf";

  const api: UseEpisodeReturn = useMemo(() => {
    const active = isRemote ? remoteRef.current : localRef.current;
    return {
      state: (active?.state ?? localRef.current!.state) as EpisodeState,
      play: () => {
        active?.start();
        setPlayingState(true);
      },
      pause: () => {
        active?.stop();
        setPlayingState(false);
      },
      step: () => {
        active?.tick();
      },
      reset: (s?: number) => {
        if (isRemote) {
          remoteRef.current?.reset(s);
        } else {
          localRef.current?.reset(s);
        }
        setTick((t) => t + 1);
      },
      setSpeed: (sps: number) => active?.setSpeed(sps),
      playing,
      remoteStatus: isRemote ? remoteStatus : null,
      remoteError: isRemote ? remoteError : null,
      isRemote,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, isRemote, remoteStatus, remoteError, backend.spaceKey]);

  return api;
}
