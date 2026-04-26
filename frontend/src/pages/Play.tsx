import { useMemo, useState } from "react";
import { Grid } from "../components/Grid";
import { AgentRoster, BroadcastFeed, PhaseBar, PostmortemFeed } from "../components/Hud";
import { Icon } from "../components/Icon";
import { HF_SPACE_ENDPOINTS, useEpisode, type BackendConfig } from "../hooks/useEpisode";

type BackendKey = "local" | keyof typeof HF_SPACE_ENDPOINTS | string;

function configFromKey(key: BackendKey): BackendConfig {
  if (key === "local") return { kind: "local" };
  return { kind: "hf", spaceKey: key };
}

export default function Play() {
  const [backendKey, setBackendKey] = useState<BackendKey>("local");
  const [fog, setFog] = useState(true);
  const [speed, setSpeed] = useState(4);
  const [seedInput, setSeedInput] = useState("42");

  const backendConfig = useMemo(() => configFromKey(backendKey), [backendKey]);
  const ep = useEpisode({ seed: 42, speed: 4, autoStart: true, loopOnEnd: false, backend: backendConfig });

  const handleSpeed = (s: number) => { setSpeed(s); ep.setSpeed(s); };
  const handleReset = () => {
    const n = Number(seedInput);
    ep.reset(Number.isFinite(n) ? n : Math.floor(Math.random() * 1e9));
  };

  return (
    <div className="max-w-7xl mx-auto px-5 lg:px-8 py-10">
      <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
        <div>
          <div className="mono-label">/play · interactive episode</div>
          <h1 className="mt-1 text-3xl font-display tracking-tight">Run the environment</h1>
          <p className="text-ink-2 mt-1 text-sm max-w-xl">
            The same engine that powers training — recompiled to TypeScript so you can step through it in your browser
            with no GPU. Or point it at a deployed Hugging Face Space to drive the real Python env over HTTP.
            Toggle <em>fog</em> to hide the infected role from the audience.
          </p>
        </div>
        <BackendPicker
          backend={backendKey}
          onChange={setBackendKey}
          isRemote={ep.isRemote}
          status={ep.remoteStatus}
          error={ep.remoteError}
        />
      </div>

      <div className="grid lg:grid-cols-[auto_1fr] gap-6 items-start">
        <div className="panel p-4 sm:p-6">
          <div className="flex items-center justify-between mb-3">
            <div className="mono-label">grid · 10×10</div>
            <div className="chip">
              <span className={`w-1.5 h-1.5 rounded-full ${ep.playing ? "bg-neon-lime animate-pulse" : "bg-ink-3"}`} />
              {ep.playing ? "running" : "paused"}
            </div>
          </div>
          <Grid state={ep.state} cell={48} hideInfectedTag={fog} />
          <Controls
            playing={ep.playing}
            onPlay={ep.play} onPause={ep.pause} onStep={ep.step} onReset={handleReset}
            speed={speed} onSpeed={handleSpeed}
            seedInput={seedInput} onSeedInput={setSeedInput}
            fog={fog} onFog={setFog}
          />
        </div>

        <div className="space-y-4">
          <PhaseBar state={ep.state} />
          <div className="panel p-4">
            <div className="mono-label mb-3">agents</div>
            <AgentRoster state={ep.state} fog={fog} />
            {fog && (
              <div className="mt-3 text-[11px] font-mono text-ink-3">
                ▸ infected role hidden until step 30 reveal — same as the LLM sees.
              </div>
            )}
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            <BroadcastFeed state={ep.state} />
            <PostmortemFeed state={ep.state} />
          </div>
          <ApiPreview state={ep.state} />
        </div>
      </div>

      <EndCard state={ep.state} onReset={handleReset} />
    </div>
  );
}

function Controls({
  playing, onPlay, onPause, onStep, onReset,
  speed, onSpeed, seedInput, onSeedInput, fog, onFog,
}: {
  playing: boolean;
  onPlay: () => void; onPause: () => void; onStep: () => void; onReset: () => void;
  speed: number; onSpeed: (n: number) => void;
  seedInput: string; onSeedInput: (s: string) => void;
  fog: boolean; onFog: (b: boolean) => void;
}) {
  const speeds = [1, 2, 4, 8, 16];
  return (
    <div className="mt-4 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {playing ? (
          <button onClick={onPause} className="btn btn-ghost"><Icon.Pause size={14} /> Pause</button>
        ) : (
          <button onClick={onPlay} className="btn btn-primary"><Icon.Play size={14} /> Play</button>
        )}
        <button onClick={onStep} className="btn btn-ghost"><Icon.SkipFwd size={14} /> Step</button>
        <button onClick={onReset} className="btn btn-ghost"><Icon.Reset size={14} /> Reset</button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <span className="mono-label">speed</span>
        <div className="inline-flex rounded-lg overflow-hidden border border-ink-4/40">
          {speeds.map(s => (
            <button key={s} onClick={() => onSpeed(s)}
              className={`px-2.5 py-1.5 text-xs font-mono cursor-pointer transition-colors ${
                speed === s ? "bg-neon-violet/30 text-ink-0" : "text-ink-2 hover:bg-ink-4/30"
              }`}>
              {s}×
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="mono-label" htmlFor="seed">seed</label>
        <input id="seed" value={seedInput} onChange={e => onSeedInput(e.target.value)}
          className="bg-bg-2 border border-ink-4/40 rounded-lg px-2.5 py-1.5 font-mono text-xs w-28
            focus:outline-none focus:border-neon-violet" />
        <label className="inline-flex items-center gap-2 cursor-pointer select-none">
          <input type="checkbox" checked={fog} onChange={e => onFog(e.target.checked)}
            className="accent-neon-violet" />
          <span className="text-xs text-ink-2">fog of war</span>
        </label>
      </div>
    </div>
  );
}

function BackendPicker({
  backend, onChange, isRemote, status, error,
}: {
  backend: string;
  onChange: (b: string) => void;
  isRemote: boolean;
  status: { kind: string; message?: string; latencyMs?: number } | null;
  error: string | null;
}) {
  const opts: { key: string; label: string; desc: string }[] = [
    { key: "local", label: "Browser engine", desc: "TS port · 0 GPU · always available" },
    ...Object.entries(HF_SPACE_ENDPOINTS).map(([key, v]) => ({
      key,
      label: v.label,
      desc: `${v.desc} · ${v.url}`,
    })),
  ];

  // Status pill colour
  const statusColor =
    error ? "bg-neon-rose/70" :
    status?.kind === "ready" ? "bg-neon-lime" :
    status?.kind === "stepping" || status?.kind === "connecting" ? "bg-neon-amber animate-pulse" :
    "bg-ink-3";
  const statusText =
    error ? "space error" :
    !isRemote ? "browser engine" :
    status?.message ?? "—";

  return (
    <div className="flex flex-col gap-1.5 items-end">
      <div className="panel p-3 flex items-center gap-2 flex-wrap">
        <div className="flex items-center gap-2 pr-2 border-r border-ink-4/30">
          <Icon.Cpu size={16} className="text-neon-purple" />
          <span className="mono-label">backend</span>
        </div>
        {opts.map(o => (
          <button key={o.key} onClick={() => onChange(o.key)}
            className={`px-3 py-1.5 rounded-lg text-xs cursor-pointer transition-colors ${
              backend === o.key
                ? "bg-neon-violet/25 text-ink-0 border border-neon-violet/40"
                : "text-ink-2 hover:bg-ink-4/20 border border-transparent"
            }`}
            title={o.desc}>
            {o.label}
          </button>
        ))}
      </div>
      <div className="chip text-[11px] font-mono">
        <span className={`w-1.5 h-1.5 rounded-full ${statusColor}`} />
        <span className={error ? "text-neon-rose" : "text-ink-2"} title={error ?? statusText}>
          {statusText}
        </span>
        {isRemote && status?.latencyMs !== undefined && !error && (
          <span className="text-ink-3">· {Math.round(status.latencyMs)}ms</span>
        )}
      </div>
    </div>
  );
}

function ApiPreview({ state }: { state: any }) {
  const lastBroadcast = state.broadcasts[state.broadcasts.length - 1];
  const sampleAction = lastBroadcast
    ? { agent_id: lastBroadcast.agentId, action_type: "broadcast", message: lastBroadcast.text }
    : { agent_id: state.step % 3, action_type: "move_right" };
  return (
    <div className="panel p-4">
      <div className="mono-label mb-3">openenv api · last step</div>
      <div className="grid md:grid-cols-2 gap-3">
        <CodeBlock title="POST /step" body={JSON.stringify(sampleAction, null, 2)} accent="rose" />
        <CodeBlock title="response · observation" body={JSON.stringify({
          step_count: state.step,
          phase: state.phase,
          done: state.done,
          reward: 0.5 + (state.agents[0].alive ? 0.05 : -0.5),
          metadata: { current_agent_id: (state.step + 1) % 3, healthy_alive: state.agents.filter((a:any)=>a.alive && !a.infected).length },
        }, null, 2)} accent="cyan" />
      </div>
    </div>
  );
}

function CodeBlock({ title, body, accent }: { title: string; body: string; accent: "rose" | "cyan" }) {
  return (
    <div className="rounded-xl bg-bg-0/60 border border-ink-4/40 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-ink-4/30 bg-bg-1/60">
        <div className={`text-[11px] font-mono uppercase tracking-wider text-neon-${accent}`}>{title}</div>
        <div className="flex gap-1">
          <span className="w-2 h-2 rounded-full bg-neon-rose/60" />
          <span className="w-2 h-2 rounded-full bg-neon-amber/60" />
          <span className="w-2 h-2 rounded-full bg-neon-lime/60" />
        </div>
      </div>
      <pre className="p-3 font-mono text-[11.5px] text-ink-1 overflow-x-auto leading-5">{body}</pre>
    </div>
  );
}

function EndCard({ state, onReset }: { state: any; onReset: () => void }) {
  if (!state.done) return null;
  const healthyAlive = state.agents.filter((a:any)=>a.alive && !a.infected).length;
  const won = healthyAlive >= 1;
  return (
    <div className={`mt-6 panel p-6 border-2 ${won ? "border-neon-lime/40" : "border-neon-rose/40"}`}>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className={`mono-label ${won ? "text-neon-lime" : "text-neon-rose"}`}>episode terminal</div>
          <div className="font-display text-2xl mt-1">
            {won ? "Healthy survivors hold the safehouse." : "All healthy agents fell. Infected wins."}
          </div>
          <div className="text-sm text-ink-2 mt-1">
            Step {state.step}/100 · {state.postmortems.length} post-mortems generated for next episode.
          </div>
        </div>
        <button className="btn btn-primary" onClick={onReset}>
          <Icon.Reset size={14} /> Run another episode
        </button>
      </div>
    </div>
  );
}
