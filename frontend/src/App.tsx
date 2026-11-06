import { AlertTriangle, Boxes, Camera, ChevronLeft, ChevronRight, Cpu, Gauge, LineChart, Map as MapIcon, Pause, Play, Radar, Route, ShieldCheck, type LucideIcon } from "lucide-react";
import { useEffect } from "react";
import { SelectionProvider, useSelection } from "./context";
import { Bev } from "./pages/Bev";
import { Cameras } from "./pages/Cameras";
import { Evaluation } from "./pages/Evaluation";
import { Future } from "./pages/Future";
import { Occupancy } from "./pages/Occupancy";
import { Overview } from "./pages/Overview";
import { Risk } from "./pages/Risk";
import { SystemPage } from "./pages/System";
import { Tracking } from "./pages/Tracking";
import { Loading, Notice, Status, fmt } from "./ui";

export const PAGES: { id: string; label: string; icon: LucideIcon; run: boolean }[] = [
  { id: "overview", label: "Overview", icon: Gauge, run: true }, { id: "cameras", label: "Camera studio", icon: Camera, run: true }, { id: "bev", label: "BEV and geometry", icon: MapIcon, run: true },
  { id: "occupancy", label: "Occupancy", icon: Boxes, run: true }, { id: "tracking", label: "Detection and tracking", icon: Radar, run: true }, { id: "future", label: "Future prediction", icon: Route, run: true },
  { id: "risk", label: "Risk and safety", icon: ShieldCheck, run: true }, { id: "evaluation", label: "Evaluation", icon: LineChart, run: false }, { id: "system", label: "System", icon: Cpu, run: false },
];

function SequencePicker() {
  const { sequences, run, setRun, sequence } = useSelection();
  if (!sequences) return null;
  const runs = sequences.flatMap((s) => s.runs.map((r) => ({ ...r, scene: s })));
  return (
    <div className="picker">
      <label>Sequence
        <select value={run ?? ""} onChange={(e) => setRun(e.target.value)}>
          {!runs.length && <option value="">No stored runs</option>}
          {sequences.map((s) => s.runs.length > 0 && (
            <optgroup key={s.name} label={`${s.name}  ${s.split}${s.night ? ", night" : ""}`}>
              {s.runs.map((r) => <option key={r.id} value={r.id}>{s.name}{r.fault ? ": front camera loss" : ""}</option>)}
            </optgroup>
          ))}
        </select>
      </label>
      {sequence && <span className={`split ${sequence.detector_saw_scene_in_training ? "split-train" : ""}`} title={sequence.status_note}>{sequence.split}{sequence.night ? ", night" : ""}</span>}
    </div>
  );
}

function FrameBar() {
  const { timeline, target: frame, setFrame, playing, setPlaying, record } = useSelection();
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).matches("input, select, textarea")) return;
      if (e.key === "ArrowRight") setFrame(Math.min(frame + 1, (timeline?.length ?? 1) - 1));
      if (e.key === "ArrowLeft") setFrame(Math.max(frame - 1, 0));
      if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [frame, timeline, playing, setFrame, setPlaying]);
  if (!timeline) return <div className="framebar"><Loading what="timeline" /></div>;
  const n = timeline.length, start = timeline[0].t, row = timeline[frame];
  return (
    <footer className="framebar">
      <div className="transport">
        <button className="icon" aria-label="Previous frame" onClick={() => setFrame(Math.max(frame - 1, 0))}><ChevronLeft size={17} /></button>
        <button className="icon primary" aria-label={playing ? "Pause replay" : "Play recorded frames"} onClick={() => setPlaying(!playing)}>{playing ? <Pause size={16} /> : <Play size={16} />}</button>
        <button className="icon" aria-label="Next frame" onClick={() => setFrame(Math.min(frame + 1, n - 1))}><ChevronRight size={17} /></button>
      </div>
      <div className="strip">
        <div className="strip-cells">{timeline.map((r) => <i key={r.frame} className={`cell-${r.status} ${r.healthy_cameras < 6 ? "cell-degraded" : ""}`} title={`Frame ${r.frame}: ${r.status}`} />)}</div>
        <input type="range" min={0} max={n - 1} value={frame} aria-label="Frame" onChange={(e) => setFrame(Number(e.target.value))} />
      </div>
      <dl className="readout">
        <div><dt>Frame</dt><dd>{frame + 1} / {n}</dd></div>
        <div><dt>Replay time</dt><dd>{fmt(row.t - start, 1)} s</dd></div>
        <div><dt>Logged speed</dt><dd>{frame === 0 ? "–" : `${fmt(row.speed, 1)} m/s`}</dd></div>
        <div><dt>Filter decision</dt><dd>{record ? <Status value={record.decision.status} /> : "…"}</dd></div>
      </dl>
    </footer>
  );
}

const REPLAY_NOTICE = "Recorded nuScenes replay — open-loop evaluation. Decisions do not control subsequent frames.";

function ReplayBanner() {
  const { detail, sequence } = useSelection();
  if (!detail) return null;
  const replay = detail.mode === "open_loop_replay" || detail.mode === "fault_injection";
  return (
    <div className="replay" role="note">
      <strong>{replay ? REPLAY_NOTICE : detail.caveat}</strong>
      {detail.mode === "fault_injection" && <span>Declared test fault: one real camera was removed for part of this drive.</span>}
      {sequence && <span>{sequence.description}</span>}
    </div>
  );
}

function Workspace() {
  const { page, run, sequencesError, detail, sequences, go } = useSelection();
  const def = PAGES.find((p) => p.id === page);
  if (!def) return <Notice title="Page not found" tone="warn"><button className="link" onClick={() => go("overview")}>Go to the overview</button></Notice>;
  if (page === "evaluation") return <Evaluation />;
  if (page === "system") return <SystemPage />;
  if (sequencesError) return <Notice title="Backend unavailable" tone="error">{sequencesError}. Start it with scripts/run_app.sh.</Notice>;
  if (sequences && !run) return <Notice title="No stored runs" tone="warn">Open System and process a sequence, or run scripts/run_showcase.sh.</Notice>;
  if (!detail) return <Loading what="run" />;
  return { overview: <Overview />, cameras: <Cameras />, bev: <Bev />, occupancy: <Occupancy />, tracking: <Tracking />, future: <Future />, risk: <Risk /> }[page] ?? null;
}

function Shell() {
  const { page, go, detail } = useSelection();
  const def = PAGES.find((p) => p.id === page);
  return (
    <div className="app">
      <nav className="rail" aria-label="Workspaces">
        <div className="brand"><strong>Tavyra</strong><span>Research prototype</span></div>
        {PAGES.map((p, i) => { const Icon = p.icon; return (
          <button key={p.id} className="nav" aria-current={p.id === page ? "page" : undefined} onClick={() => go(p.id)}><Icon size={17} strokeWidth={1.8} aria-hidden />{p.label}{i === 6 && <span className="sep" />}</button>
        ); })}
        <p className="rail-note"><AlertTriangle size={13} aria-hidden />Recorded data only. Not a safety system or vehicle controller.</p>
      </nav>
      <div className="main">
        <header className="topbar"><h1>{def?.label ?? "Tavyra"}</h1><SequencePicker /></header>
        {def?.run && detail && <ReplayBanner />}
        <main><Workspace /></main>
        {def?.run && <FrameBar />}
      </div>
    </div>
  );
}

export function App() { return <SelectionProvider><Shell /></SelectionProvider>; }
