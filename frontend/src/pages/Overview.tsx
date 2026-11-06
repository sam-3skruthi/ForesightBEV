import { useSelection } from "../context";
import { Share } from "../charts";
import { useApi } from "../hooks";
import { Scene, StateKey } from "../Scene";
import type { Json } from "../types";
import { Loading, Panel, REASONS, Stat, Status, Unavailable, fmt, pretty } from "../ui";

const STEPS = [["cameras", "Six synchronised cameras"], ["cameras", "Calibration and health"], ["bev", "Bird's-eye projection"], ["occupancy", "Occupancy and reliability"], ["tracking", "3D detection"], ["tracking", "Tracked agents"],
  ["future", "Future motion"], ["risk", "Candidate plans"], ["risk", "Safety decision"], ["evaluation", "Stored evaluation"]] as const;

export function Overview() {
  const { run, frame, record, detail, go, sequence, recordError } = useSelection();
  const system = useApi<Json>("/system");
  const experiments = useApi<Json[]>("/experiments/index");
  if (recordError) return <Unavailable what="Frame unavailable" why={recordError} />;
  if (!record || !detail || !run) return <Loading what="frame" />;
  const f = record.state_fractions, healthy = Object.values(record.cameras).filter((c) => c.status === "ok").length;
  const total = f.FREE + f.OCCUPIED + f.UNKNOWN + f.OUTSIDE_FOV;
  const reasons = record.decision.reasons;
  const warnings = [
    healthy < 6 && `${6 - healthy} camera lost at this frame. Its coverage is unknown, not free.`,
    sequence?.detector_saw_scene_in_training && "The detector trained on this scene, so its quality here is inflated.",
    sequence?.night && "Night scene. Segmentation was trained on daytime data, and occupancy is weaker here.",
    !record.ground_plane && "Too few detected objects for a ground fit at this frame. Using flat ground.",
      ].filter(Boolean) as string[];
  const last = experiments.data?.at(-1);
  const ms = Object.values(record.timings_ms).reduce((a, b) => a + b, 0);
  return (
    <div className="split-layout">
      <section className="stage">
        <Scene run={run} frame={frame} record={record} grid={detail.grid} overlays={{ tracks: true, trails: true, candidates: "decision", forecast: 2 }} />
        <StateKey />
      </section>
      <div className="side">
        <section className="decision">
          <Status value={record.decision.status} size="large" />
          <ul className="reasons">{reasons.map((r) => <li key={r}>{REASONS[r] ?? r}</li>)}</ul>
          <small>Filter selected <b>{pretty(record.decision.selected)}</b> for this recorded frame</small>
        </section>
        <div className="stats three">
          <Stat label="Cameras healthy" value={`${healthy} / 6`} />
          <Stat label="Active tracks" value={record.tracks.filter((t) => t.confirmed).length} />
          <Stat label="Detections" value={record.detections.length} />
          <Stat label="Forecast horizon" value={fmt(record.forecast.horizons.at(-1), 1)} unit="s" />
          <Stat label="Pipeline time" value={fmt(ms, 0)} unit="ms" />
          <Stat label="Logged speed" value={frame === 0 ? "–" : fmt(record.ego.speed, 1)} unit={frame === 0 ? undefined : "m/s"} />
        </div>
        <Panel title="Observation of the 80 × 44 m grid" note={`Model-asserted occupied cells this frame: ${record.model_occupied_cells}. Ground: ${record.ground_plane ? "local plane" : "flat"}.`}>
          <Share parts={[{ label: "Free", value: f.FREE / total, tone: "k-free" }, { label: "Occupied", value: f.OCCUPIED / total, tone: "k-occ" }, { label: "Unknown", value: f.UNKNOWN / total, tone: "k-unk" }, { label: "No coverage", value: f.OUTSIDE_FOV / total, tone: "k-out" }]} />
        </Panel>
        {warnings.length > 0 && <Panel title="Warnings"><ul className="warnings">{warnings.map((w) => <li key={w}>{w}</li>)}</ul></Panel>}
        <Panel title="Pipeline" note={`Detector: ${detail.provenance.find((p) => p.component === "Detection")?.kind ?? "unknown"}. Last stored evaluation: ${experiments.error ? "unavailable" : !experiments.data ? "loading" : last ? `${last.name} (${last.status})` : "none stored"}.`}>
          <ol className="steps">{STEPS.map(([page, text], i) => <li key={i}><button className="link" onClick={() => go(page)}>{text}</button></li>)}</ol>
        </Panel>
        {system.data && <Panel title="Integrations"><ul className="kv">{system.data.integrations.map((i: Json) => <li key={i.name}><span>{i.name}</span><b className={i.status === "verified" ? "ok" : "off"}>{i.status}</b></li>)}</ul></Panel>}
      </div>
    </div>
  );
}
