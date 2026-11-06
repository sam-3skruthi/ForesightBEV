import { useState } from "react";
import { Compare } from "../charts";
import { useSelection } from "../context";
import { useApi } from "../hooks";
import { Scene } from "../Scene";
import type { Json, Pooled } from "../types";
import { Loading, Panel, Segmented, Stat, Toggle, Unavailable, fmt } from "../ui";

export function Tracking() {
  const { run, frame, record, detail, recordError } = useSelection();
  const evaluation = useApi<Json>("/evaluation");
  const [show, setShow] = useState<"tracks" | "detections">("tracks");
  const [truth, setTruth] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);
  if (recordError) return <Unavailable what="Frame unavailable" why={recordError} />;
  if (!record || !detail || !run) return <Loading what="frame" />;
  const tracks = record.tracks.filter((t) => t.confirmed).sort((a, b) => Math.hypot(a.x, a.y) - Math.hypot(b.x, b.y));
  const detector = detail.provenance.find((p) => p.component === "Detection");
  const base: Pooled | null = evaluation.data?.real?.baseline ?? null, now: Pooled | null = evaluation.data?.real?.showcase ?? null;
  const perClass = evaluation.data?.real?.per_class;
  return (
    <div className="split-layout">
      <section className="stage">
        <div className="toolbar">
          <Segmented label="Show" value={show} onChange={setShow} options={[["tracks", "Tracks"], ["detections", "Detections"]] as const} />
          <Toggle checked={truth} onChange={setTruth}>Annotations</Toggle>
        </div>
        <Scene run={run} frame={frame} record={record} grid={detail.grid} overlays={{ tracks: show === "tracks", trails: show === "tracks", detections: show === "detections", truth, selectedTrack: picked, onTrack: setPicked }} />
        <div className="state-key"><span><i className="k-veh" />Vehicle</span><span><i className="k-ped" />Person</span><span><i className="k-truth" />Annotation</span></div>
      </section>
      <div className="side">
        <Panel title="This frame" note={detector?.detail}>
          <div className="stats"><Stat label="Detections" value={record.detections.length} /><Stat label="Confirmed tracks" value={tracks.length} /><Stat label="Coasting" value={tracks.filter((t) => t.misses > 0).length} /><Stat label="Tentative" value={record.tracks.length - tracks.length} /></div>
        </Panel>
        <Panel title="Tracks" note="Velocity comes from the Kalman filter, updated with the model's velocity when configured. Scores are raw network outputs, not probabilities.">
          <div className="scroll tall"><table><thead><tr><th>ID</th><th>Class</th><th className="n">x, y m</th><th className="n">Speed</th><th className="n">σ pos</th><th className="n">Age</th></tr></thead>
            <tbody>{tracks.map((t) => <tr key={t.id} className={t.id === picked ? "row-on" : ""} onClick={() => setPicked(t.id)}><td>{t.id}</td><td>{t.cls.toLowerCase()}</td><td className="n">{fmt(t.x, 1)}, {fmt(t.y, 1)}</td><td className="n">{fmt(Math.hypot(t.vx, t.vy), 1)}</td><td className="n">{fmt(t.position_std)}</td><td className="n">{t.age}</td></tr>)}</tbody></table></div>
        </Panel>
        {base && now ? (
          <Panel title="Baseline against StreamPETR" note={`mini_val, ${now.pooled.frames} frames, previously seen, preliminary. 2 m matching within 40 m.`}>
            <Compare a="Apparent size" b="StreamPETR" rows={[
              { label: "Vehicle recall", a: base.pooled.detection.VEHICLE.recall_2m, b: now.pooled.detection.VEHICLE.recall_2m }, { label: "Vehicle precision", a: base.pooled.detection.VEHICLE.precision_2m, b: now.pooled.detection.VEHICLE.precision_2m },
              { label: "Person recall", a: base.pooled.detection.PERSON.recall_2m, b: now.pooled.detection.PERSON.recall_2m }, { label: "Person precision", a: base.pooled.detection.PERSON.precision_2m, b: now.pooled.detection.PERSON.precision_2m }, { label: "Track recall", a: base.pooled.tracking.recall_2m, b: now.pooled.tracking.recall_2m }]} />
            <ul className="kv">
              <li><span>Velocity error</span><b>{fmt(base.pooled.tracking.mean_velocity_error_mps)} → {fmt(now.pooled.tracking.mean_velocity_error_mps)} m/s</b></li>
              <li><span>ID switches per 100 matched</span><b>{fmt(100 * base.pooled.tracking.id_switches / base.pooled.tracking.matched, 1)} → {fmt(100 * now.pooled.tracking.id_switches / now.pooled.tracking.matched, 1)}</b></li>
              <li><span>NEES, consistent near 4</span><b>{fmt(base.pooled.tracking.mean_nees_4dof, 1)} → {fmt(now.pooled.tracking.mean_nees_4dof, 1)}</b></li>
            </ul>
          </Panel>
        ) : <Unavailable what="Comparison not stored" why="Run the mini_val evaluation." />}
        {perClass && <Panel title="Per class, StreamPETR">
          <table><thead><tr><th>Class</th><th className="n">Annotated</th><th className="n">Recall</th><th className="n">Precision</th></tr></thead>
            <tbody>{Object.entries<any>(perClass.per_class).filter(([, v]) => v.annotated > 0).map(([k, v]) => <tr key={k}><td>{k}</td><td className="n">{v.annotated}</td><td className="n">{fmt(v.recall, 3)}</td><td className="n">{fmt(v.precision, 3)}</td></tr>)}</tbody></table>
        </Panel>}
      </div>
    </div>
  );
}
