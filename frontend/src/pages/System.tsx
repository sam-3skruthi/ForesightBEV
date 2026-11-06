import { useEffect, useState } from "react";
import { post } from "../api";
import { useSelection } from "../context";
import { useApi } from "../hooks";
import type { Json } from "../types";
import { Loading, Notice, Panel, Tag, Unavailable, fmt } from "../ui";

export function SystemPage() {
  const { sequences, sequence, reload } = useSelection();
  const [tick, setTick] = useState(0);
  const system = useApi<Json>("/system", tick), status = useApi<Json>("/status", tick), logs = useApi<Json[]>("/logs?lines=30", tick), jobs = useApi<Json[]>("/jobs", tick);
  const [config, setConfig] = useState("pipeline_showcase.yaml");
  const effective = useApi<Json>(`/config/pipeline/${config}`);
  const [scene, setScene] = useState("");
  const [armed, setArmed] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const active = jobs.data?.find((j) => j.status === "running");
  useEffect(() => { if (!active) return; const id = setInterval(() => setTick((t) => t + 1), 3000); return () => clearInterval(id); }, [active?.id]);
  useEffect(() => { if (jobs.data?.[0]?.status === "succeeded") reload(); }, [jobs.data?.[0]?.id, jobs.data?.[0]?.status]); // eslint-disable-line
  if (system.error) return <Notice title="Backend unavailable" tone="error">{system.error}. Start it with scripts/run_app.sh.</Notice>;
  if (!system.data || !status.data) return <Loading what="system status" />;
  const s = system.data;
  const chosen = scene || sequence?.name || sequences?.[0]?.name || "";
  const start = async () => {
    setArmed(false); setMessage(null);
    try { const job = await post<Json>("/jobs", { kind: "showcase", params: { scene: chosen } }); setMessage(`Started job ${job.id}.`); setTick((t) => t + 1); }
    catch (e) { setMessage((e as Error).message); }
  };
  return (
    <div className="stack page-scroll">
      <div className="grid-3">
        <Panel title="Backend" note={`Version ${status.data.version}. Source ${String(status.data.environment.source_sha256).slice(0, 12)}.`}>
          <ul className="kv">{status.data.checks.map((c: Json) => <li key={c.name} title={c.detail}><span>{c.name.replace(/_/g, " ")}</span><b className={c.ok ? "ok" : "off"}>{c.ok ? "ok" : "missing"}</b></li>)}</ul>
        </Panel>
        <Panel title="Integrations" note="Not executed means no real server or stack ran. It is not claimed as integrated.">
          <ul className="kv">{s.integrations.map((i: Json) => <li key={i.name} title={i.detail}><span>{i.name}</span><b className={i.status === "verified" ? "ok" : "off"}>{i.status}</b></li>)}</ul>
          <p className="note">{s.ros_runs.filter((r: Json) => r.matches_in_process).length} ROS 2 runs identical to in-process. Environment {s.ros_environment_present ? "present" : "missing"}.</p>
        </Panel>
        <Panel title="Dataset" note={s.dataset.root}>
          <ul className="kv"><li><span>Version</span><b>{s.dataset.version}</b></li><li><span>Tables found</span><b className={s.dataset.tables_found ? "ok" : "off"}>{String(s.dataset.tables_found)}</b></li><li><span>Scenes, samples</span><b>{s.dataset.scenes ?? "–"}, {s.dataset.samples ?? "–"}</b></li><li><span>Occupancy labels</span><b className="off">none</b></li></ul>
        </Panel>
      </div>
      <Panel wide title="Models">
        <div className="scroll"><table><thead><tr><th>Model</th><th>Kind</th><th>Available</th><th>Used in</th><th>Licence</th></tr></thead><tbody>{s.models.map((m: Json) => <tr key={m.name}><td>{m.name}</td><td><Tag kind={m.kind.startsWith("trained") ? "trained" : "pretrained"}>{m.kind}</Tag></td><td>{m.available ? "yes" : "no"}</td><td>{m.used_in}</td><td>{m.licence}</td></tr>)}</tbody></table></div>
      </Panel>
      <div className="grid-2">
        <Panel title="Process a sequence" note="Runs the full pipeline on real frames and replaces that sequence's stored run. Thresholds are not editable here.">
          <div className="form row">
            <label>Sequence<select value={chosen} onChange={(e) => { setScene(e.target.value); setArmed(false); }}>{sequences?.map((q) => <option key={q.name}>{q.name}</option>)}</select></label>
            {!armed ? <button className="primary-button" disabled={!!active} onClick={() => setArmed(true)}>{active ? "A job is running" : "Process"}</button> : <button className="primary-button danger" onClick={start}>Replace run for {chosen}</button>}
          </div>
          {message && <p className="note">{message}</p>}
          {jobs.data && jobs.data.length > 0 && <div className="scroll short"><table><thead><tr><th>Job</th><th>Kind</th><th>Status</th><th className="n">Seconds</th></tr></thead><tbody>{jobs.data.slice(0, 6).map((j) => <tr key={j.id}><td>{j.id.slice(9, 15)}</td><td>{j.kind}</td><td>{j.status}</td><td className="n">{j.finished_unix ? fmt(j.finished_unix - j.started_unix, 0) : "–"}</td></tr>)}</tbody></table></div>}
        </Panel>
        <Panel title="Configuration" note="Read-only view. Changes are made in files under configs/ and validated at load." actions={<select aria-label="Configuration file" value={config} onChange={(e) => setConfig(e.target.value)}>{s.pipeline_configs.map((c: string) => <option key={c}>{c}</option>)}</select>}>
          {effective.error ? <Unavailable what="Configuration invalid" why={effective.error} /> : effective.data ? <ul className="kv">
            <li><span>Detector</span><b>{effective.data.effective.detector}</b></li><li><span>Ground model</span><b>{effective.data.effective.ground_plane}</b></li><li><span>Model boxes fused</span><b>{String(effective.data.effective.fuse_model_boxes)}</b></li>
            <li><span>Max event score</span><b>{effective.data.effective.safety.max_event_score}</b></li><li><span>Max perception age</span><b>{effective.data.effective.safety.max_perception_age_s} s</b></li><li><span>Phantom speed</span><b>{effective.data.effective.risk.phantom_speed} m/s</b></li><li><span>Model velocity in tracker</span><b>{String(effective.data.effective.tracker.use_model_velocity)}</b></li>
          </ul> : <Loading what="configuration" />}
        </Panel>
      </div>
      <Panel wide title="Recent log entries" actions={<button className="link" onClick={() => setTick((t) => t + 1)}>Refresh</button>}>
        {logs.data && logs.data.length ? <div className="scroll logs">{[...logs.data].reverse().map((l, i) => <div key={i} className={`log log-${l.level}`}><time>{String(l.time ?? "").slice(11, 19)}</time><b>{l.level}</b><span>{l.message === "request" ? `${l.method} ${l.path} ${l.status} ${l.duration_ms} ms` : l.message}</span></div>)}</div> : <Unavailable what="No log entries" why="Nothing has been logged yet." />}
      </Panel>
    </div>
  );
}
