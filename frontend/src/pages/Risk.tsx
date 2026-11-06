import { useEffect, useState } from "react";
import { post } from "../api";
import { useSelection } from "../context";
import { Scene, StateKey } from "../Scene";
import type { PlanResult } from "../types";
import { Loading, Panel, REASONS, Stat, Status, Toggle, Unavailable, fmt, pct, pretty } from "../ui";

export function Risk() {
  const { run, frame, record, detail, recordError } = useSelection();
  const [picked, setPicked] = useState<string | null>(null);
  const [all, setAll] = useState(true);
  const [speed, setSpeed] = useState(8);
  const [lateral, setLateral] = useState(0);
  const [plan, setPlan] = useState<{ frame: number; value: PlanResult } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setPlan(null); setError(null); }, [run, frame]);
  if (recordError) return <Unavailable what="Frame unavailable" why={recordError} />;
  if (!record || !detail || !run) return <Loading what="frame" />;
  const nominalRejected = record.decision.status !== "pass";
  const rows = [...record.candidates].sort((a, b) => Number(b.id === record.decision.selected) - Number(a.id === record.decision.selected) || Number(b.nominal) - Number(a.nominal) || a.event_score - b.event_score);
  const chosen = record.candidates.find((c) => c.id === (picked ?? record.decision.selected));
  const evaluate = async () => {
    setBusy(true); setError(null);
    try { setPlan({ frame, value: await post<PlanResult>(`/runs/${run}/frames/${frame}/plan`, { target_speed: speed, lateral_offset: lateral }) }); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const mine = plan?.frame === frame ? plan.value : null;
  const user = mine?.risks.find((r) => r.candidate_id === "user_plan");
  return (
    <div className="split-layout">
      <section className="stage">
        <div className="toolbar"><Toggle checked={all} onChange={setAll}>All candidates</Toggle></div>
        <Scene run={run} frame={frame} record={record} grid={detail.grid} overlays={{ tracks: true, candidates: all ? "all" : "decision", picked, onPick: setPicked, extraPath: mine?.path ?? null }} />
        <StateKey />
      </section>
      <div className="side">
        <section className="decision">
          <Status value={record.decision.status} size="large" />
          {nominalRejected && <p>Nominal plan <Status value="reject" />, replaced by <b>{pretty(record.decision.selected)}</b>.</p>}
          <ul className="reasons">{record.decision.reasons.map((r) => <li key={r}>{REASONS[r] ?? r}</li>)}</ul>
        </section>
        <Panel title="Candidates" note="Event score is the share of forecast samples where the plan overlaps a predicted agent. It is a model frequency, not a collision probability.">
          <div className="scroll tall"><table><thead><tr><th>Plan</th><th className="n">Score</th><th className="n">Clearance</th><th className="n">TTC</th><th>Flags</th></tr></thead>
            <tbody>{rows.map((c) => <tr key={c.id} onClick={() => setPicked(c.id)} className={c.id === record.decision.selected ? "row-on" : c.id === picked ? "row-picked" : ""}>
              <td>{pretty(c.id)}</td><td className="n">{fmt(c.event_score)}</td><td className="n">{c.min_clearance_m === null ? "–" : `${fmt(c.min_clearance_m, 1)} m`}</td><td className="n">{c.time_to_conflict_s === null ? "–" : `${fmt(c.time_to_conflict_s, 2)} s`}</td>
              <td>{[c.nominal && "nominal", c.static_collision && "occupied", c.unknown_violation && "unknown"].filter(Boolean).join(", ")}</td></tr>)}</tbody></table></div>
        </Panel>
        {chosen && <Panel title={pretty(chosen.id)} note="Clearance is an approximate gap to mean constant-velocity agent paths. TTC is the first conflict horizon under that model.">
          <div className="stats"><Stat label="Unknown exposure" value={pct(chosen.unknown_exposure)} /><Stat label="Stopping distance" value={fmt(chosen.stopping_distance, 1)} unit="m" /><Stat label="To unknown" value={chosen.distance_to_unknown === null ? "none" : fmt(chosen.distance_to_unknown, 0)} unit={chosen.distance_to_unknown === null ? undefined : "m"} /><Stat label="End speed" value={fmt(chosen.end_speed, 1)} unit="m/s" /></div>
        </Panel>}
        <Panel title="Test a plan on this recorded frame" note="Computed live against this frame's stored forecast, with this run's configuration.">
          <div className="form">
            <label>Target speed <output>{fmt(speed, 1)} m/s</output><input type="range" min={0} max={15} step={0.5} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} /></label>
            <label>Lateral offset <output>{fmt(lateral, 1)} m</output><input type="range" min={-4} max={4} step={0.5} value={lateral} onChange={(e) => setLateral(Number(e.target.value))} /></label>
            <button className="primary-button" onClick={evaluate} disabled={busy}>{busy ? "Evaluating" : "Evaluate plan"}</button>
          </div>
          {error && <p className="error-text" role="alert">{error}</p>}
          {mine && user && <div className="result">
            <Status value={mine.decision.selected === "user_plan" ? "pass" : "reject"} />
            <ul className="kv"><li><span>Event score</span><b>{fmt(user.event_score)}</b></li><li><span>Clearance</span><b>{user.min_clearance_m === null ? "–" : `${fmt(user.min_clearance_m, 1)} m`}</b></li><li><span>TTC</span><b>{user.time_to_conflict_s === null ? "none" : `${fmt(user.time_to_conflict_s, 2)} s`}</b></li><li><span>Filter selects</span><b>{pretty(mine.decision.selected)}</b></li></ul>
            {user.problems.length > 0 && <ul className="reasons">{user.problems.map((p) => <li key={p}>{REASONS[p] ?? p}</li>)}</ul>}
          </div>}
        </Panel>
      </div>
    </div>
  );
}
