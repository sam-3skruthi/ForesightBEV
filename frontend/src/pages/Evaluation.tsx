import { useState } from "react";
import { Compare, Lines } from "../charts";
import { useApi } from "../hooks";
import type { Json, Metrics } from "../types";
import { Loading, Notice, Panel, Segmented, Stat, Tag, Unavailable, fmt } from "../ui";

const MODES = ["blind_no_filter", "idm_no_filter", "filter", "filter_calibrated", "filter_unknown_as_free", "filter_oracle_tracks", "idm_no_filter_front_dropout", "filter_front_dropout"];
const ci = (v: Json) => (v?.mean === null || v?.mean === undefined ? "–" : `${fmt(v.mean, 3)} [${fmt(v.low, 3)}, ${fmt(v.high, 3)}]`);

function Real({ r }: { r: Json }) {
  if (!r.showcase || !r.baseline) return <Unavailable what="mini_val results not stored" why="Run the mini_val evaluation." />;
  const now: Metrics = r.showcase.pooled, base: Metrics = r.baseline.pooled;
  return (
    <div className="stack">
      <Notice title="Preliminary, on previously seen scenes" tone="warn">Official mini_val: scene-0103 and scene-0916, {now.frames} frames, both daytime. Scene-0103 was used for tuning and scene-0916 was in an earlier reported set. Annotations are used for scoring only. nuScenes mini has no occupancy labels.</Notice>
      <div className="grid-2">
        <Panel title="Detection and tracking" note={`Vehicles ${now.detection.VEHICLE.annotated}, people ${now.detection.PERSON.annotated} annotated. 2 m match within 40 m, score cutoff 0.3 fixed before evaluation.`}>
          <Compare a="Apparent-size baseline" b="Showcase" rows={[{ label: "Vehicle recall", a: base.detection.VEHICLE.recall_2m, b: now.detection.VEHICLE.recall_2m }, { label: "Vehicle precision", a: base.detection.VEHICLE.precision_2m, b: now.detection.VEHICLE.precision_2m }, { label: "Person recall", a: base.detection.PERSON.recall_2m, b: now.detection.PERSON.recall_2m }, { label: "Person precision", a: base.detection.PERSON.precision_2m, b: now.detection.PERSON.precision_2m }, { label: "Track recall", a: base.tracking.recall_2m, b: now.tracking.recall_2m }]} />
        </Panel>
        <Panel title="Occupancy proxy and safety on logs" note={`Logged human path: ${now.safety.logged_path_frames} frames, ${now.safety.logged_path_true_contact} true contacts. A low alarm rate on safe driving says nothing about missed hazards.`}>
          <table><thead><tr><th>Measure</th><th className="n">Baseline</th><th className="n">Showcase</th></tr></thead><tbody>
            <tr><td>Box cells marked free (proxy)</td><td className="n">{fmt(base.occupancy_against_dynamic_boxes.box_cells_marked_free, 3)}</td><td className="n">{fmt(now.occupancy_against_dynamic_boxes.box_cells_marked_free, 3)}</td></tr>
            <tr><td>Boxes with occupied contact</td><td className="n">{fmt(base.occupancy_against_dynamic_boxes.boxes_with_dynamic_occupied_contact, 3)}</td><td className="n">{fmt(now.occupancy_against_dynamic_boxes.boxes_with_dynamic_occupied_contact, 3)}</td></tr>
            <tr><td>Alarms on human path</td><td className="n">{base.safety.logged_path_event_score_alarm}</td><td className="n">{now.safety.logged_path_event_score_alarm}</td></tr>
            <tr><td>Pass / modify / fallback</td><td className="n">{base.safety.decisions.pass ?? 0} / {base.safety.decisions.modify ?? 0} / {base.safety.decisions.fallback ?? 0}</td><td className="n">{now.safety.decisions.pass ?? 0} / {now.safety.decisions.modify ?? 0} / {now.safety.decisions.fallback ?? 0}</td></tr>
            <tr><td>Track velocity error, m/s</td><td className="n">{fmt(base.tracking.mean_velocity_error_mps)}</td><td className="n">{fmt(now.tracking.mean_velocity_error_mps)}</td></tr>
            <tr><td>ID switches</td><td className="n">{base.tracking.id_switches}</td><td className="n">{now.tracking.id_switches}</td></tr>
            <tr><td>Mean NEES</td><td className="n">{fmt(base.tracking.mean_nees_4dof, 1)}</td><td className="n">{fmt(now.tracking.mean_nees_4dof, 1)}</td></tr>
          </tbody></table>
        </Panel>
        {r.per_class && <Panel title="Score sweep" note="Reported only. No operating point was selected from this curve.">
          <Lines label="Precision and recall by score cutoff" x={Object.keys(r.per_class.score_sweep_reporting_only)} xLabel="Score cutoff" yLabel="Value" yMax={1} series={[{ name: "Recall", tone: 1, values: Object.values<any>(r.per_class.score_sweep_reporting_only).map((v) => v.recall) }, { name: "Precision", tone: 2, values: Object.values<any>(r.per_class.score_sweep_reporting_only).map((v) => v.precision) }]} />
        </Panel>}
        {r.tracking_study_mini_train && <Panel title="Tracker selection" note="Compared on mini_train, where detection figures are inflated, then scored once on mini_val. Two-stage association was rejected: it tripled false tracks.">
          <table><thead><tr><th>Variant</th><th className="n">False</th><th className="n">ID sw</th><th className="n">Vel err</th><th className="n">NEES</th></tr></thead>
            <tbody>{Object.entries<any>(r.tracking_study_mini_train.variants).map(([k, v]) => <tr key={k} className={k === "model_velocity" ? "row-on" : ""}><td>{k.replace(/_/g, " ")}</td><td className="n">{v.false_track_frames}</td><td className="n">{v.id_switches}</td><td className="n">{fmt(v.velocity_error)}</td><td className="n">{fmt(v.nees, 1)}</td></tr>)}</tbody></table>
        </Panel>}
      </div>
      {r.safety ? <Panel wide title="Interventions checked against annotations" note={r.safety.note}>
        <div className="scroll"><table><thead><tr><th>Unknown policy</th><th className="n">Frames</th><th className="n">Nominal plan conflicts</th><th className="n">Interventions</th><th className="n">With a conflict</th><th className="n">Without a conflict</th><th className="n">Conflicts passed</th></tr></thead>
          <tbody>{([["Conservative (used)", r.safety.unknown_conservative], ["Unknown as free", r.safety.unknown_as_free]] as const).map(([name, v]) => <tr key={name} className={name.endsWith("(used)") ? "row-on" : ""}><td>{name}</td><td className="n">{v.counts.frames}</td><td className="n">{v.counts.nominal_conflicts_in_annotations}</td><td className="n">{v.counts.interventions}</td><td className="n">{v.counts.interventions_with_conflict}</td><td className="n">{v.counts.interventions_without_conflict}</td><td className="n">{v.counts.conflicts_passed_without_intervention}</td></tr>)}</tbody></table></div>
        <p className="note">Only {r.safety.unknown_conservative.counts.nominal_conflicts_in_annotations} conflicts occur, so no rate can be estimated. Treating unknown as free changes interventions from {r.safety.unknown_conservative.counts.interventions} to {r.safety.unknown_as_free.counts.interventions} and conflicts passed from {r.safety.unknown_conservative.counts.conflicts_passed_without_intervention} to {r.safety.unknown_as_free.counts.conflicts_passed_without_intervention}.</p>
      </Panel> : <Unavailable what="Intervention study not stored" why="Run foresightbev.evaluation.safety_real." />}
      {r.heldout_classical && <Panel title="Classical pipeline, eight scenes" note="Without the pretrained detector. These scenes include night. The pipeline had not seen them when its configuration was frozen.">
        <div className="stats"><Stat label="Frames" value={r.heldout_classical.pooled.frames} /><Stat label="Vehicle recall" value={fmt(r.heldout_classical.pooled.detection.VEHICLE.recall_2m, 3)} /><Stat label="Person recall" value={fmt(r.heldout_classical.pooled.detection.PERSON.recall_2m, 3)} /><Stat label="Human path alarms" value={fmt(r.heldout_classical.pooled.safety.logged_path_alarm_rate, 3)} /></div>
      </Panel>}
      {r.dataset && <Panel title="Dataset validation"><div className="stats"><Stat label="Scenes" value={r.dataset.metadata.table_rows.scene} /><Stat label="Samples" value={r.dataset.metadata.table_rows.sample} /><Stat label="Annotations" value={r.dataset.metadata.table_rows.sample_annotation} /><Stat label="Problems" value={r.dataset.problems.length} /></div></Panel>}
    </div>
  );
}

function Synthetic({ s }: { s: Json }) {
  if (!s.closed_loop) return <Unavailable what="Simulator results not stored" why="Run the final evaluation." />;
  const modes = MODES.filter((m) => s.closed_loop.modes[m]);
  return (
    <div className="stack">
      <Notice title="Synthetic" tone="warn">{s.label} These figures do not transfer to real driving.</Notice>
      <Panel wide title="Closed loop, held-out seeds" note={`${s.closed_loop.modes.filter.overall.episodes} episodes per mode, seeds ${s.experiment?.closed_loop_seeds?.join(" to ")}. Brackets are 95% bootstrap intervals.`}>
        <div className="scroll"><table><thead><tr><th>Mode</th><th className="n">Collision rate</th><th className="n">Route completion</th><th className="n">Freeze rate</th></tr></thead>
          <tbody>{modes.map((m) => { const o = s.closed_loop.modes[m].overall; return <tr key={m} className={m === "filter" ? "row-on" : ""}><td>{m.replace(/_/g, " ")}</td><td className="n">{ci(o.collision_rate)}</td><td className="n">{ci(o.route_completion)}</td><td className="n">{ci(o.freeze_rate)}</td></tr>; })}</tbody></table></div>
      </Panel>
      <div className="grid-2">
        {s.calibration && <Panel title="Event-score calibration" note={`${s.calibration.test_rows} held-out rows, simulator only. Calibration here improved probabilities, not driving outcomes.`}>
          <table><thead><tr><th>Score</th><th className="n">Brier</th><th className="n">ECE</th><th className="n">AUROC</th></tr></thead><tbody>{(["raw", "calibrated"] as const).map((k) => <tr key={k}><td>{k}</td><td className="n">{fmt(s.calibration[k].brier, 4)}</td><td className="n">{fmt(s.calibration[k].ece_10_equal_width, 4)}</td><td className="n">{fmt(s.calibration[k].auroc, 3)}</td></tr>)}</tbody></table>
        </Panel>}
        {s.forecaster && <Panel title="Simulator forecasters" note={`${s.forecaster.train.rows_test} held-out frames at 1 m cells. Trained only on simulator data.`}>
          <Lines label="Agent IoU by horizon on simulator data" x={s.forecaster.horizons_s} xLabel="Horizon (s)" yLabel="IoU" yMax={1} series={[{ name: "U-Net", tone: 1, values: s.forecaster.models.learned_unet.iou }, { name: "Constant velocity", tone: 3, values: s.forecaster.models.constant_velocity_monte_carlo.iou }, { name: "Frozen", tone: 2, dash: true, values: s.forecaster.models.frozen_tracks.iou }]} />
        </Panel>}
      </div>
    </div>
  );
}

function History() {
  const records = useApi<Json[]>("/experiments/index"), sets = useApi<Json[]>("/experiments");
  if (records.error) return <Unavailable what="Experiment history unavailable" why={records.error} />;
  if (!records.data || !sets.data) return <Loading what="experiments" />;
  return (
    <div className="stack">
      <Panel wide title="Recorded experiments" note="Append-only. Each entry stores the source hash.">
        <div className="scroll"><table><thead><tr><th>Name</th><th>Kind</th><th>Status</th><th>Source</th></tr></thead>
          <tbody>{[...records.data].reverse().map((r, i) => <tr key={i}><td>{r.name}</td><td>{r.kind ?? "evaluation"}</td><td>{r.status}</td><td className="n">{String(r.source_sha256 ?? "").slice(0, 10) || "–"}</td></tr>)}</tbody></table></div>
      </Panel>
      <Panel wide title="Stored result sets"><div className="scroll"><table><thead><tr><th>Name</th><th className="n">Files</th><th>Report</th></tr></thead><tbody>{sets.data.map((s) => <tr key={s.name}><td>{s.name}</td><td className="n">{s.files.length}</td><td>{s.has_report ? "yes" : "–"}</td></tr>)}</tbody></table></div></Panel>
    </div>
  );
}

export function Evaluation() {
  const [tab, setTab] = useState<"real" | "synthetic" | "history">("real");
  const data = useApi<Json>("/evaluation");
  return (
    <div className="stack page-scroll">
      <div className="toolbar"><Segmented label="Evaluation source" value={tab} onChange={setTab} options={[["real", "Real footage"], ["synthetic", "Simulator"], ["history", "Experiment history"]] as const} />
        {data.data?.verification && <span className="verified"><Tag kind="good">{data.data.verification.filter((s: Json) => s.result === "pass").length} of {data.data.verification.length} checks</Tag> in the stored verification</span>}</div>
      {data.error && <Notice title="Evaluation unavailable" tone="error">{data.error}</Notice>}
      {!data.data && !data.error && <Loading what="evaluation" />}
      {data.data && tab === "real" && <Real r={data.data.real} />}
      {data.data && tab === "synthetic" && <Synthetic s={data.data.synthetic} />}
      {tab === "history" && <History />}
    </div>
  );
}
