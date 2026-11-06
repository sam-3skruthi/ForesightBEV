import { useState } from "react";
import { Lines } from "../charts";
import { useSelection } from "../context";
import { useApi } from "../hooks";
import { Scene, StateKey } from "../Scene";
import type { Json } from "../types";
import { Loading, Panel, Segmented, Stat, Toggle, Unavailable, fmt, pct } from "../ui";

const NAMES: Record<string, string> = { flat_no_tolerance: "Flat ground", "current_tolerance_0.4": "Previous baseline", pose_tilt_correction: "Pose tilt", "tolerance_0.8": "More tolerance", local_plane_from_detections: "Local plane", current_plus_fused_boxes: "Baseline + model boxes", local_plane_plus_fused_boxes: "Local plane + model boxes" };

export function Occupancy() {
  const { run, frame, record, detail, recordError } = useSelection();
  const evaluation = useApi<Json>("/evaluation");
  const [view, setView] = useState<"compare" | "after" | "before">("compare");
  const [truth, setTruth] = useState(true);
  if (recordError) return <Unavailable what="Frame unavailable" why={recordError} />;
  if (!record || !detail || !run) return <Loading what="frame" />;
  const d = record.diagnostics;
  const variants = evaluation.data?.real?.occupancy_variants;
  const overlays = { truth };
  return (
    <div className="stack">
      <div className="toolbar">
        <Segmented label="Occupancy view" value={view} onChange={setView} options={[["compare", "Before and after"], ["after", "Current"], ["before", "Previous"]] as const} />
        <Toggle checked={truth} onChange={setTruth} disabled={!record.ground_truth_for_display_only}>Annotated boxes (display only)</Toggle>
      </div>
      <div className="occupancy-row">
        {view !== "after" && <figure className="labelled"><figcaption>Previous: flat ground</figcaption><Scene run={run} frame={frame} record={record} grid={detail.grid} base="bev_original" overlays={overlays} /></figure>}
        {view !== "before" && <figure className="labelled"><figcaption>Current: local plane with model boxes</figcaption><Scene run={run} frame={frame} record={record} grid={detail.grid} overlays={overlays} /></figure>}
        <div className="side">
          <StateKey />
          <Panel title="This frame" note="Annotated box cells marked free: a spatial proxy on dynamic-object footprints. nuScenes mini has no occupancy labels, so this is not voxel accuracy and static scenery is unscored.">
            {d ? <div className="stats">
              <Stat label="Box cells" value={d.box_cells} />
              <Stat label="Marked free, previous" value={d.box_cells ? pct(d.box_cells_free_before / d.box_cells) : "–"} />
              <Stat label="Marked free, current" value={d.box_cells ? pct(d.box_cells_free_after / d.box_cells) : "–"} />
              <Stat label="Free area, current vs previous" value={d.free_cells_before ? pct(d.free_cells_after / d.free_cells_before) : "–"} />
            </div> : <Unavailable what="No annotations" why="This run has no annotation diagnostics." />}
          </Panel>
          <Panel title="Occupied sources" note="Geometric cells come from camera rays. Model-asserted cells come from confident 3D detections and are kept as a separate layer.">
            <div className="stats"><Stat label="Model-asserted cells" value={record.model_occupied_cells} /><Stat label="Ground model" value={record.ground_plane ? "Local plane" : "Flat"} /></div>
          </Panel>
        </div>
      </div>
      {variants ? (
        <div className="grid-2">
          <Panel title="Marked free by distance" note={`${variants.scene_status}: ${variants.scenes.join(", ")}. Two scenes, preliminary.`}>
            <Lines label="Box cells marked free by distance" x={Object.keys(variants.variants.flat_no_tolerance.by_range)} xLabel="Distance" yLabel="Share marked free" yMax={0.7}
              series={[{ name: "Flat ground", tone: 2, dash: true, values: Object.values(variants.variants.flat_no_tolerance.by_range).map((r: any) => r.free) }, { name: "Previous baseline", tone: 3, values: Object.values(variants.variants["current_tolerance_0.4"].by_range).map((r: any) => r.free) }, { name: "Current", tone: 1, values: Object.values(variants.variants.local_plane_plus_fused_boxes.by_range).map((r: any) => r.free) }]} />
          </Panel>
          <Panel title="Corrections compared" note="Free area is the cost side: marking nothing free would score perfectly and be useless.">
            <table><thead><tr><th>Variant</th><th className="n">Marked free</th><th className="n">Occupied</th><th className="n">Free m²</th></tr></thead>
              <tbody>{Object.entries<any>(variants.variants).map(([k, v]) => <tr key={k} className={k === "local_plane_plus_fused_boxes" ? "row-on" : ""}><td>{NAMES[k] ?? k}</td><td className="n">{fmt(v.annotated_box_cells.free, 3)}</td><td className="n">{fmt(v.annotated_box_cells.occupied, 3)}</td><td className="n">{fmt(v.free_area_m2_per_frame, 0)}</td></tr>)}</tbody></table>
          </Panel>
        </div>
      ) : <Unavailable what="Distance diagnostics not stored" why="Run foresightbev.evaluation.occupancy_diagnostics." />}
    </div>
  );
}
