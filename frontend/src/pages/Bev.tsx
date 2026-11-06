import { useState } from "react";
import { layer } from "../api";
import { useSelection } from "../context";
import { Scene } from "../Scene";
import { Loading, Panel, Segmented, Stat, Unavailable, fmt } from "../ui";

const CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"];

export function Bev() {
  const { run, frame, record, detail, recordError } = useSelection();
  const [view, setView] = useState<"ipm" | "coverage">("ipm");
  if (recordError) return <Unavailable what="Frame unavailable" why={recordError} />;
  if (!record || !detail || !run) return <Loading what="frame" />;
  const p = record.ground_plane;
  return (
    <div className="split-layout">
      <section className="stage">
        <div className="toolbar"><Segmented label="Combined view" value={view} onChange={setView} options={[["ipm", "Combined BEV"], ["coverage", "Camera coverage"]] as const} /></div>
        <Scene run={run} frame={frame} record={record} grid={detail.grid} base={view} overlays={{}} />
        {view === "coverage" ? <div className="ramp"><span>0</span><i /><span>6 cameras</span></div> : <div className="state-key"><span><i className="l-road" />Road</span><span><i className="l-walk" />Sidewalk</span><span><i className="l-terrain" />Other ground</span><span><i className="l-static" />Static</span><span><i className="k-veh" />Vehicle</span><span><i className="k-ped" />Person</span></div>}
      </section>
      <div className="side">
        <Panel title="Ground model" note={p ? "Fitted per frame from the bottoms of confidently detected objects, with no LiDAR or annotations. A coarse tilt and offset, not a road surface." : "Too few detected objects at this frame, so the pipeline keeps flat ground."}>
          <div className="stats">
            <Stat label="Mode" value={p ? "Local plane" : "Flat"} />
            <Stat label="Height at ego" value={p ? fmt(p.a) : "0.00"} unit="m" />
            <Stat label="Forward slope" value={p ? fmt(p.b * 100, 1) : "0.0"} unit="%" />
            <Stat label="Lateral slope" value={p ? fmt(p.c * 100, 1) : "0.0"} unit="%" />
          </div>
        </Panel>
        <Panel title="Coverage" note="Share of grid cells each camera count can see on the ground plane. Overlap regions are narrow in this rig.">
          <div className="stats">
            <Stat label="One camera" value={fmt(record.coverage.single * 100, 1)} unit="%" />
            <Stat label="Overlap" value={fmt(record.coverage.overlap * 100, 1)} unit="%" />
            <Stat label="Uncovered" value={fmt(record.coverage.none * 100, 1)} unit="%" />
          </div>
        </Panel>
        <Panel title="Individual projections" note="Each camera's labels placed on flat ground. Cars smear away from the camera because only the road lies on that plane.">
          <div className="projections">
            {CAMERAS.map((c) => <figure key={c}>{record.cameras[c]?.status === "ok" ? <img src={layer(run, frame, "projection", c)} alt={`${c} projection`} /> : <div className="camera-empty small">lost</div>}<figcaption>{c.replace("CAM_", "").replace(/_/g, " ").toLowerCase()}</figcaption></figure>)}
          </div>
        </Panel>
      </div>
    </div>
  );
}
