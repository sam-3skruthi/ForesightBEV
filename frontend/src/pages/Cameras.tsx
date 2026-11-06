import { useEffect, useState } from "react";
import { camera, layer } from "../api";
import { useSelection } from "../context";
import { useApi } from "../hooks";
import type { CameraView } from "../types";
import { Loading, Panel, Segmented, Stat, Unavailable, fmt } from "../ui";

const ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"];
const short = (n: string) => { const s = n.replace("CAM_", "").replace(/_/g, " ").toLowerCase(); return s[0].toUpperCase() + s.slice(1); };

export function Cameras() {
  const { run, frame } = useSelection();
  const views = useApi<CameraView[]>(run ? `/runs/${run}/frames/${frame}/cameras` : null);
  // Images and statuses are always drawn together, from one frame.
  const [shown, setShown] = useState<{ run: string; frame: number; views: CameraView[] } | null>(null);
  useEffect(() => { if (views.data && run) setShown({ run, frame, views: views.data }); }, [views.data]); // eslint-disable-line
  const [mode, setMode] = useState<"rgb" | "labels">("rgb");
  const [pick, setPick] = useState("CAM_FRONT");
  if (views.error) return <Unavailable what="Cameras unavailable" why={views.error} />;
  if (!shown || shown.run !== run) return <Loading what="cameras" />;
  const map = Object.fromEntries(shown.views.map((v) => [v.name, v]));
  const selected = map[pick] ?? shown.views[0];
  const c = selected.calibration;
  return (
    <div className="split-layout wide-stage">
      <section className="stage">
        <div className="toolbar"><Segmented label="Camera view" value={mode} onChange={setMode} options={[["rgb", "Camera images"], ["labels", "Segmentation"]] as const} /></div>
        <div className="camera-grid">
          {ORDER.map((name) => {
            const v = map[name];
            const lost = !v || v.status !== "ok";
            return (
              <button key={name} className={`camera ${name === pick ? "camera-picked" : ""} ${lost ? "camera-lost" : ""}`} onClick={() => setPick(name)} aria-pressed={name === pick}>
                {lost ? <div className="camera-empty"><strong>Camera lost</strong><span>{v?.status ?? "no data"} at this frame</span></div> : <img src={mode === "rgb" ? camera(run, shown.frame, name) : layer(run, shown.frame, "labels", name)} alt={`${short(name)} camera at frame ${shown.frame}`} />}
                <span className="camera-label"><b>{short(name)}</b>{v && <span>{fmt(v.skew_ms, 1)} ms</span>}</span>
              </button>
            );
          })}
        </div>
      </section>
      <div className="side">
        <Panel title={short(selected.name)} note="Skew is measured from the front camera's exposure. Health is the usable-label fraction, a heuristic. It is not an OOD score.">
          <div className="stats">
            <Stat label="Status" value={selected.status === "ok" ? "OK" : selected.status} />
            <Stat label="Label health" value={fmt(selected.reliability)} />
            <Stat label="Skew" value={fmt(selected.skew_ms, 1)} unit="ms" />
          </div>
          <ul className="kv spaced"><li><span>Capture time</span><b>{fmt(selected.timestamp, 3)} s</b></li></ul>
        </Panel>
        <Panel title="Calibration" note="Validated at load. Intrinsics are scaled with the 800×450 processing size.">
          <ul className="kv">
            <li><span>Image size</span><b>{c.width} × {c.height}</b></li>
            <li><span>Focal length</span><b>{fmt(c.fx, 1)} px</b></li>
            <li><span>Horizontal FOV</span><b>{fmt(c.horizontal_fov_deg, 1)}°</b></li>
            <li><span>Origin, ego (m)</span><b>{c.origin_ego.map((v) => fmt(v)).join(", ")}</b></li>
            <li><span>Optical axis, ego</span><b>{c.forward_ego.map((v) => fmt(v)).join(", ")}</b></li>
          </ul>
        </Panel>
        <Panel title="All cameras">
          <table><thead><tr><th>Camera</th><th>Status</th><th className="n">Skew ms</th></tr></thead>
            <tbody>{shown.views.map((v) => <tr key={v.name} className={v.name === pick ? "row-on" : ""} onClick={() => setPick(v.name)}><td>{short(v.name)}</td><td>{v.status}</td><td className="n">{fmt(v.skew_ms, 1)}</td></tr>)}</tbody></table>
        </Panel>
      </div>
    </div>
  );
}
