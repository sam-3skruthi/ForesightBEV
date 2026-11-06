import { useEffect, useMemo, useState } from "react";
import { layer } from "./api";
import type { Candidate, FrameRecord, Grid } from "./types";

export interface Overlays { truth?: boolean; detections?: boolean; tracks?: boolean; trails?: boolean; forecast?: number | null; candidates?: "none" | "decision" | "all"; extraPath?: [number, number][] | null; picked?: string | null; selectedTrack?: number | null; onPick?: (id: string) => void; onTrack?: (id: number) => void }

const AGENT: Record<string, string> = { VEHICLE: "agent-vehicle", PERSON: "agent-person" };

/** Bird's-eye view. Metres, ego frame, forward up, left left. Overlays share the image's space. */
export function Scene({ run, frame, record, grid, base = "bev", overlays, label }: { run: string; frame: number; record: FrameRecord; grid: Grid; base?: string; overlays: Overlays; label?: string }) {
  const source = layer(run, frame, base);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => setFailed(null), [run]);
  const w = grid.y_max - grid.y_min, h = grid.x_max - grid.x_min;
  const sx = (y: number) => grid.y_max - y, sy = (x: number) => grid.x_max - x;
  const line = (p: [number, number][]) => p.map(([x, y]) => `${sx(y)},${sy(x)}`).join(" ");
  const box = (x: number, y: number, yaw: number, l: number, wd: number, cls: string, key: string | number, onClick?: () => void) => (
    <rect key={key} className={cls} x={sx(y) - wd / 2} y={sy(x) - l / 2} width={wd} height={l} rx={0.2} onClick={onClick} transform={`rotate(${(-yaw * 180) / Math.PI} ${sx(y)} ${sy(x)})`} />
  );
  const decided = record.decision.selected;
  const shown = useMemo(() => record.candidates.filter((c: Candidate) => overlays.candidates === "all" || (overlays.candidates === "decision" && (c.id === decided || c.nominal)) || c.id === overlays.picked), [record.candidates, overlays.candidates, overlays.picked, decided]);
  const ticks = [10, 20, 30, 40].filter((d) => d <= grid.x_max);
  return (
    <figure className="scene" aria-label={label ?? `Bird's-eye view at frame ${frame}`}>
      <div className="scene-stage" style={{ aspectRatio: `${w} / ${h}` }}>
        {failed === source ? <div className="scene-missing">Layer not stored for this frame</div> : <img src={source} alt="" onError={() => setFailed(source)} />}
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
          {ticks.map((d) => <g key={d} className="range"><line x1={0} x2={w} y1={sy(d)} y2={sy(d)} /><text x={0.6} y={sy(d) - 0.4}>{d} m</text></g>)}
          {overlays.truth && record.ground_truth_for_display_only?.filter((t) => t.cls).map((t) => box(t.x, t.y, t.yaw, t.length, t.width, t.scored ? "truth" : "truth truth-hidden", t.name))}
          {overlays.detections && record.detections.map((d, i) => box(d.x, d.y, d.yaw ?? 0, d.length, d.width, `det ${AGENT[d.cls] ?? ""}`, i))}
          {shown.map((c) => <polyline key={c.id} points={line(c.path)} className={`plan ${c.id === decided ? "plan-decided" : c.nominal ? "plan-nominal" : ""} ${c.id === overlays.picked ? "plan-picked" : ""}`} onClick={() => overlays.onPick?.(c.id)} />)}
          {overlays.extraPath && <polyline points={line(overlays.extraPath)} className="plan plan-user" />}
          {overlays.tracks && record.tracks.filter((t) => t.confirmed).map((t) => {
            const yaw = Math.hypot(t.vx, t.vy) > 1 ? Math.atan2(t.vy, t.vx) : 0;
            return (
              <g key={t.id}>
                {overlays.trails && t.trail.length > 1 && <polyline points={line(t.trail)} className="trail" />}
                {box(t.x, t.y, yaw, t.length, t.width, `track ${AGENT[t.cls] ?? ""} ${t.id === overlays.selectedTrack ? "track-selected" : ""}`, `t${t.id}`, () => overlays.onTrack?.(t.id))}
                {Math.hypot(t.vx, t.vy) > 1 && <line className="velocity" x1={sx(t.y)} y1={sy(t.x)} x2={sx(t.y + t.vy)} y2={sy(t.x + t.vx)} />}
                {(t.id === overlays.selectedTrack || Math.hypot(t.vx, t.vy) > 1) && <text className="track-id" x={sx(t.y) + t.width / 2 + 0.5} y={sy(t.x) + 0.5}>{t.id}</text>}
              </g>
            );
          })}
          {overlays.forecast !== undefined && overlays.forecast !== null && record.forecast_paths.map((p) => {
            const upto = p.points.filter((q) => q.t <= overlays.forecast! + 1e-6);
            const end = upto.at(-1);
            if (!end) return null;
            return <g key={p.id}>
              <polyline className="future" points={line([[record.tracks.find((t) => t.id === p.id)?.x ?? upto[0].x, record.tracks.find((t) => t.id === p.id)?.y ?? upto[0].y], ...upto.map((q) => [q.x, q.y] as [number, number])])} />
              <circle className="spread" cx={sx(end.y)} cy={sy(end.x)} r={Math.max(end.std_m, 0.3)} />
            </g>;
          })}
          {box(0, 0, 0, 4.6, 2, "ego", "ego")}
        </svg>
      </div>
    </figure>
  );
}

export const StateKey = () => (
  <div className="state-key"><span><i className="k-free" />Observed free</span><span><i className="k-occ" />Occupied</span><span><i className="k-unk" />Unknown</span><span><i className="k-out" />No coverage</span></div>
);
