import { useState } from "react";
import { fmt } from "./ui";

export interface Series { name: string; tone: 1 | 2 | 3; dash?: boolean; values: (number | null)[] }
const M = { l: 46, r: 14, t: 10, b: 36 };

function Legend({ series }: { series: { name: string; tone: number; dash?: boolean }[] }) {
  return <div className="legend">{series.map((s) => <span key={s.name}><svg width="20" height="8" aria-hidden><line x1="0" x2="20" y1="4" y2="4" className={`s${s.tone} ${s.dash ? "dash" : ""}`} /></svg>{s.name}</span>)}</div>;
}

export function Lines({ x, series, xLabel, yLabel, yMax, label, digits = 2 }: { x: (number | string)[]; series: Series[]; xLabel: string; yLabel: string; yMax?: number; label: string; digits?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const top = yMax ?? Math.max(...series.flatMap((s) => s.values.map((v) => v ?? 0)), 0.01) * 1.1;
  const W = 520, H = 240, pw = W - M.l - M.r, ph = H - M.t - M.b;
  const sx = (i: number) => M.l + (i / Math.max(x.length - 1, 1)) * pw;
  const sy = (v: number) => M.t + ph - (Math.min(v, top) / top) * ph;
  return (
    <figure className="chart">
      <Legend series={series} />
      <div className="chart-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onMouseLeave={() => setHover(null)}
          onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setHover(Math.max(0, Math.min(x.length - 1, Math.round((((e.clientX - r.left) / r.width) * W - M.l) / pw * (x.length - 1))))); }}>
          {[0, .5, 1].map((f) => <g key={f}><line x1={M.l} x2={W - M.r} y1={sy(f * top)} y2={sy(f * top)} className="grid" /><text x={M.l - 7} y={sy(f * top) + 4} textAnchor="end">{fmt(f * top, top < 2 ? 2 : 1)}</text></g>)}
          {x.map((v, i) => <text key={i} x={sx(i)} y={H - 18} textAnchor="middle">{v}</text>)}
          <text x={M.l + pw / 2} y={H - 2} textAnchor="middle">{xLabel}</text>
          <text transform={`translate(11 ${M.t + ph / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
          {series.map((s) => <polyline key={s.name} className={`s${s.tone} ${s.dash ? "dash" : ""}`} fill="none" points={s.values.map((v, i) => (v === null ? "" : `${sx(i)},${sy(v)}`)).join(" ")} />)}
          {hover !== null && <line x1={sx(hover)} x2={sx(hover)} y1={M.t} y2={M.t + ph} className="cross" />}
          {hover !== null && series.map((s) => s.values[hover] !== null && <circle key={s.name} cx={sx(hover)} cy={sy(s.values[hover]!)} r={4} className={`dot d${s.tone}`} />)}
        </svg>
        {hover !== null && <div className="tip" style={{ left: `${(sx(hover) / W) * 100}%` }}><strong>{xLabel}: {x[hover]}</strong>{series.map((s) => <span key={s.name}>{s.name}: {fmt(s.values[hover], digits)}</span>)}</div>}
      </div>
    </figure>
  );
}

/** Horizontal paired bars for comparing two systems on a shared 0 to 1 scale. */
export function Compare({ rows, a, b, max = 1, digits = 2, unit = "" }: { rows: { label: string; a: number | null; b: number | null }[]; a: string; b: string; max?: number; digits?: number; unit?: string }) {
  return (
    <div className="compare">
      <div className="legend"><span><i className="swatch t2" />{a}</span><span><i className="swatch t1" />{b}</span></div>
      {rows.map((r) => (
        <div className="compare-row" key={r.label}>
          <span className="compare-label">{r.label}</span>
          <div className="compare-bars">
            <div className="bar t2" style={{ width: `${(Math.min(r.a ?? 0, max) / max) * 100}%` }} /><b>{fmt(r.a, digits)}{unit}</b>
            <div className="bar t1" style={{ width: `${(Math.min(r.b ?? 0, max) / max) * 100}%` }} /><b>{fmt(r.b, digits)}{unit}</b>
          </div>
        </div>
      ))}
    </div>
  );
}

export function Share({ parts }: { parts: { label: string; value: number; tone: string }[] }) {
  return (
    <div className="share">
      <div className="share-bar" role="img" aria-label={parts.map((p) => `${p.label} ${(p.value * 100).toFixed(1)}%`).join(", ")}>{parts.map((p) => <i key={p.label} className={p.tone} style={{ width: `${p.value * 100}%` }} />)}</div>
      <div className="share-key">{parts.map((p) => <span key={p.label}><i className={p.tone} />{p.label} <b>{(p.value * 100).toFixed(1)}%</b></span>)}</div>
    </div>
  );
}
