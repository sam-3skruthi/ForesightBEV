import { Ban, Check, OctagonAlert, TriangleAlert, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(d));
export const pct = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`);
export const pretty = (s: string) => s.replace(/_/g, " ").replace("lane+0.0 ", "hold lane, ").replace(/lane([+-][\d.]+) /, "shift $1 m, ").replace(/, v([\d.]+)$/, ", $1× speed");

const STATUS: Record<string, { label: string; icon: LucideIcon }> = {
  pass: { label: "Pass", icon: Check }, modify: { label: "Modified", icon: TriangleAlert }, fallback: { label: "Fallback", icon: OctagonAlert }, reject: { label: "Rejected", icon: Ban },
};

export function Status({ value, size }: { value: string; size?: "large" }) {
  const entry = STATUS[value] ?? { label: value, icon: TriangleAlert };
  const Icon = entry.icon;
  return <span className={`status status-${value} ${size ?? ""}`}><Icon size={size ? 22 : 14} strokeWidth={2.2} aria-hidden />{entry.label}</span>;
}

export function Notice({ title, children, tone = "plain" }: { title: string; children?: ReactNode; tone?: "plain" | "error" | "warn" }) {
  return <div className={`notice notice-${tone}`} role={tone === "error" ? "alert" : undefined}><strong>{title}</strong>{children && <p>{children}</p>}</div>;
}

export const Unavailable = ({ what, why }: { what: string; why: string }) => <div className="unavailable"><strong>{what}</strong><span>{why}</span></div>;
export const Loading = ({ what }: { what: string }) => <div className="loading" role="status">Loading {what}</div>;

export function Panel({ title, note, children, wide, actions }: { title: string; note?: ReactNode; children: ReactNode; wide?: boolean; actions?: ReactNode }) {
  return <section className={`panel ${wide ? "panel-wide" : ""}`}><header><h2>{title}</h2>{actions}</header>{children}{note && <p className="note">{note}</p>}</section>;
}

export function Stat({ label, value, unit }: { label: string; value: ReactNode; unit?: string }) {
  return <div className="stat"><span>{label}</span><strong>{value}{unit && <small> {unit}</small>}</strong></div>;
}

export function Tag({ kind, children }: { kind: string; children: ReactNode }) { return <span className={`tag tag-${kind}`}>{children}</span>; }

export function Toggle({ checked, onChange, children, disabled, title }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode; disabled?: boolean; title?: string }) {
  return <label className="toggle" title={title}><input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />{children}</label>;
}

export function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: readonly (readonly [T, string])[]; onChange: (v: T) => void; label: string }) {
  return <div className="segmented" role="radiogroup" aria-label={label}>{options.map(([v, text]) => <button key={v} role="radio" aria-checked={v === value} onClick={() => onChange(v)}>{text}</button>)}</div>;
}

export const REASONS: Record<string, string> = {
  NOMINAL_ADMISSIBLE: "The nominal plan passed every gate.",
  NOMINAL_EVENT_SCORE_EXCEEDED: "Forecast samples crossed the plan more often than the limit allows.",
  NOMINAL_STATIC_COLLISION: "The plan sweeps through cells occupied by something that is not moving.",
  NOMINAL_UNKNOWN_CORRIDOR: "The car could not stop before space an unseen agent might enter from.",
  POST_FALLBACK_HOLD: "A recent fallback blocks accelerating plans for one second.",
  PERCEPTION_MISSING: "No complete perception frame arrived.",
  PERCEPTION_STALE: "The newest perception is too old to act on.",
  CAMERA_COVERAGE_LOST: "One or more cameras are unusable, so coverage is reduced.",
  COVERAGE_SPEED_HOLD: "The plan would speed up with reduced coverage, so it was replaced.",
  NO_ADMISSIBLE_CANDIDATE: "Every alternative plan also failed a gate.",
  FALLBACK_COMFORT_STOP: "Selected a comfortable stop as the fallback.",
  FALLBACK_EMERGENCY_STOP: "Selected an emergency stop as the fallback.",
};
