export interface SequenceRun { id: string; mode: string | null; fault: { type: string; camera: string; first_frame: number; last_frame: number } | null; frames: number | null }
export interface Sequence { name: string; description: string; samples: number; split: "mini_val" | "mini_train"; detector_saw_scene_in_training: boolean; status_note: string; night: boolean; runs: SequenceRun[] }
export interface Grid { x_min: number; x_max: number; y_min: number; y_max: number; resolution: number; nx: number; ny: number }
export interface Provenance { stage: number; component: string; kind: string; detail: string }
export interface RunDetail {
  id: string; kind: string | null; title: string | null; scenario: string | null; mode: string | null; frames: number; input: string | null; caveat: string | null; has_arrays: boolean;
  grid: Grid; provenance: Provenance[]; environment: Record<string, unknown>; summary: { metrics?: Metrics; wall_s?: number; decisions?: unknown[] } | null;
}
export interface TimelineRow { frame: number; t: number; speed: number; status: string; selected: string; reasons: string[]; tracks: number; detections: number; healthy_cameras: number }
export interface Candidate {
  id: string; nominal: boolean; fallback: boolean; path: [number, number][]; end_speed: number; event_score: number; calibrated_probability: number | null; static_collision: boolean; first_contact_time: number | null;
  unknown_exposure: number; distance_to_unknown: number | null; stopping_distance: number; unknown_violation: boolean; min_clearance_m: number | null; time_to_conflict_s: number | null; closest_track_id: number | null; per_horizon: number[];
}
export interface Track { id: number; cls: string; x: number; y: number; vx: number; vy: number; length: number; width: number; height: number; confirmed: boolean; hits: number; misses: number; age: number; position_std: number; velocity_std: number; trail: [number, number][] }
export interface Detection { cls: string; model_class: string | null; x: number; y: number; length: number; width: number; height: number; yaw: number | null; vx: number | null; vy: number | null; score: number; provenance: string }
export interface Truth { name: string; cls: string | null; category: string; x: number; y: number; yaw: number; length: number; width: number; visibility: number; scored: boolean }
export interface ForecastPath { id: number; cls: string; stationary: boolean; points: { t: number; x: number; y: number; std_m: number }[] }
export interface FrameRecord {
  frame: number; t: number; ego: { x: number; y: number; yaw: number; speed: number; yaw_rate: number };
  cameras: Record<string, { status: string; reliability: number; timestamp: number; skew_ms: number }>;
  state_fractions: Record<string, number>; ground_plane: { a: number; b: number; c: number } | null; model_occupied_cells: number; coverage: { single: number; overlap: number; none: number };
  detections: Detection[]; tracks: Track[]; candidates: Candidate[]; forecast_paths: ForecastPath[];
  decision: { status: string; selected: string; nominal: string | null; reasons: string[]; fallback: string | null; constraints: Record<string, number> };
  forecast: { method: string; uncertainty: string; n_samples: number; horizons: number[]; t_ref: number; last_input: number; provenance: string };
  timings_ms: Record<string, number>; diagnostics: { box_cells: number; box_cells_free_after: number; box_cells_free_before: number; annotated_scored: number; free_cells_after: number; free_cells_before: number } | null;
  ground_truth_for_display_only: Truth[] | null;
}
export interface CameraView { name: string; status: string; reliability: number; timestamp: number; skew_ms: number; calibration: { name: string; width: number; height: number; fx: number; fy: number; origin_ego: number[]; forward_ego: number[]; horizontal_fov_deg: number } }
export interface RiskRow { candidate_id: string; event_score: number; score_semantics: string; static_collision: boolean; first_contact_time_s: number | null; unknown_exposure: number; distance_to_unknown_m: number | null; stopping_distance_m: number; unknown_corridor_violation: boolean; min_clearance_m: number | null; time_to_conflict_s: number | null; admissible: boolean; problems: string[] }
export interface PlanResult { risks: RiskRow[]; decision: { status: string; selected: string; reasons: string[]; fallback: string | null }; path: [number, number][] | null }
export interface Metrics {
  frames: number; detection: Record<string, { annotated: number; recall_2m: number | null; detections: number; precision_2m: number | null }>; detection_position_error_m: Record<string, { mean: number | null; matches: number }>;
  tracking: { annotated: number; matched: number; recall_2m: number | null; false_track_frames: number; id_switches: number; mean_position_error_m: number | null; mean_velocity_error_mps: number | null; mean_nees_4dof: number | null };
  occupancy_against_dynamic_boxes: { box_cells: number; box_cells_marked_free: number | null; boxes: number; boxes_with_dynamic_occupied_contact: number | null };
  forecast: { horizons_s: number[]; constant_velocity_error_m: (number | null)[]; frozen_error_m: (number | null)[]; pairs: number[] };
  safety: { decisions: Record<string, number>; logged_path_frames: number; logged_path_true_contact: number; logged_path_event_score_alarm: number; logged_path_alarm_rate: number | null; logged_path_unknown_violation_rate: number | null };
  mean_pipeline_ms: number | null;
}
export interface Pooled { pooled: Metrics; per_scene: Record<string, Metrics>; scenes: string[] }
export type Json = Record<string, any>; // eslint-disable-line
