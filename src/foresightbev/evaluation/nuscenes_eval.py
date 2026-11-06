"""Score a replayed nuScenes sequence against annotations. Offline only.

Annotations are read here, after the pipeline has produced its outputs for a frame.
Nothing computed here is fed back into the pipeline. No occupancy labels exist in
this data, so occupancy is scored against annotated dynamic boxes only and static
scene occupancy is not scored. Ten scenes support verification and short-sequence
checks only. They do not support claims about generalisation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from foresightbev.contracts import CellState, SemanticClass as C, TrajectoryCandidate
from foresightbev.data.nuscenes import GroundTruthBox, NuScenesDataset, NuScenesGroundTruth
from foresightbev.data.sequence import LabelSequence
from foresightbev.forecast.forecaster import is_stationary, rasterize, track_to_ego
from foresightbev.pipeline import FrameResult, PipelineConfig
from foresightbev.replay import run_sequence
from foresightbev.risk.calibration import RiskCalibrator
from foresightbev.risk.evaluator import RiskEvaluator
from foresightbev.sim.world import obb_corners, obb_overlap

MAX_RANGE_M = 40.0
MATCH_M = 2.0
MIN_VISIBILITY = 2
HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
KEYFRAME_S = 0.5
RANGE_BINS = ((0, 15), (15, 30), (30, 40))


def eligible(boxes: list[GroundTruthBox], ego) -> list[tuple[GroundTruthBox, np.ndarray]]:
    """Dynamic boxes in range and at least partly visible, with ego-frame centres."""
    out = []
    for box in boxes:
        if box.semantic is None or box.visibility < MIN_VISIBILITY:
            continue
        center = ego.world_to_ego(np.array([box.x, box.y]))
        if np.hypot(*center) <= MAX_RANGE_M:
            out.append((box, center))
    return out


def in_range(items: list, position) -> list:
    """Predictions inside the scored range. Nothing beyond it is annotated for scoring."""
    return [item for item in items if np.hypot(*position(item)) <= MAX_RANGE_M]


def excused(items: list, position, boxes: list[GroundTruthBox], ego) -> int:
    """How many unmatched predictions lie on annotated low-visibility objects.

    Those are not false alarms. This runs after matching, so it can never remove a
    prediction that matched a visible object. Applied equally to all detectors.
    """
    hidden = [ego.world_to_ego(np.array([b.x, b.y])) for b in boxes if b.semantic is not None and b.visibility < MIN_VISIBILITY]
    return sum(any(np.hypot(*(position(item) - h)) <= MATCH_M for h in hidden) for item in items)


def greedy_match(predicted: list, targets: list[np.ndarray], position, limit: float = MATCH_M) -> list[tuple[int, int, float]]:
    """Globally nearest pairs first, one to one, within ``limit`` metres."""
    pairs = sorted((float(np.hypot(*(position(p) - t))), i, j) for i, p in enumerate(predicted) for j, t in enumerate(targets))
    used_i, used_j, out = set(), set(), []
    for d, i, j in pairs:
        if d <= limit and i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j); out.append((i, j, d))
    return out


@dataclass
class Scores:
    frames: int = 0
    gt: Counter = field(default_factory=Counter)
    detected: Counter = field(default_factory=Counter)
    detections: Counter = field(default_factory=Counter)
    position_errors: dict = field(default_factory=lambda: {b: [] for b in RANGE_BINS})
    tracked: int = 0
    track_gt: int = 0
    false_tracks: int = 0
    id_switches: int = 0
    track_position: list = field(default_factory=list)
    track_velocity: list = field(default_factory=list)
    nees: list = field(default_factory=list)
    box_cells: int = 0
    box_free: int = 0
    box_seen: int = 0
    box_total: int = 0
    dynamic_cells: int = 0
    dynamic_true: int = 0
    forecast_error: dict = field(default_factory=lambda: {h: [] for h in HORIZONS})
    frozen_error: dict = field(default_factory=lambda: {h: [] for h in HORIZONS})
    decisions: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    logged_alarm: int = 0
    logged_unknown: int = 0
    logged_frames: int = 0
    logged_contact: int = 0
    runtime_ms: list = field(default_factory=list)
    assigned: dict = field(default_factory=dict)

    def merge(self, other: "Scores") -> None:
        for name in ("frames", "tracked", "track_gt", "false_tracks", "id_switches", "box_cells", "box_free", "box_seen", "box_total", "dynamic_cells",
                     "dynamic_true", "logged_alarm", "logged_unknown", "logged_frames", "logged_contact"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in ("gt", "detected", "detections", "decisions", "reasons"):
            getattr(self, name).update(getattr(other, name))
        for name in ("track_position", "track_velocity", "nees", "runtime_ms"):
            getattr(self, name).extend(getattr(other, name))
        for b in RANGE_BINS:
            self.position_errors[b].extend(other.position_errors[b])
        for h in HORIZONS:
            self.forecast_error[h].extend(other.forecast_error[h])
            self.frozen_error[h].extend(other.frozen_error[h])

    def summary(self) -> dict:
        def mean(values):
            return float(np.mean(values)) if len(values) else None

        def ratio(a, b):
            return a / b if b else None
        classes = {c.name: dict(annotated=self.gt[c], recall_2m=ratio(self.detected[c], self.gt[c]), detections=self.detections[c],
                                precision_2m=ratio(self.detected[c], self.detections[c])) for c in (C.VEHICLE, C.PERSON)}
        return dict(
            frames=self.frames, detection=classes,
            detection_position_error_m={f"{a}-{b}m": dict(mean=mean(v), matches=len(v)) for (a, b), v in self.position_errors.items()},
            tracking=dict(annotated=self.track_gt, matched=self.tracked, recall_2m=ratio(self.tracked, self.track_gt), false_track_frames=self.false_tracks, id_switches=self.id_switches,
                          mean_position_error_m=mean(self.track_position), mean_velocity_error_mps=mean(self.track_velocity), mean_nees_4dof=mean(self.nees)),
            occupancy_against_dynamic_boxes=dict(box_cells=self.box_cells, box_cells_marked_free=ratio(self.box_free, self.box_cells), boxes=self.box_total,
                                                 boxes_with_dynamic_occupied_contact=ratio(self.box_seen, self.box_total), dynamic_occupied_cells=self.dynamic_cells,
                                                 dynamic_occupied_inside_inflated_box=ratio(self.dynamic_true, self.dynamic_cells)),
            forecast=dict(horizons_s=list(HORIZONS), constant_velocity_error_m=[mean(self.forecast_error[h]) for h in HORIZONS], frozen_error_m=[mean(self.frozen_error[h]) for h in HORIZONS],
                          pairs=[len(self.forecast_error[h]) for h in HORIZONS]),
            safety=dict(decisions=dict(self.decisions), reasons=dict(self.reasons), logged_path_frames=self.logged_frames, logged_path_true_contact=self.logged_contact,
                        logged_path_event_score_alarm=self.logged_alarm, logged_path_alarm_rate=ratio(self.logged_alarm, self.logged_frames),
                        logged_path_unknown_violation_rate=ratio(self.logged_unknown, self.logged_frames)),
            mean_pipeline_ms=mean(self.runtime_ms))


def logged_candidate(path: list[tuple[float, float, float, float]], index: int, ego, horizon: float = 3.0) -> TrajectoryCandidate | None:
    """The human driver's actual future path, for offline scoring as a candidate."""
    times = np.arange(0, horizon + 0.025, 0.05)
    t = np.array([p[0] for p in path]) - ego.timestamp
    if t[-1] < horizon:
        return None
    xy = np.stack([np.interp(times, t, [p[1] for p in path]), np.interp(times, t, [p[2] for p in path])], -1)
    local = ego.world_to_ego(xy)
    yaw = np.interp(times, t, np.unwrap([p[3] for p in path])) - ego.yaw
    speed = np.hypot(*np.gradient(xy, 0.05, axis=0).T)
    return TrajectoryCandidate("logged_human_path", ego.timestamp, times, np.column_stack([local, yaw, speed]))


class SceneScorer:
    def __init__(self, truth: NuScenesGroundTruth, scene: str, sequence: LabelSequence, config: PipelineConfig):
        self.truth, self.scene, self.sequence = truth, scene, sequence
        self.tokens = truth.dataset.sample_tokens(scene)
        self.ego_path = truth.ego_path(scene)
        self.evaluator = RiskEvaluator(config.risk)
        self.threshold = config.safety.max_event_score
        self.forecast_config = config.forecast
        self.scores = Scores()
        self.instances: dict[str, dict[int, GroundTruthBox]] = {}
        for i, token in enumerate(self.tokens):
            for box in truth.boxes(token):
                self.instances.setdefault(box.instance, {})[i] = box

    def add(self, result: FrameResult) -> None:
        record = self.sequence.frames[result.bundle.frame_index]
        index = record["source_index"]
        ego = result.bundle.ego
        s = self.scores
        s.frames += 1
        s.runtime_ms.append(sum(result.timings_ms.values()))
        s.decisions[result.decision.status.value] += 1
        s.reasons.update(r.value for r in result.decision.reasons)
        self.all_boxes = self.truth.boxes(self.tokens[index])
        boxes = eligible(self.all_boxes, ego)
        self._detection(result, boxes)
        self._tracking(result, boxes, index)
        self._occupancy(result, boxes, index)
        self._forecast(result, boxes, index)
        self._logged_path(result, index)

    def _detection(self, result, boxes) -> None:
        s = self.scores
        for cls in (C.VEHICLE, C.PERSON):
            targets = [center for box, center in boxes if box.semantic == cls]
            predicted = in_range([d for d in result.detections if d.semantic == cls], lambda d: d.center_ego)
            s.gt[cls] += len(targets)
            matches = greedy_match(predicted, targets, lambda d: d.center_ego)
            taken = {i for i, _, _ in matches}
            unmatched = [d for i, d in enumerate(predicted) if i not in taken]
            s.detections[cls] += len(predicted) - excused(unmatched, lambda d: d.center_ego, self.all_boxes, result.bundle.ego)
            for _, j, distance in matches:
                s.detected[cls] += 1
                r = float(np.hypot(*targets[j]))
                for low, high in RANGE_BINS:
                    if low <= r < high or (high == RANGE_BINS[-1][1] and r == high):
                        s.position_errors[(low, high)].append(distance)

    def _velocity(self, box: GroundTruthBox, index: int) -> np.ndarray | None:
        """Central difference of annotations. Offline, so future annotations are allowed."""
        history = self.instances[box.instance]
        before, after = history.get(index - 1), history.get(index + 1)
        if before is None or after is None:
            return None
        return np.array([after.x - before.x, after.y - before.y]) / (2 * KEYFRAME_S)

    def _tracking(self, result, boxes, index: int) -> None:
        s = self.scores
        ego = result.bundle.ego
        tracks = [t for t in result.tracks if t.confirmed and t.misses == 0]
        for cls in (C.VEHICLE, C.PERSON):
            subset_boxes = [(b, c) for b, c in boxes if b.semantic == cls]
            subset_tracks = in_range([t for t in tracks if t.semantic == cls], lambda t: ego.world_to_ego(t.position))
            s.track_gt += len(subset_boxes)
            matches = greedy_match(subset_tracks, [c for _, c in subset_boxes], lambda t: ego.world_to_ego(t.position))
            s.tracked += len(matches)
            taken = {i for i, _, _ in matches}
            unmatched = [t for i, t in enumerate(subset_tracks) if i not in taken]
            s.false_tracks += len(unmatched) - excused(unmatched, lambda t: ego.world_to_ego(t.position), self.all_boxes, ego)
            for i, j, distance in matches:
                track, box = subset_tracks[i], subset_boxes[j][0]
                s.track_position.append(distance)
                previous = s.assigned.get(box.instance)
                s.id_switches += int(previous is not None and previous != track.track_id)
                s.assigned[box.instance] = track.track_id
                velocity = self._velocity(box, index)
                if velocity is not None:
                    s.track_velocity.append(float(np.hypot(*(track.velocity - velocity))))
                    error = np.array([box.x, box.y, *velocity]) - np.concatenate([track.position, track.velocity])
                    s.nees.append(float(error @ np.linalg.solve(track.covariance + np.eye(4) * 1e-9, error)))

    def _occupancy(self, result, boxes, index: int) -> None:
        s = self.scores
        ego = result.bundle.ego
        grid = result.occupancy.grid.bev
        state, classes = result.occupancy.bev_state, result.occupancy.bev_class
        dynamic = (state == CellState.OCCUPIED) & np.isin(classes, [C.VEHICLE, C.PERSON])
        inflated = np.zeros(grid.shape, bool)
        for box in self.truth.boxes(self.tokens[index]):
            inflated |= rasterize(ego.world_to_ego(obb_corners(box.x, box.y, box.yaw, box.length + 2.0, box.width + 2.0)), grid)
        for box, _ in boxes:
            mask = rasterize(ego.world_to_ego(obb_corners(box.x, box.y, box.yaw, box.length, box.width)), grid)
            near = rasterize(ego.world_to_ego(obb_corners(box.x, box.y, box.yaw, box.length + 2.0, box.width + 2.0)), grid)
            s.box_cells += int(mask.sum())
            s.box_free += int((mask & (state == CellState.FREE)).sum())
            s.box_total += 1
            s.box_seen += int((near & dynamic).any())
        s.dynamic_cells += int(dynamic.sum())
        s.dynamic_true += int((dynamic & inflated).sum())

    def _forecast(self, result, boxes, index: int) -> None:
        s = self.scores
        ego = result.bundle.ego
        tracks = [t for t in result.tracks if t.confirmed]
        matches = greedy_match(tracks, [c for _, c in boxes], lambda t: ego.world_to_ego(t.position))
        for i, j, _ in matches:
            track, box = tracks[i], boxes[j][0]
            position, velocity, covariance = track_to_ego(track, ego)
            history = self.instances[box.instance]
            for h in HORIZONS:
                future = history.get(index + int(round(h / KEYFRAME_S)))
                if future is None:
                    continue
                truth = ego.world_to_ego(np.array([future.x, future.y]))
                moved = np.zeros(2) if is_stationary(velocity, covariance, self.forecast_config) else velocity * h
                s.forecast_error[h].append(float(np.hypot(*(position + moved - truth))))
                s.frozen_error[h].append(float(np.hypot(*(position - truth))))

    def _logged_path(self, result, index: int) -> None:
        s = self.scores
        ego = result.bundle.ego
        candidate = logged_candidate(self.ego_path, index, ego)
        if candidate is None:
            return
        unobserved = self.evaluator._corridor_unobserved(result.forecast, result.tracks, ego)
        risk = self.evaluator.evaluate(candidate, result.forecast, unobserved)
        s.logged_frames += 1
        s.logged_alarm += int(risk.static_collision or risk.event_score > self.threshold)
        s.logged_unknown += int(risk.unknown_corridor_violation)
        s.logged_contact += int(self._true_contact(candidate, index, ego))

    def _true_contact(self, candidate: TrajectoryCandidate, index: int, ego) -> bool:
        """Did the logged ego footprint really meet an annotated box at a keyframe?"""
        cfg = self.evaluator.config
        for offset in range(1, 7):
            i = index + offset
            if i >= len(self.tokens):
                break
            pose = candidate.poses[min(int(round(offset * KEYFRAME_S / 0.05)), len(candidate.times) - 1)]
            xy = ego.ego_to_world(pose[:2])
            ego_box = obb_corners(xy[0], xy[1], pose[2] + ego.yaw, cfg.ego_length, cfg.ego_width)
            if any(obb_overlap(ego_box, obb_corners(b.x, b.y, b.yaw, b.length, b.width)) for b in self.truth.boxes(self.tokens[i])):
                return True
        return False


def evaluate_scenes(root: Path, version: str, sequences: dict[str, Path], config: PipelineConfig, calibrator: RiskCalibrator | None = None, on_frame=None) -> dict:
    dataset = NuScenesDataset(root, version)
    truth = NuScenesGroundTruth(dataset)
    pooled = Scores()
    per_scene = {}
    decisions = {}
    for scene, directory in sequences.items():
        sequence = LabelSequence(directory)
        scorer = SceneScorer(truth, scene, sequence, config)

        def observe(result):
            scorer.add(result)
            if on_frame:
                on_frame(scene, result)

        decisions[scene] = run_sequence(sequence, config, calibrator, observe)
        per_scene[scene] = scorer.scores.summary()
        pooled.merge(scorer.scores)
    return dict(dataset=f"nuScenes {version}", scenes=list(sequences), split="no split: verification of a pretrained and classical pipeline on annotated logs",
                segmenter=LabelSequence(next(iter(sequences.values()))).doc.get("segmenter"), match_threshold_m=MATCH_M, max_range_m=MAX_RANGE_M, min_visibility_level=MIN_VISIBILITY,
                limitations="Ten scenes. Visibility levels come from annotations. No occupancy labels are available; static occupancy is unscored. Replay is open loop.",
                pooled=pooled.summary(), per_scene=per_scene), decisions
