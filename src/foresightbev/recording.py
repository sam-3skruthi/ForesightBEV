"""Write pipeline outputs to a run directory. The dashboard reads only these files."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from foresightbev import __version__
from foresightbev.contracts import CellState
from foresightbev.data.taxonomy import PALETTE
from foresightbev.forecast.forecaster import is_stationary, track_to_ego
from foresightbev.pipeline import FrameResult

STATE_COLORS = {CellState.INVALID: (30, 38, 48), CellState.FREE: (31, 98, 104), CellState.OCCUPIED: (235, 92, 72),
                CellState.UNKNOWN: (88, 98, 111), CellState.OUTSIDE_FOV: (14, 20, 28)}

DETECTOR_ROWS = dict(
    bev_clusters=dict(kind="heuristic", detail="Clustering of dynamic cells plus class-size box completion. Not a trained detector."),
    image_size=dict(kind="heuristic", detail="Image blobs ranged by apparent size with class-height priors. Not a trained detector."),
    streampetr=dict(kind="pretrained", detail="StreamPETR R50 428q, nuScenes and nuImages pretraining, PyTorch inference port. Not retrained. Weights licence unverified, treated as non-commercial."),
)


def provenance_table(input_kind: str, input_detail: str, segmenter: str | None = None, closed_loop: str | None = None, detector: str = "bev_clusters") -> list[dict]:
    """What actually ran. Built per run, so a record can never claim the wrong thing."""
    rows = [dict(stage=1, component="Input", kind=input_kind, detail=input_detail)]
    if segmenter:
        rows.append(dict(stage=1, component="Segmentation", kind="pretrained", detail=f"{segmenter}, Cityscapes weights, not retrained. Cross-dataset on this input."))
    else:
        rows.append(dict(stage=1, component="Segmentation", kind="none", detail="Input is already semantic labels."))
    rows += [
        dict(stage=2, component="BEV projection", kind="classical", detail="Analytical flat-ground IPM."),
        dict(stage=3, component="3D occupancy", kind="classical", detail="Semantic space carving with free, occupied, unknown and outside-FOV states. No trained network."),
        dict(stage=3, component="Camera reliability", kind="heuristic", detail="Usable-label fraction per camera. Not an OOD score or epistemic uncertainty."),
        dict(stage=4, component="Detection", **DETECTOR_ROWS[detector]),
        dict(stage=4, component="Tracking", kind="classical", detail="Constant-velocity Kalman filter with Hungarian association."),
        dict(stage=5, component="Forecast", kind="classical", detail="Monte Carlo constant-velocity rollout of tracks. Not a learned 4D occupancy model."),
        dict(stage=6, component="Candidates", kind="classical", detail="Fixed lattice with IDM nominal plan."),
        dict(stage=6, component="Event score", kind="classical", detail="Forecast sample frequency. Not a collision probability unless a calibrator is applied."),
        dict(stage=6, component="Safety filter", kind="heuristic", detail="Deterministic rules with reason codes. No formal guarantee."),
    ]
    if closed_loop:
        rows.append(dict(stage=7, component="Execution", kind="simulated", detail=closed_loop))
    return rows


PROVENANCE = provenance_table("simulated", "Semantic images rendered by the built-in simulator.", closed_loop="Built-in kinematic simulator. CARLA and Autoware were not executed.")


def source_hash() -> str:
    """SHA-256 of all package source, which identifies code that was never committed."""
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment() -> dict:
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except OSError:
            return ""
    return dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__, platform=platform.platform(),
                machine=platform.machine(), foresightbev=__version__, git_commit=git("rev-parse", "HEAD"),
                git_dirty=bool(git("status", "--porcelain")), source_sha256=source_hash(), created_unix=time.time())


def state_image(state: np.ndarray) -> np.ndarray:
    """RGB image with +x up and +y left, so the map reads like a view from above."""
    rgb = np.zeros((*state.shape, 3), np.uint8)
    for value, color in STATE_COLORS.items():
        rgb[state == value] = color
    return rgb[::-1, ::-1]


def label_mosaic(result: FrameResult) -> np.ndarray:
    tiles = []
    for name, frame in result.bundle.frames.items():
        cam = next(c for c in result.bundle.calibration.cameras if c.name == name)
        if frame.labels is None:
            tile = np.full((cam.height, cam.width, 3), 30, np.uint8)
            cv2.putText(tile, f"{name}: DROPPED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 80, 70), 2)
        else:
            tile = PALETTE[frame.labels].copy()
            cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    return np.vstack(rows)


def forecast_mosaic(result: FrameResult, indices=(3, 7, 11)) -> np.ndarray:
    """Forecast occupancy at selected horizons: unobserved grey, then probability in red."""
    forecast = result.forecast
    tiles = []
    for k in indices:
        if k >= len(forecast.horizons):
            continue
        base = np.full((*forecast.grid.shape, 3), 34, np.uint8)
        base[forecast.unobserved] = (90, 92, 104)
        p = forecast.occupancy_prob[k][..., None]
        base = (base * (1 - p) + np.array([235, 70, 60]) * p).astype(np.uint8)
        tile = base[::-1, ::-1].copy()
        cv2.putText(tile, f"+{forecast.horizons[k]:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        tiles.append(np.full((tile.shape[0], 3, 3), 12, np.uint8))
    return np.hstack(tiles[:-1])


def _round(values, digits=3):
    return np.round(np.asarray(values, float), digits).tolist()


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def forecast_paths(result: FrameResult) -> list[dict]:
    """Mean future positions of confirmed tracks, in the ego frame at this frame's time.

    ``std_m`` is the Kalman position spread propagated by constant velocity. It describes the
    model, is uncalibrated, and is not a confidence interval.
    """
    ego = result.bundle.ego
    forecast = result.forecast
    config = result.forecast_config
    out = []
    for track in result.tracks:
        if not track.confirmed:
            continue
        position, velocity, covariance = track_to_ego(track, ego)
        stationary = config is not None and is_stationary(velocity, covariance, config)
        v = np.zeros(2) if stationary else velocity
        spread_p = float(np.trace(covariance[:2, :2]) / 2)
        spread_v = 0.0 if stationary else float(np.trace(covariance[2:, 2:]) / 2)
        points = []
        for h in forecast.horizons[1::2]:
            p = position + v * float(h)
            points.append(dict(t=float(h), x=float(p[0]), y=float(p[1]), std_m=float(np.sqrt(spread_p + spread_v * float(h) ** 2))))
        out.append(dict(id=track.track_id, cls=track.semantic.name, stationary=bool(stationary), points=points))
    return out


def frame_record(result: FrameResult, ground_truth: list[dict] | None, diagnostics: dict | None = None) -> dict:
    ego = result.bundle.ego
    occupancy = result.occupancy
    tracks = []
    for t in result.tracks:
        position, velocity = ego.world_to_ego(t.position), ego.rotate_to_ego(t.velocity)
        trail = [ego.world_to_ego(np.array(h[1:])).round(3).tolist() for h in t.history]
        tracks.append(dict(id=t.track_id, cls=t.semantic.name, x=float(position[0]), y=float(position[1]), vx=float(velocity[0]),
                           vy=float(velocity[1]), length=t.length, width=t.width, height=t.height, confirmed=t.confirmed,
                           hits=t.hits, misses=t.misses, age=t.age, position_std=float(np.sqrt(np.trace(t.covariance[:2, :2]) / 2)),
                           velocity_std=float(np.sqrt(np.trace(t.covariance[2:, 2:]) / 2)), trail=trail))
    risks = {r.candidate_id: r for r in result.risks}
    candidates = []
    for c in result.candidates:
        r = risks.get(c.candidate_id)
        candidates.append(dict(id=c.candidate_id, nominal=c.is_nominal, fallback=c.is_fallback, path=_round(c.poses[::10, :2], 2),
                               end_speed=float(c.poses[-1, 3]), event_score=r.event_score, calibrated_probability=r.calibrated_probability,
                               calibration_status=r.calibration_status, per_horizon=_round(r.per_horizon_score), static_collision=r.static_collision,
                               first_contact_time=r.first_contact_time, unknown_exposure=r.unknown_exposure,
                               distance_to_unknown=_finite(r.distance_to_unknown), stopping_distance=r.stopping_distance,
                               unknown_violation=r.unknown_corridor_violation, min_clearance_m=_finite(r.min_clearance_m), time_to_conflict_s=r.time_to_conflict_s,
                               closest_track_id=r.closest_track_id))
    d = result.decision
    frames = result.bundle.frames
    reference = result.bundle.timestamp
    return dict(
        frame=result.bundle.frame_index, t=reference,
        ego=dict(x=ego.x, y=ego.y, yaw=ego.yaw, speed=ego.speed, yaw_rate=ego.yaw_rate),
        cameras={name: dict(status=f.status.value, reliability=occupancy.reliability.get(name, 0.0), timestamp=f.timestamp, skew_ms=1000 * (f.timestamp - reference)) for name, f in frames.items()},
        state_fractions={s.name: float((occupancy.bev_state == s).mean()) for s in CellState},
        voxel_state_counts={s.name: int((occupancy.voxel_state == s).sum()) for s in CellState},
        ground_plane=dict(a=occupancy.ground_plane[0], b=occupancy.ground_plane[1], c=occupancy.ground_plane[2]) if occupancy.ground_plane else None,
        model_occupied_cells=int(occupancy.model_occupied.sum()) if occupancy.model_occupied is not None else 0,
        coverage=dict(single=float((occupancy.support_count == 1).mean()), overlap=float((occupancy.support_count >= 2).mean()), none=float((occupancy.support_count == 0).mean())),
        detections=[dict(cls=x.semantic.name, model_class=x.model_class, x=float(x.center_ego[0]), y=float(x.center_ego[1]), length=x.length, width=x.width, height=x.height, yaw=x.yaw,
                         vx=None if x.velocity_ego is None else float(x.velocity_ego[0]), vy=None if x.velocity_ego is None else float(x.velocity_ego[1]), score=x.support, cells=x.n_cells,
                         provenance=x.provenance.value) for x in result.detections],
        tracks=tracks, candidates=candidates, forecast_paths=forecast_paths(result),
        decision=dict(status=d.status.value, selected=d.selected_id, nominal=d.nominal_id, reasons=[r.value for r in d.reasons],
                      fallback=d.fallback_action, constraints=d.active_constraints),
        forecast=dict(method=result.forecast.method, uncertainty=result.forecast.uncertainty, n_samples=result.forecast.n_samples,
                      horizons=result.forecast.horizons.tolist(), t_ref=result.forecast.t_ref, last_input=result.forecast.last_input_timestamp, provenance=result.forecast.provenance.value),
        timings_ms={k: round(v, 2) for k, v in result.timings_ms.items()},
        diagnostics=diagnostics,
        ground_truth_for_display_only=ground_truth,
    )


def save_arrays(result: FrameResult, directory: Path, name: str) -> None:
    """Stage outputs for retrieval and re-evaluation. Bit-packed where that is lossless."""
    o, f = result.occupancy, result.forecast
    extra = {} if o.model_occupied is None else dict(model_occupied=o.model_occupied)
    np.savez_compressed(directory / "arrays" / f"{name}.npz", **extra, bev_state=o.bev_state, bev_class=o.bev_class, bev_height=o.bev_height.astype(np.float16),
                        support_count=o.support_count, voxel_state=o.voxel_state, voxel_class=o.voxel_class, ipm_labels=result.bev.labels)
    samples = np.stack(f.agent_samples)
    np.savez_compressed(directory / "forecast_arrays" / f"{name}.npz", horizons=f.horizons, occupancy_prob=np.round(f.occupancy_prob * 255).astype(np.uint8),
                        flow=f.flow.astype(np.float16), static_occupied=f.static_occupied, unobserved=f.unobserved,
                        agent_samples=np.packbits(samples, axis=1), n_samples=f.n_samples, t_ref=f.t_ref, last_input_timestamp=f.last_input_timestamp)


class RunRecorder:
    def __init__(self, directory: Path, metadata: dict, grid, *, voxel_grid=None, store_arrays: bool = True):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        for sub in ("frames", "bev", "cameras", "forecast"):
            (directory / sub).mkdir(exist_ok=True)
        self.store_arrays = store_arrays
        if store_arrays:
            (directory / "arrays").mkdir(exist_ok=True)
            (directory / "forecast_arrays").mkdir(exist_ok=True)
        self.metadata = dict(metadata, environment=environment(), provenance=metadata.get("provenance", PROVENANCE),
                             grid=dict(x_min=grid.x_min, x_max=grid.x_max, y_min=grid.y_min, y_max=grid.y_max,
                                       resolution=grid.resolution, nx=grid.nx, ny=grid.ny,
                                       image_axes="row 0 is x_max, column 0 is y_max"))
        self.count = 0

    def add(self, result: FrameResult, ground_truth: list[dict] | None = None, diagnostics: dict | None = None) -> None:
        name = f"{result.bundle.frame_index:05d}"
        (self.directory / "frames" / f"{name}.json").write_text(json.dumps(frame_record(result, ground_truth, diagnostics)))
        cv2.imwrite(str(self.directory / "bev" / f"{name}.png"), state_image(result.occupancy.bev_state)[..., ::-1])
        cv2.imwrite(str(self.directory / "cameras" / f"{name}.png"), label_mosaic(result)[..., ::-1])
        cv2.imwrite(str(self.directory / "forecast" / f"{name}.png"), forecast_mosaic(result)[..., ::-1])
        if self.store_arrays:
            save_arrays(result, self.directory, name)
        self.count += 1

    def add_layer(self, layer: str, name: str, rgb: np.ndarray) -> None:
        """Extra preview image, RGB, in BEV display orientation."""
        folder = self.directory / layer
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / f"{name}.png"), rgb[..., ::-1])

    def finish(self, summary: dict) -> None:
        (self.directory / "manifest.json").write_text(json.dumps(dict(self.metadata, frames=self.count, summary=summary), indent=1, default=str))
