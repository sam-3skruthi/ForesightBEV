"""Stage 4 tracking: constant-velocity Kalman filter with Hungarian association.

State is [x, y, vx, vy] in the world frame. Covariance is the filter's own model
covariance. Its consistency is measured (NEES) in Stage 9, not assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from foresightbev.contracts import Detection, EgoState, SemanticClass, TrackState

logger = logging.getLogger(__name__)
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
GATE_CHI2_2DOF_99 = 9.21


@dataclass
class TrackerConfig:
    accel_noise: float = 1.2
    accel_noise_person: float = 3.0
    measurement_std_vehicle: float = 0.7
    measurement_std_person: float = 0.35
    initial_velocity_std: float = 8.0
    initial_velocity_std_person: float = 3.0
    confirm_hits: int = 3
    confirm_hits_person: int = 2
    extent_smoothing: float = 0.2
    use_model_velocity: bool = False
    velocity_std: float = 0.7
    birth_score: float = 0.0
    second_stage: bool = False
    max_misses: int = 5
    history: int = 25


def transition(dt: float, accel_noise: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = accel_noise ** 2
    Q = q * np.array([[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2],
                      [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]])
    return F, Q


class Track:
    def __init__(self, track_id: int, detection: Detection, position: np.ndarray, config: TrackerConfig, noise: np.ndarray):
        self.id = track_id
        self.config = config
        self.semantic = detection.semantic
        self.x = np.array([position[0], position[1], 0.0, 0.0])
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = noise
        person = detection.semantic == SemanticClass.PERSON
        self.confirm_hits = config.confirm_hits_person if person else config.confirm_hits
        velocity_std = config.initial_velocity_std_person if person else config.initial_velocity_std
        self.P[2:, 2:] = np.eye(2) * velocity_std**2
        self.length, self.width, self.height = detection.length, detection.width, detection.height
        self.timestamp = detection.timestamp
        self.age, self.hits, self.misses = 1, 1, 0
        self.history = [(detection.timestamp, float(position[0]), float(position[1]))]

    def predict(self, timestamp: float) -> None:
        dt = timestamp - self.timestamp
        noise = self.config.accel_noise_person if self.semantic == SemanticClass.PERSON else self.config.accel_noise
        F, Q = transition(dt, noise)
        self.x, self.P = F @ self.x, F @ self.P @ F.T + Q
        self.timestamp = timestamp
        self.age += 1

    def mahalanobis(self, z: np.ndarray, R: np.ndarray) -> float:
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        return float(innovation @ np.linalg.solve(S, innovation))

    def update(self, z: np.ndarray, R: np.ndarray, detection: Detection, velocity: np.ndarray | None = None) -> None:
        """Position update, or position and velocity when a model velocity is supplied."""
        if velocity is None:
            G, y, N = H, z, R
        else:
            std = self.config.velocity_std
            G, y = np.eye(4), np.concatenate([z, velocity])
            N = np.zeros((4, 4))
            N[:2, :2] = R
            N[2:, 2:] = np.eye(2) * std**2
        S = G @ self.P @ G.T + N
        K = self.P @ G.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - G @ self.x)
        I_KH = np.eye(4) - K @ G
        self.P = I_KH @ self.P @ I_KH.T + K @ N @ K.T
        self.hits += 1
        self.misses = 0
        rate = self.config.extent_smoothing
        self.length += rate * (detection.length - self.length)
        self.width += rate * (detection.width - self.width)
        self.height += rate * (detection.height - self.height)
        self.history.append((self.timestamp, float(self.x[0]), float(self.x[1])))
        del self.history[:-self.config.history]

    def state(self) -> TrackState:
        return TrackState(self.id, self.timestamp, self.semantic, self.x[:2].copy(), self.x[2:].copy(), self.P.copy(),
                          self.length, self.width, self.height, self.age, self.hits, self.misses,
                          self.hits >= self.confirm_hits, list(self.history))


class Tracker:
    """Causal multi-object tracker. Non-increasing timestamps are rejected."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_timestamp: float | None = None

    def _noise(self, detection: Detection, ego: EgoState) -> np.ndarray:
        """World-frame measurement covariance, from the detection when it has one."""
        if detection.covariance is not None:
            c, s = np.cos(ego.yaw), np.sin(ego.yaw)
            R = np.array([[c, -s], [s, c]])
            return R @ detection.covariance @ R.T
        cfg = self.config
        std = cfg.measurement_std_person if detection.semantic == SemanticClass.PERSON else cfg.measurement_std_vehicle
        return np.eye(2) * std**2

    def update(self, detections: list[Detection], ego: EgoState) -> list[TrackState]:
        timestamp = ego.timestamp
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError(f"non-increasing timestamp {timestamp} after {self.last_timestamp}")
        self.last_timestamp = timestamp
        for track in self.tracks:
            track.predict(timestamp)

        finite = [d for d in detections if np.isfinite(d.center_ego).all()]
        if len(finite) != len(detections):
            logger.warning("dropped %d non-finite detections at t=%.3f", len(detections) - len(finite), timestamp)
        detections = finite
        measurements = [ego.ego_to_world(d.center_ego) for d in detections]
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        def world_velocity(detection: Detection):
            if not self.config.use_model_velocity or detection.velocity_ego is None:
                return None
            c, s = np.cos(ego.yaw), np.sin(ego.yaw)
            vx, vy = detection.velocity_ego
            return np.array([c * vx - s * vy, s * vx + c * vy])

        def associate(track_ids: list[int], detection_ids: list[int]) -> None:
            if not track_ids or not detection_ids:
                return
            cost = np.full((len(track_ids), len(detection_ids)), 1e6)
            for a, i in enumerate(track_ids):
                for b, j in enumerate(detection_ids):
                    if detections[j].semantic == self.tracks[i].semantic:
                        d2 = self.tracks[i].mahalanobis(measurements[j], self._noise(detections[j], ego))
                        if d2 <= GATE_CHI2_2DOF_99:
                            cost[a, b] = d2
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e6:
                    i, j = track_ids[a], detection_ids[b]
                    self.tracks[i].update(measurements[j], self._noise(detections[j], ego), detections[j], world_velocity(detections[j]))
                    matched_tracks.add(i)
                    matched_detections.add(j)

        strong = [j for j, d in enumerate(detections) if d.support >= self.config.birth_score]
        weak = [j for j, d in enumerate(detections) if d.support < self.config.birth_score]
        associate(list(range(len(self.tracks))), strong)
        if self.config.second_stage:
            associate([i for i in range(len(self.tracks)) if i not in matched_tracks], weak)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.config.max_misses
                       and (t.hits >= t.confirm_hits or t.misses == 0)]
        for j, detection in enumerate(detections):
            if j not in matched_detections and detection.support >= self.config.birth_score:
                noise = self._noise(detection, ego)
                track = Track(self.next_id, detection, measurements[j], self.config, noise)
                velocity = world_velocity(detection)
                if velocity is not None:
                    track.x[2:] = velocity
                    track.P[2:, 2:] = np.eye(2) * self.config.velocity_std**2
                self.tracks.append(track)
                self.next_id += 1
        return [t.state() for t in self.tracks]
