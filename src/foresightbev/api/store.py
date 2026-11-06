"""Read stored pipeline outputs. Nothing here invents or defaults a value."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import numpy as np

from foresightbev.contracts import BEVGrid, EgoState, ForecastVolume, Provenance, SemanticClass, TrackState
from foresightbev.errors import DatasetError, DatasetNotFound


class RunNotFound(DatasetNotFound):
    code = "run_not_found"


class OutputNotStored(DatasetNotFound):
    code = "output_not_stored"


def encode_array(array: np.ndarray) -> dict:
    data = np.asarray(array)
    return dict(shape=list(data.shape), dtype=str(data.dtype), encoding="zlib+base64", data=base64.b64encode(zlib.compress(data.tobytes(), 6)).decode())


def decode_array(payload: dict) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(payload["data"])), payload["dtype"]).reshape(payload["shape"])


class RunStore:
    def __init__(self, data_dir: Path):
        self.runs = data_dir / "runs"

    def directory(self, run: str) -> Path:
        path = self.runs / run
        if not (path / "manifest.json").is_file():
            raise RunNotFound(f"no completed run named {run}")
        return path

    def manifest(self, run: str) -> dict:
        try:
            return json.loads((self.directory(run) / "manifest.json").read_text())
        except json.JSONDecodeError as error:
            raise DatasetError(f"run {run} has a corrupt manifest: {error}") from error

    def list(self) -> list[dict]:
        return [dict(self.manifest(p.parent.name), id=p.parent.name) for p in sorted(self.runs.glob("*/manifest.json")) if not p.parent.name.startswith(".")]

    def frame(self, run: str, frame: int) -> dict:
        path = self.directory(run) / "frames" / f"{frame:05d}.json"
        if not path.is_file():
            raise OutputNotStored(f"run {run} has no frame {frame}")
        return json.loads(path.read_text())

    def _npz(self, run: str, folder: str, frame: int):
        path = self.directory(run) / folder / f"{frame:05d}.npz"
        if not path.is_file():
            raise OutputNotStored(f"run {run} has no stored {folder} for frame {frame}. Record it again to store arrays.")
        try:
            return np.load(path)
        except (OSError, ValueError) as error:
            raise DatasetError(f"corrupt {path.name}: {error}") from error

    def arrays(self, run: str, frame: int):
        return self._npz(run, "arrays", frame)

    def forecast_arrays(self, run: str, frame: int):
        return self._npz(run, "forecast_arrays", frame)

    def forecast_volume(self, run: str, frame: int) -> ForecastVolume:
        """Rebuild the exact stored forecast, joint samples included."""
        d = self.forecast_arrays(run, frame)
        g = self.manifest(run)["grid"]
        grid = BEVGrid(g["x_min"], g["x_max"], g["y_min"], g["y_max"], g["resolution"])
        record = self.frame(run, frame)
        n = int(d["n_samples"])
        K = len(d["horizons"])
        samples = np.unpackbits(d["agent_samples"], axis=1)[:, :n].astype(bool)
        return ForecastVolume(float(d["t_ref"]), float(d["last_input_timestamp"]), d["horizons"], grid, d["occupancy_prob"].astype(np.float32) / 255.0, d["flow"].astype(np.float32),
                              d["static_occupied"], d["unobserved"], [samples[k] for k in range(K)], n, record["forecast"]["method"], record["forecast"]["uncertainty"], Provenance.CLASSICAL)

    def tracks(self, run: str, frame: int) -> tuple[list[TrackState], EgoState]:
        """Tracks as stored. Only position, velocity and class are needed downstream here."""
        record = self.frame(run, frame)
        e = record["ego"]
        ego = EgoState(record["t"], e["x"], e["y"], e["yaw"], e["speed"])
        tracks = []
        for t in record["tracks"]:
            position = ego.ego_to_world(np.array([t["x"], t["y"]]))
            c, s = np.cos(ego.yaw), np.sin(ego.yaw)
            velocity = np.array([c * t["vx"] - s * t["vy"], s * t["vx"] + c * t["vy"]])
            tracks.append(TrackState(t["id"], record["t"], SemanticClass[t["cls"]], position, velocity, np.eye(4), t["length"], t["width"], t["height"], t["hits"] + t["misses"],
                                     t["hits"], t["misses"], t["confirmed"]))
        return tracks, ego
