"""Replaceable stage interfaces. Implementations declare what they are via provenance."""

from __future__ import annotations

from typing import Protocol

from foresightbev.contracts import Detection, EgoState, ForecastVolume, FrameBundle, OccupancyFrame, TrackState


class Detector(Protocol):
    def __call__(self, occupancy: OccupancyFrame, bundle: FrameBundle) -> list[Detection]: ...


class Forecaster(Protocol):
    method: str

    def forecast(self, occupancy: OccupancyFrame, tracks: list[TrackState], ego: EgoState) -> ForecastVolume:
        """Must use nothing later than ``ego.timestamp``, and must supply joint samples."""
        ...
