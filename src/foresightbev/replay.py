"""Run the pipeline over a recorded label sequence, open loop.

The ego follows the log, so decisions here do not affect later frames. This is a
replay, not closed-loop driving, and its decisions are records of what the filter
would have done.
"""

from __future__ import annotations

from typing import Callable

from foresightbev.data.sequence import LabelSequence
from foresightbev.perception.detection_cache import CachedDetector
from foresightbev.pipeline import FrameResult, ForesightPipeline, PipelineConfig
from foresightbev.risk.calibration import RiskCalibrator


def decision_row(result: FrameResult) -> dict:
    d = result.decision
    return dict(frame=result.bundle.frame_index, status=d.status.value, selected=d.selected_id, reasons=[r.value for r in d.reasons])


def build_pipeline(sequence: LabelSequence, config: PipelineConfig, calibrator: RiskCalibrator | None = None) -> ForesightPipeline:
    """Pipeline for a sequence, with cached model detections when configured."""
    floor = config.detector_score_threshold if not config.tracker.second_stage else min(config.detector_score_threshold, 0.1)
    detector = CachedDetector(sequence.directory, floor) if config.detector == "streampetr" else None
    return ForesightPipeline(sequence.calibration, config, calibrator, detector=detector)


def run_sequence(sequence: LabelSequence, config: PipelineConfig, calibrator: RiskCalibrator | None = None,
                 on_frame: Callable[[FrameResult], None] | None = None) -> list[dict]:
    pipeline = build_pipeline(sequence, config, calibrator)
    decisions = []
    for index in range(len(sequence)):
        result = pipeline.step(sequence.bundle(index), sequence.route(index))
        decisions.append(decision_row(result))
        if on_frame is not None:
            on_frame(result)
    return decisions
