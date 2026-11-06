"""Exceptions with a stable machine-readable code, shared by the CLI and the API."""

from __future__ import annotations


class ForesightError(Exception):
    """Base error. ``code`` is stable; the message says what failed and how to recover."""

    code = "foresightbev_error"
    http_status = 500


class ConfigurationError(ForesightError):
    code = "invalid_configuration"
    http_status = 422


class DatasetError(ForesightError):
    code = "invalid_dataset"
    http_status = 422


class DatasetNotFound(DatasetError):
    code = "dataset_not_found"
    http_status = 404


class CalibrationError(ForesightError):
    code = "invalid_calibration"
    http_status = 422


class SynchronizationError(ForesightError):
    code = "synchronization_failed"
    http_status = 422


class ModelUnavailable(ForesightError):
    code = "model_unavailable"
    http_status = 503


class JobConflict(ForesightError):
    code = "job_conflict"
    http_status = 409
