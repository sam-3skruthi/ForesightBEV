# ForesightBEV

ForesightBEV is a research project exploring camera-based 3D scene understanding
and uncertainty-aware trajectory risk assessment for autonomous driving.

## Research direction

The project investigates a pipeline for:

- Constructing 3D bird's-eye-view occupancy representations from vehicle-camera
  observations and calibration data.
- Distinguishing occupied, observed-free, and unknown space.
- Estimating perception uncertainty and studying its calibration.
- Modelling dynamic road users and forecasting future scene occupancy.
- Evaluating candidate trajectories against predicted occupancy and uncertainty.
- Testing safety-oriented decision mechanisms through reproducible experiments.

## Research status

This repository is under active development. Its existing experiments include
a source-level audit and geometric reproduction of selected Cam2BEV operations.
These are research and verification artefacts, not a completed autonomous
driving system.

No claims of real-time performance, autonomous driving capability, or
generalisation to unseen driving conditions are made.

## Repository structure

- `docs/` — architecture and research design documents.
- `experiments/` — experiment outputs, reports, and reproduction logs.
- `research/` — literature, dataset assessments, and repository audits.
- `src/foresightbev/` — project Python package.
- `pyproject.toml` — Python project metadata and dependencies.

## Development environment

The main project environment is managed with
[uv](https://docs.astral.sh/uv/). Research reproductions may use separate,
isolated environments.

## Research principles

- Establish reproducible baselines before proposing new methods.
- Keep measured results separate from hypotheses and illustrative outputs.
- Record dataset provenance, licences, calibration assumptions, and evaluation
  protocols.
- Preserve unknown space rather than treating missing observations as free.
