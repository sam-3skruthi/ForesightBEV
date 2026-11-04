# ForesightBEV — Research Status

## Current research direction

ForesightBEV is focused on camera-based autonomous-driving scene
understanding, 3D occupancy estimation, perception uncertainty, future
occupancy prediction, and trajectory-risk assessment.

The current research direction is automotive and multi-camera oriented.
The architecture, datasets, baselines, coordinate conventions, and evaluation
protocols must be established specifically for this scope.

## Historical work

Earlier project work explored RGB-D geometry and robot-centric bird's-eye-view
mapping. The following documents are retained as historical research records:

- `docs/architecture/COORDINATE_SYSTEMS_AND_CALIBRATION.md`
- `docs/architecture/RGBD_GEOMETRY_DESIGN.md`
- `research/experiments/STAGE_1_RGBD_GEOMETRY_PLAN.md`
- `research/experiments/STAGE_1_DESIGN_REVIEW_ADDENDUM.md`
- `research/datasets/RGBD_DATASET_ASSESSMENT.md`

These documents are not the current automotive system specification. Their
coordinate conventions, sensor assumptions, and design decisions must not be
carried forward without review.

## Retained repository audits

The Cam2BEV and Birds_Eye_View audits are retained as prior-art and geometric
reproduction research. They do not establish the final ForesightBEV model
architecture or research contribution.

The upstream repository clones are kept locally under
`research/repositories/` and excluded from the main Git repository.

## Research integrity

- Distinguish reproduced results from published claims and proposed work.
- Preserve dataset provenance and licence restrictions.
- Do not claim model capabilities before implementation and evaluation.
- Establish automotive datasets, baselines, coordinate conventions, and
  metrics before implementing the main perception pipeline.
