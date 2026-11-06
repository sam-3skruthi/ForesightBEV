import numpy as np
import pytest

from foresightbev.contracts import BEVGrid, CellState, EgoState, OccupancyFrame, Provenance, SemanticClass as C, VoxelGrid
from foresightbev.forecast.learned import HISTORY, HistoryBuffer, coarse_grid, downsample, per_horizon, warp_to

GRID = BEVGrid(-20, 60, -16, 16, 0.25)


def occ(t, ix=None):
    v = VoxelGrid(GRID)
    state = np.full(GRID.shape, int(CellState.FREE), np.uint8)
    classes = np.zeros(GRID.shape, np.uint8)
    if ix is not None:
        state[ix:ix + 8, 60:68] = CellState.OCCUPIED
        classes[ix:ix + 8, 60:68] = C.VEHICLE
    return OccupancyFrame(t, v, np.zeros(v.shape, np.uint8), np.zeros(v.shape, np.uint8), state, classes,
                          np.zeros(GRID.shape, np.float32), np.ones(GRID.shape, np.uint8), {}, "t", Provenance.SIMULATED)


def test_downsample_is_area_coverage():
    m = np.zeros((8, 8), bool)
    m[:4, :2] = True
    assert downsample(m)[0, 0] == pytest.approx(.5)


def test_ego_motion_warp_moves_static_object_backwards():
    coarse = coarse_grid(GRID)
    c = np.zeros((1, *coarse.shape), np.float32)
    ix, iy = coarse.to_index(30.0, 0.0)
    c[0, ix, iy] = 1
    out = warp_to(c, EgoState(0, 0, 0, 0), EgoState(1, 10, 0, 0), coarse)
    assert np.unravel_index(out[0].argmax(), out[0].shape) == (ix - 10, iy)


def test_history_is_causal_and_ordered():
    buffer = HistoryBuffer(GRID)
    x = buffer.push(occ(0, 200), EgoState(0, 0, 0, 0))
    assert x.shape[0] == 16 and (x[:4] == x[12:]).all()
    with pytest.raises(ValueError):
        buffer.push(occ(0), EgoState(0, 0, 0, 0))


def test_newest_frame_is_last_channel_group():
    buffer = HistoryBuffer(GRID)
    for i in range(HISTORY - 1):
        buffer.push(occ(i), EgoState(i, 0, 0, 0))
    x = buffer.push(occ(9, 200), EgoState(9, 0, 0, 0))
    assert x[12].max() > 0 and x[0].max() == 0


def test_per_horizon_metrics():
    y = np.array([[[[True, False]]]])
    assert per_horizon(np.array([[[[.9, .1]]]]), y)[0] == [1.0]


def test_model_output_shape():
    torch = pytest.importorskip("torch")
    from foresightbev.forecast.learned import build_model
    out = build_model()(torch.zeros(2, 16, 80, 44))
    assert out.shape == (2, 6, 80, 44)


def test_models_are_compared_on_a_shared_region():
    from foresightbev.forecast.learned import score_models
    truth = np.zeros((1, 1, 4, 4), bool)
    truth[0, 0, 0, 0] = True
    sparse, wide = np.zeros((1, 1, 4, 4)), np.full((1, 1, 4, 4), .02)
    sparse[0, 0, 0, 0] = 1.0
    scores = score_models(dict(sparse=sparse, wide=wide), truth)
    assert scores["sparse"]["brier"][0] == 0 and scores["wide"]["brier"][0] > 0
