import inspect

import numpy as np
import pytest

from foresightbev.contracts import (
    BEVGrid, CellState, Detection, EgoState, OccupancyFrame, Provenance, SemanticClass as C, VoxelGrid,
)
from foresightbev.forecast.forecaster import ConstantVelocityForecaster, ForecastConfig, persistence_forecast
from foresightbev.perception.detection import detect
from foresightbev.tracking.tracker import Tracker, TrackerConfig

GRID = BEVGrid(-20, 60, -15, 15, 0.25)
VGRID = VoxelGrid(GRID)


def det(t, x, y, cls=C.VEHICLE):
    return Detection(t, cls, np.array([x, y], float), 4.6, 1.9, 1.5, 1.0, 20)


def empty_occ(t=0.0):
    state = np.full(GRID.shape, int(CellState.FREE), np.uint8)
    return OccupancyFrame(t, VGRID, np.zeros(VGRID.shape, np.uint8), np.zeros(VGRID.shape, np.uint8), state,
                          np.zeros(GRID.shape, np.uint8), np.zeros(GRID.shape, np.float32),
                          np.ones(GRID.shape, np.uint8), {}, "test", Provenance.SIMULATED)


def run_track(n=15, dt=0.2, v=5.0, y=1.0):
    tracker = Tracker()
    out = []
    for i in range(n):
        t = i * dt
        out = tracker.update([det(t, 10 + v * t, y)], EgoState(t, 0, 0, 0))
    return tracker, out


def test_track_confirmed_and_velocity_converges():
    _, tracks = run_track()
    assert len(tracks) == 1 and tracks[0].confirmed and tracks[0].track_id == 1
    assert tracks[0].velocity[0] == pytest.approx(5.0, abs=0.3)


def test_covariance_shrinks_with_updates():
    tracker = Tracker()
    tracker.update([det(0, 10, 0)], EgoState(0, 0, 0, 0))
    first = tracker.tracks[0].P[2, 2]
    tracker.update([det(.2, 11, 0)], EgoState(.2, 0, 0, 0))
    tracker.update([det(.4, 12, 0)], EgoState(.4, 0, 0, 0))
    assert tracker.tracks[0].P[2, 2] < first


def test_ego_motion_is_compensated():
    """A stationary world object seen from a moving ego has zero world velocity."""
    tracker = Tracker()
    tracks = []
    for i in range(15):
        t = i * .2
        tracks = tracker.update([det(t, 30 - 10 * t, 0)], EgoState(t, 10 * t, 0, 0, 10))
    assert np.hypot(*tracks[0].velocity) < 0.3 and tracks[0].position[0] == pytest.approx(30, abs=.3)


def test_identity_kept_for_crossing_lanes():
    tracker = Tracker()
    tracks = []
    for i in range(20):
        t = i * .2
        tracks = tracker.update([det(t, 10 + 5 * t, 3), det(t, 40 - 5 * t, -3)], EgoState(t, 0, 0, 0))
    assert sorted(x.track_id for x in tracks) == [1, 2]


def test_classes_never_associate():
    tracker = Tracker()
    tracker.update([det(0, 10, 0, C.VEHICLE)], EgoState(0, 0, 0, 0))
    tracker.update([det(.2, 10, 0, C.VEHICLE)], EgoState(.2, 0, 0, 0))
    tracker.update([det(.4, 10, 0, C.VEHICLE)], EgoState(.4, 0, 0, 0))
    tracker.update([det(.6, 10, 0, C.PERSON)], EgoState(.6, 0, 0, 0))
    assert {t.semantic for t in tracker.tracks} == {C.VEHICLE, C.PERSON}


def test_missed_track_coasts_then_dies():
    tracker = Tracker(TrackerConfig(max_misses=2))
    for i in range(3):
        tracker.update([det(i * .2, 10, 0)], EgoState(i * .2, 0, 0, 0))
    tracks = tracker.update([], EgoState(.6, 0, 0, 0))
    assert tracks[0].misses == 1
    tracker.update([], EgoState(.8, 0, 0, 0))
    assert tracker.update([], EgoState(1.0, 0, 0, 0)) == []


def test_person_confirms_faster_than_vehicle():
    tracker = Tracker()
    tracker.update([det(0, 10, 0, C.PERSON), det(0, 30, 0)], EgoState(0, 0, 0, 0))
    tracks = tracker.update([det(.2, 10, 0, C.PERSON), det(.2, 30, 0)], EgoState(.2, 0, 0, 0))
    assert {t.semantic: t.confirmed for t in tracks} == {C.PERSON: True, C.VEHICLE: False}


def test_unconfirmed_clutter_is_dropped_at_first_miss():
    tracker = Tracker()
    tracker.update([det(0, 10, 0)], EgoState(0, 0, 0, 0))
    assert tracker.update([], EgoState(.2, 0, 0, 0)) == []


def test_non_increasing_timestamp_rejected():
    tracker = Tracker()
    tracker.update([], EgoState(1, 0, 0, 0))
    with pytest.raises(ValueError):
        tracker.update([], EgoState(1, 0, 0, 0))


def test_constant_velocity_forecast_matches_analytic_position():
    _, tracks = run_track(n=25, v=4.0, y=2.0)
    t = 24 * .2
    ego = EgoState(t, 0, 0, 0)
    f = ConstantVelocityForecaster(GRID, ForecastConfig(n_samples=64, accel_std=0.0)).forecast(empty_occ(t), tracks, ego)
    X, Y = GRID.cell_centers()
    for k, h in enumerate(f.horizons):
        p = f.occupancy_prob[k]
        cx = (X * p).sum() / p.sum()
        assert cx == pytest.approx(10 + 4 * (t + h), abs=0.6)
        assert (Y * p).sum() / p.sum() == pytest.approx(2.0, abs=0.3)


def test_flow_is_forward_flow_in_ego_frame():
    _, tracks = run_track(n=25, v=4.0)
    t = 24 * .2
    # Ego yawed 90 deg left: world +x motion appears as -y in the ego frame.
    ego = EgoState(t, 10 + 4 * t, 0, np.pi / 2)
    f = ConstantVelocityForecaster(GRID, ForecastConfig(n_samples=32, accel_std=0)).forecast(empty_occ(t), tracks, ego)
    p = f.occupancy_prob[0] > 0.5
    assert f.flow[0][p][:, 1].mean() == pytest.approx(-4.0, abs=.4)
    assert abs(f.flow[0][p][:, 0].mean()) < .4


def test_forecast_rejects_future_inputs():
    _, tracks = run_track()
    t = 14 * .2
    forecaster = ConstantVelocityForecaster(GRID)
    with pytest.raises(ValueError):
        forecaster.forecast(empty_occ(t + 1), tracks, EgoState(t, 0, 0, 0))
    with pytest.raises(ValueError):
        forecaster.forecast(empty_occ(t - 5), tracks, EgoState(t - 5, 0, 0, 0))


def test_forecast_api_accepts_no_future_argument():
    params = inspect.signature(ConstantVelocityForecaster.forecast).parameters
    assert list(params) == ["self", "occupancy", "tracks", "ego"]


def test_range_noise_grows_with_square_of_distance():
    from foresightbev.perception.detection import DetectionConfig, detection_covariance
    cfg = DetectionConfig(range_std_per_m2=0.01)
    near = detection_covariance(np.array([20.0, 0.0]), cfg)
    far = detection_covariance(np.array([40.0, 0.0]), cfg)
    assert np.sqrt(far[0, 0]) / np.sqrt(near[0, 0]) == pytest.approx(4.0)
    assert far[1, 1] == pytest.approx(near[1, 1])


def test_stationary_track_does_not_get_phantom_motion_in_forecast():
    tracker = Tracker()
    rng = np.random.default_rng(0)
    tracks = []
    for i in range(20):
        t = i * .2
        tracks = tracker.update([det(t, 15 + rng.normal(0, .2), rng.normal(0, .2))], EgoState(t, 0, 0, 0))
    t = 19 * .2
    f = ConstantVelocityForecaster(GRID).forecast(empty_occ(t), tracks, EgoState(t, 0, 0, 0))
    X, _ = GRID.cell_centers()
    p = f.occupancy_prob[-1]
    assert (X * p).sum() / p.sum() == pytest.approx(15, abs=.6)


def test_detection_fragments_of_one_car_merge():
    occ = empty_occ(0)
    for y in (-.9, .9):
        ix, iy = GRID.to_index(np.array([20.0, 20.3, 20.6]), np.full(3, y))
        occ.bev_class[ix, iy] = C.VEHICLE
    assert len(detect(occ)) == 1


def test_stale_track_is_extrapolated_by_its_lag():
    _, tracks = run_track(n=25, v=4.0, y=0.0)
    t = 24 * .2
    late = t + 1.0
    f = ConstantVelocityForecaster(GRID, ForecastConfig(n_samples=32, accel_std=0)).forecast(
        empty_occ(t), tracks, EgoState(late, 0, 0, 0))
    X, _ = GRID.cell_centers()
    p = f.occupancy_prob[0]
    assert (X * p).sum() / p.sum() == pytest.approx(10 + 4 * (late + .25), abs=.8)


def test_spread_grows_with_horizon():
    _, tracks = run_track(n=25, v=4.0)
    t = 24 * .2
    f = ConstantVelocityForecaster(GRID).forecast(empty_occ(t), tracks, EgoState(t, 0, 0, 0))
    assert (f.occupancy_prob[-1] > 0).sum() > (f.occupancy_prob[0] > 0).sum()


def test_unknown_is_preserved_into_forecast_and_static_persists():
    occ = empty_occ(0)
    occ.bev_state[100:110, 50:60] = CellState.UNKNOWN
    occ.bev_state[150, 60] = CellState.OCCUPIED
    f = ConstantVelocityForecaster(GRID).forecast(occ, [], EgoState(0, 0, 0, 0))
    assert f.unobserved[105, 55] and f.static_occupied[150, 60]
    assert (f.occupancy_prob[:, 150, 60] == 1).all()
    assert persistence_forecast(occ, f.horizons).shape[0] == len(f.horizons)


def test_detection_from_contact_cells_pushes_centre_away_from_camera():
    occ = empty_occ(0)
    ix, iy = GRID.to_index(20.1, np.array([-.8, -.5, -.25, 0, .25, .5, .8]))
    occ.bev_class[ix, iy] = C.VEHICLE
    occ.bev_height[ix, iy] = 1.5
    dets = detect(occ)
    assert len(dets) == 1 and dets[0].center_ego[0] == pytest.approx(22.3, abs=.4)
    assert dets[0].height == 1.5
