import numpy as np
import pytest

from foresightbev.contracts import (
    BEVGrid, DecisionStatus as DS, EgoState, ForecastVolume, Provenance, ReasonCode as R, TrackState,
    SemanticClass,
)
from foresightbev.risk.calibration import RiskCalibrator, pool_adjacent_violators
from foresightbev.risk.evaluator import RiskConfig, RiskEvaluator, stopping_distance
from foresightbev.risk.safety_filter import PerceptionHealth, SafetyConfig, SafetyFilter
from foresightbev.risk.trajectories import (
    PlannerConfig, RouteContext, generate_candidates, idm_acceleration, lead_from_tracks,
)

GRID = BEVGrid(-20, 80, -15, 15, 0.25)
HORIZONS = np.arange(1, 13) * 0.25
OK = PerceptionHealth(True, 0.0, 4, 4)
ROUTE = RouteContext(desired_speed=10, lanes=(0.0, 3.5), home_lane=0.0)
EGO = EgoState(0, 0, 0, 0, 10)


def forecast(static=None, agents=None, unobserved=None, n=20):
    K = len(HORIZONS)
    samples = [np.zeros((n, *GRID.shape), bool) for _ in range(K)]
    if agents:
        agents(samples)
    occ = np.stack([s.mean(0) for s in samples]).astype(np.float32)
    st = np.zeros(GRID.shape, bool) if static is None else static
    un = np.zeros(GRID.shape, bool) if unobserved is None else unobserved
    return ForecastVolume(0, 0, HORIZONS, GRID, occ, np.zeros((K, *GRID.shape, 2)), st, un, samples, n, "test", "none",
                          Provenance.SIMULATED)


def box(x0, x1, y0, y1):
    m = np.zeros(GRID.shape, bool)
    X, Y = GRID.cell_centers()
    m[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = True
    return m


def assess(fc, cfg=None, route=ROUTE, ego=EGO, health=OK, cal=None):
    cands = generate_candidates(ego, route, [])
    risks = RiskEvaluator(cfg, cal).evaluate_all(cands, fc, [], ego)
    return cands, {r.candidate_id: r for r in risks}, SafetyFilter().decide(cands, risks, health, route.desired_speed)


def test_clear_road_passes_nominal():
    _, risks, d = assess(forecast())
    assert d.status == DS.PASS and d.reasons == [R.NOMINAL_ADMISSIBLE]
    assert risks["nominal"].event_score == 0 and not risks["nominal"].unknown_corridor_violation


def test_candidates_always_include_nominal_and_both_stops():
    ids = [c.candidate_id for c in generate_candidates(EGO, ROUTE, [])]
    assert ids[0] == "nominal" and {"comfort_stop", "emergency_stop"} <= set(ids)


def test_trajectory_kinematics_are_consistent():
    for c in generate_candidates(EGO, ROUTE, []):
        assert np.allclose(c.times, np.arange(len(c.times)) * 0.05)
        assert (np.hypot(*np.diff(c.poses[:, :2], axis=0).T) <= c.poses[:-1, 3] * .05 + c.poses[1:, 3] * .05).all()
        assert (c.poses[:, 3] >= 0).all()


def test_heading_error_rotates_route_into_ego_frame():
    route = RouteContext(10, 0.0, np.deg2rad(30), (0.0,))
    c = generate_candidates(EGO, route, [])[1]
    end = c.poses[-1]
    assert end[1] / end[0] == pytest.approx(-np.tan(np.deg2rad(30)), abs=.01)


def test_occupied_corridor_is_rejected_and_avoided():
    _, risks, d = assess(forecast(static=box(25, 27, -1.5, 1.5)))
    assert risks["nominal"].static_collision and d.status == DS.MODIFY
    assert R.NOMINAL_STATIC_COLLISION in d.reasons
    assert not risks[d.selected_id].static_collision


def test_swept_path_catches_thin_obstacle_between_horizon_samples():
    """At 20 m/s a 0.25 s horizon step is 5 m; a 0.5 m obstacle must still be found."""
    ego = EgoState(0, 0, 0, 0, 20)
    route = RouteContext(20, lanes=(0.0,))
    for x in np.arange(20, 26, .5):
        _, risks, _ = assess(forecast(static=box(x, x + .5, -.5, .5)), ego=ego, route=route)
        assert risks["nominal"].static_collision


def test_event_is_time_aligned_not_just_spatial():
    """An agent in the path at t=3 s while ego is elsewhere does not count."""
    def late(s):
        s[-1][:, box(5, 8, -1, 1)] = True
    _, risks, _ = assess(forecast(agents=late))
    assert risks["nominal"].event_score == 0


def test_joint_event_counts_samples_not_independent_cells():
    """All 40 cells share one sample, so the score is 0.25, not 1-(0.75)^40."""
    def half(s):
        for k in range(len(s)):
            s[k][:5][:, box(10, 45, -1, 1)] = True
    _, risks, _ = assess(forecast(agents=half))
    assert risks["nominal"].event_score == pytest.approx(0.25)


def test_per_horizon_score_is_monotone():
    def one(s):
        s[6][:10][:, box(15, 22, -1, 1)] = True
    _, risks, _ = assess(forecast(agents=one))
    assert (np.diff(risks["nominal"].per_horizon_score) >= 0).all()
    assert risks["nominal"].first_contact_time == pytest.approx(1.75)


def test_unknown_corridor_is_not_treated_as_free():
    fc = forecast(unobserved=box(20, 80, -15, 15))
    _, risks, d = assess(fc)
    assert risks["nominal"].unknown_corridor_violation and d.status in (DS.MODIFY, DS.FALLBACK)
    assert R.NOMINAL_UNKNOWN_CORRIDOR in d.reasons
    assert d.selected.poses[-1, 3] < 10


def test_unknown_corridor_never_passes_at_speed_even_if_all_else_is_clear():
    fc = forecast(unobserved=box(12, 80, -15, 15))
    _, _, d = assess(fc)
    assert d.status != DS.PASS


def test_unknown_as_free_is_an_explicit_ablation_only():
    fc = forecast(unobserved=box(20, 80, -15, 15))
    _, _, d = assess(fc, cfg=RiskConfig(treat_unknown_as_free=True))
    assert d.status == DS.PASS


def test_slow_candidate_may_approach_unknown_if_it_can_stop():
    ego = EgoState(0, 0, 0, 0, 2.0)
    _, risks, _ = assess(forecast(unobserved=box(30, 80, -15, 15)), ego=ego, route=RouteContext(2.0, lanes=(0.0,)))
    assert not risks["nominal"].unknown_corridor_violation


def test_no_safe_candidate_triggers_fallback():
    fc = forecast(static=box(5, 40, -15, 15))
    _, _, d = assess(fc)
    assert d.status == DS.FALLBACK and d.fallback_action in ("comfort_stop", "emergency_stop")
    assert R.NO_ADMISSIBLE_CANDIDATE in d.reasons


def test_imminent_block_escalates_to_emergency_stop():
    _, _, d = assess(forecast(static=box(14, 16, -15, 15)))
    assert d.selected_id == "emergency_stop" and R.FALLBACK_EMERGENCY_STOP in d.reasons


def test_distant_block_uses_comfort_stop():
    ego = EgoState(0, 0, 0, 0, 4.0)
    route = RouteContext(4.0, lanes=(0.0,))
    _, _, d = assess(forecast(static=box(9, 10, -15, 15)), ego=ego, route=route)
    assert d.status != DS.PASS and d.selected.poses[-1, 3] < 4.0


@pytest.mark.parametrize("health,code", [
    (PerceptionHealth(False, 0, 4, 4), R.PERCEPTION_MISSING),
    (PerceptionHealth(True, 1.0, 4, 4), R.PERCEPTION_STALE),
    (PerceptionHealth(True, 0.0, 0, 4), R.CAMERA_COVERAGE_LOST),
])
def test_bad_perception_is_never_a_clear_road(health, code):
    _, _, d = assess(forecast(), health=health)
    assert d.status == DS.FALLBACK and code in d.reasons and d.selected.is_fallback


def test_partial_camera_loss_is_reported_in_reason_codes():
    _, _, d = assess(forecast(), health=PerceptionHealth(True, 0, 3, 4))
    assert R.CAMERA_COVERAGE_LOST in d.reasons


def test_decision_is_deterministic():
    fc = forecast(static=box(25, 27, -1.5, 1.5))
    a, b = assess(fc)[2], assess(fc)[2]
    assert (a.status, a.selected_id, a.reasons) == (b.status, b.selected_id, b.reasons)


def test_reason_codes_are_machine_readable():
    assert R.NOMINAL_ADMISSIBLE.value == "NOMINAL_ADMISSIBLE"


def test_fallback_requires_stop_candidates():
    with pytest.raises(ValueError):
        SafetyFilter().decide([], [], PerceptionHealth(False, 0, 0, 4), 10)


def test_calibrated_value_overrides_raw_score_when_present():
    def half(s):
        for k in range(len(s)):
            s[k][:5][:, box(10, 45, -1, 1)] = True
    cal = RiskCalibrator(np.array([0, 1.0]), np.array([0, .02]), "test")
    _, risks, d = assess(forecast(agents=half), cal=cal)
    assert risks["nominal"].calibrated_probability == pytest.approx(.005)
    assert risks["nominal"].calibration_status == "test" and d.status == DS.PASS


def test_uncalibrated_output_says_so():
    _, risks, _ = assess(forecast())
    assert risks["nominal"].calibrated_probability is None
    assert risks["nominal"].calibration_status == "uncalibrated_model_frequency"


def test_isotonic_fit_is_monotone_and_averages_violators():
    x, y = pool_adjacent_violators(np.array([.1, .2, .3, .4]), np.array([0, 1, 0, 1.]))
    assert (np.diff(y) >= 0).all() and y.max() == 1 and y.min() == 0
    assert RiskCalibrator.fit(np.array([.1, .2, .3, .4]), np.array([0, 1, 0, 1.]), "t").apply(.25) == pytest.approx(.5)


def test_calibrator_rejects_empty_and_round_trips(tmp_path):
    with pytest.raises(ValueError):
        RiskCalibrator.fit(np.array([]), np.array([]), "x")
    cal = RiskCalibrator.fit(np.array([0, 1.]), np.array([0, 1.]), "s")
    cal.save(tmp_path / "c.json")
    assert RiskCalibrator.load(tmp_path / "c.json").status == "s"


def test_idm_brakes_for_close_lead_and_accelerates_when_free():
    cfg = PlannerConfig()
    assert idm_acceleration(5, 10, None, 0, cfg) > 0
    assert idm_acceleration(10, 10, 8, 0, cfg) < -2


def test_lead_selection_uses_corridor_and_ignores_unconfirmed():
    def t(x, y, confirmed=True):
        return TrackState(1, 0, SemanticClass.VEHICLE, np.array([x, y], float), np.zeros(2), np.eye(4), 4.6, 1.9, 1.5,
                          3, 3, 0, confirmed)
    cfg = PlannerConfig()
    assert lead_from_tracks([t(30, 8)], EGO, cfg)[0] is None
    assert lead_from_tracks([t(20, 0, False)], EGO, cfg)[0] is None
    assert lead_from_tracks([t(40, 0), t(20, 0)], EGO, cfg)[0] == pytest.approx(20 - 2.3 - 2.3)


def test_stopped_ego_does_not_slide_sideways():
    ego = EgoState(0, 0, 0, 0, 0.0)
    stop = [c for c in generate_candidates(ego, ROUTE, []) if c.candidate_id == "comfort_stop"][0]
    change = [c for c in generate_candidates(ego, ROUTE, []) if c.candidate_id == "lane+3.5_v0.0"][0]
    assert np.allclose(stop.poses[:, :2], 0) and np.allclose(change.poses[:, :2], 0)


def test_stopped_ego_holds_without_declaring_an_emergency():
    ego = EgoState(0, 0, 0, 0, 0.0)
    _, _, d = assess(forecast(static=box(3, 8, -15, 15)), ego=ego, route=RouteContext(10, lanes=(0.0,)))
    assert R.FALLBACK_EMERGENCY_STOP not in d.reasons


def _track(x, y, vx=0.0):
    return TrackState(1, 0, SemanticClass.VEHICLE, np.array([x, y], float), np.array([vx, 0.0]), np.eye(4), 4.6, 1.9,
                      1.5, 5, 5, 0, True)


def test_nominal_goes_around_stationary_blocker():
    c = generate_candidates(EGO, ROUTE, [_track(20, 0)])[0]
    assert c.is_nominal and c.terminal_lateral == 3.5


def test_nominal_keeps_lane_behind_moving_lead():
    c = generate_candidates(EGO, ROUTE, [_track(20, 0, 8.0)])[0]
    assert c.terminal_lateral == 0.0


def test_nominal_stays_home_if_every_lane_is_blocked():
    c = generate_candidates(EGO, ROUTE, [_track(20, 0), _track(20, 3.5)])[0]
    assert c.terminal_lateral == 0.0 and c.poses[-1, 3] < 10


def test_contact_at_current_footprint_does_not_veto_every_candidate():
    """Uncertainty about a car beside a stopped ego is not a risk of moving forward."""
    def beside(s):
        for k in range(len(s)):
            s[k][:10][:, box(-3, 2, -2.5, -0.9)] = True
    ego = EgoState(0, 0, 0, 0, 0.0)
    _, risks, d = assess(forecast(agents=beside), ego=ego)
    assert risks["comfort_stop"].event_score == 0 and d.status == DS.PASS


def test_occlusion_beside_the_path_slows_fast_plans_but_not_slow_ones():
    """A shadow outside the lane still matters: someone can walk out of it."""
    shadow = box(15, 80, -6, -2.6)
    _, fast, _ = assess(forecast(unobserved=shadow))
    assert fast["lane+0.0_v1.0"].unknown_corridor_violation
    ego = EgoState(0, 0, 0, 0, 3.0)
    _, slow, _ = assess(forecast(unobserved=shadow), ego=ego, route=RouteContext(3.0, lanes=(0.0,)))
    assert not slow["nominal"].unknown_corridor_violation


def test_no_acceleration_right_after_a_fallback():
    fc_block, fc_clear = forecast(static=box(14, 16, -15, 15)), forecast()
    flt = SafetyFilter()
    cands = generate_candidates(EGO, ROUTE, [])
    risks = RiskEvaluator().evaluate_all(cands, fc_block, [], EGO)
    assert flt.decide(cands, risks, OK, 10).status == DS.FALLBACK
    ego = EgoState(.2, 0, 0, 0, 6.0)
    later = generate_candidates(ego, ROUTE, [])
    decision = flt.decide(later, RiskEvaluator().evaluate_all(later, fc_clear, [], ego), OK, 10)
    assert R.POST_FALLBACK_HOLD in decision.reasons and decision.selected.poses[-1, 3] <= 6.0
    ego = EgoState(2.0, 0, 0, 0, 6.0)
    final = generate_candidates(ego, ROUTE, [])
    assert flt.decide(final, RiskEvaluator().evaluate_all(final, fc_clear, [], ego), OK, 10).status == DS.PASS


def test_replanned_lane_change_keeps_progressing():
    """Regression: each replan must continue the manoeuvre, not restart its slow start."""
    route_y = 0.0
    speed = 10.0
    for i in range(20):
        route = RouteContext(10, route_y, 0.0, (0.0, 3.5), 0.0)
        plan = [c for c in generate_candidates(EgoState(i * .2, 0, route_y, 0, speed), route, []) if c.candidate_id == "lane+3.5_v1.0"][0]
        route_y += float(np.interp(.2, plan.times, plan.poses[:, 1]))
    assert route_y > 3.0


def test_filter_prefers_the_lane_the_nominal_chose():
    stalled = _track(25, 0)
    cands = generate_candidates(EGO, ROUTE, [stalled])
    assert cands[0].terminal_lateral == 3.5
    fc = forecast(unobserved=box(20, 80, -15, 15))
    risks = RiskEvaluator().evaluate_all(cands, fc, [], EGO)
    d = SafetyFilter().decide(cands, risks, OK, 10)
    assert d.status != DS.PASS and d.selected.terminal_lateral == 3.5


def test_blocker_is_found_when_ego_is_yawed():
    """Regression: a car in the lane must stay in the lane while the ego is mid-manoeuvre."""
    yaw = 0.18
    ego = EgoState(0, 0.0, 0.0, yaw, 5.0)
    route = RouteContext(10, 0.0, yaw, (0.0, 3.5), 0.0)
    plan = generate_candidates(ego, route, [_track(20, 0)])[0]
    assert plan.terminal_lateral == 3.5


def test_lead_gap_uses_route_coordinates_when_yawed():
    yaw = 0.18
    ego = EgoState(0, 0.0, 0.0, yaw, 5.0)
    route = RouteContext(10, 0.0, yaw, (0.0,), 0.0)
    gap, _ = lead_from_tracks([_track(20, 0, 8.0)], ego, PlannerConfig(), route)
    assert gap == pytest.approx(20 - 4.6, abs=.01)


def test_slow_ego_can_creep_past_an_occlusion_it_could_stop_for():
    """Regression: the stopping check covers the next replan, not the whole 3 s plan."""
    ego = EgoState(0, 0, 0, 0, 2.0)
    route = RouteContext(2.0, lanes=(0.0,))
    _, risks, d = assess(forecast(unobserved=box(7, 12, -6, -1.5)), ego=ego, route=route)
    assert not risks["nominal"].unknown_corridor_violation and d.status == DS.PASS


def test_stopping_distance_formula():
    assert stopping_distance(10.0, RiskConfig()) == pytest.approx(2 + 12.5)


def test_coverage_loss_can_forbid_speeding_up():
    ego = EgoState(0, 0, 0, 0, 2.0)
    route = RouteContext(desired_speed=10, lanes=(0.0,), home_lane=0.0)
    cands = generate_candidates(ego, route, [])
    risks = RiskEvaluator().evaluate_all(cands, forecast(), [], ego)
    degraded = PerceptionHealth(True, 0.0, 5, 6)
    assert SafetyFilter().decide(cands, risks, degraded, 10).status == DS.PASS
    decision = SafetyFilter(SafetyConfig(hold_speed_on_coverage_loss=True)).decide(cands, risks, degraded, 10)
    assert decision.status == DS.MODIFY and R.COVERAGE_SPEED_HOLD in decision.reasons
    assert decision.selected.poses[-1, 3] <= decision.selected.poses[0, 3] + 0.1
