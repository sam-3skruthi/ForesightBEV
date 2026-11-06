
import numpy as np
import pytest

from foresightbev.contracts import EgoState, SemanticClass as C
from foresightbev.data.nuscenes import GroundTruthBox
from foresightbev.evaluation.nuscenes_eval import Scores, eligible, excused, greedy_match, in_range, logged_candidate

EGO = EgoState(0, 100, 200, 0, 5)


def box(x, y, cls=C.VEHICLE, vis=4, name="a"):
    return GroundTruthBox(name, "vehicle.car", cls, 100 + x, 200 + y, 1, 0, 4.6, 1.9, 1.5, vis)


def test_eligible_applies_class_range_and_visibility():
    boxes = [box(10, 0), box(10, 0, vis=1), box(80, 0), box(10, 0, cls=None)]
    assert len(eligible(boxes, EGO)) == 1


def test_greedy_match_is_one_to_one_and_prefers_nearest():
    predicted = [np.array([0.5, 0]), np.array([0.2, 0])]
    assert greedy_match(predicted, [np.array([0.0, 0])], lambda p: p) == [(1, 0, 0.2)]


def test_greedy_match_respects_limit():
    assert greedy_match([np.array([5.0, 0])], [np.array([0.0, 0])], lambda p: p) == []


def test_scores_merge_and_summary_handle_empty():
    a, b = Scores(frames=1), Scores(frames=2)
    a.gt[C.VEHICLE] = 2; a.detected[C.VEHICLE] = 1; a.detections[C.VEHICLE] = 2
    a.merge(b)
    s = a.summary()
    assert s["frames"] == 3 and s["detection"]["VEHICLE"]["recall_2m"] == .5 and s["detection"]["PERSON"]["recall_2m"] is None


def test_logged_candidate_uses_future_only_offline_and_is_in_ego_frame():
    path = [(i * .5, 100 + 5 * i * .5, 200.0, 0.0) for i in range(10)]
    c = logged_candidate(path, 0, EGO)
    assert c.candidate_id == "logged_human_path" and c.poses[-1, 0] == pytest.approx(15) and c.poses[20, 3] == pytest.approx(5)


def test_logged_candidate_absent_near_the_end_of_a_scene():
    assert logged_candidate([(0, 0, 0, 0), (1, 1, 0, 0)], 0, EgoState(0, 0, 0, 0)) is None


def test_range_limit_drops_only_far_predictions():
    assert len(in_range([np.array([30.0, 0]), np.array([50.0, 0])], lambda p: p)) == 1


def test_only_unmatched_predictions_on_hidden_objects_are_excused():
    hidden, visible = box(10, 0, vis=1), box(30, 0)
    position = lambda p: p
    assert excused([np.array([10.5, 0.0])], position, [hidden, visible], EGO) == 1
    assert excused([np.array([20.0, 0.0])], position, [hidden, visible], EGO) == 0


def test_excusing_cannot_lower_recall():
    """Regression: a correct detection beside a hidden object must still match."""
    target = [np.array([11.0, 0.0])]
    predicted = [np.array([11.2, 0.0])]
    assert len(greedy_match(predicted, target, lambda p: p)) == 1
