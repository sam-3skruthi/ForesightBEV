import numpy as np
import pytest

from foresightbev.evaluation import metrics as M


def test_brier_and_ece_on_known_values():
    p, y = np.array([.8, .8, .2, .2]), np.array([1., 1., 0., 0.])
    assert M.brier(p, y) == pytest.approx(.04)
    assert M.ece(p, y) == pytest.approx(.2)


def test_perfectly_calibrated_bin_has_zero_ece():
    p = np.full(10, .3)
    y = np.array([1] * 3 + [0] * 7, float)
    assert M.ece(p, y) == pytest.approx(0)


def test_auroc_handles_ties_and_single_class():
    assert M.auroc(np.array([.1, .9]), np.array([0., 1.])) == 1
    assert M.auroc(np.array([.5, .5]), np.array([0., 1.])) == .5
    assert M.auroc(np.array([.5, .5]), np.array([0., 0.])) is None


def test_bootstrap_interval_contains_mean():
    r = M.bootstrap_mean(np.array([0, 0, 1, 1, 0, 1.]), np.random.default_rng(0))
    assert r["low"] <= r["mean"] <= r["high"] and r["n"] == 6
    assert M.bootstrap_mean(np.array([]), np.random.default_rng(0))["mean"] is None


def test_iou():
    a = np.array([True, True, False])
    assert M.iou(a, np.array([False, True, True])) == pytest.approx(1 / 3)
    assert M.iou(np.zeros(2, bool), np.zeros(2, bool)) is None
