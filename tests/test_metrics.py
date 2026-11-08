"""Tests for evaluation metrics and baselines."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.metrics import evaluate, mae, r2, rmse
from src.evaluation.validation import assert_no_station_leakage
from src.models.baselines import idw_predict, random_forest_predict


def test_metrics_known_values():
    y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 2.0, 6.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(6 / 4))
    assert r2(y, y) == pytest.approx(1.0)


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.normal(100, 30, 500), rng.normal(100, 30, 500)
    assert mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))
    assert r2(y, p) == pytest.approx(r2_score(y, p))


def test_evaluate_ignores_nan_and_reports_n():
    m = evaluate(np.array([1.0, np.nan, 3.0]), np.array([1.0, 5.0, 4.0]))
    assert m["n"] == 2 and m["mae"] == pytest.approx(0.5) and m["bias"] == pytest.approx(0.5)


def test_leakage_detected():
    tr = pd.DataFrame({"location_id": [1, 2]})
    te = pd.DataFrame({"location_id": [3, 2]})
    with pytest.raises(AssertionError):
        assert_no_station_leakage(tr, te)
    assert_no_station_leakage(tr, pd.DataFrame({"location_id": [3]}))


def _obs():
    t = pd.date_range("2024-11-10", periods=3, freq="h", tz="UTC")
    rows = []
    for i, (x, y) in enumerate([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]):
        for k, tt in enumerate(t):
            rows.append(dict(location_id=i, time=tt, x=x, y=y, pm25=100.0 + 10 * i + k))
    return pd.DataFrame(rows)


def test_idw_exact_at_station_and_bounded():
    train = _obs()
    target = train[train.location_id == 1][["location_id", "time", "x", "y"]]
    np.testing.assert_allclose(idw_predict(train, target), train[train.location_id == 1].pm25)
    mid = target.assign(x=4.0, y=3.0)
    p = idw_predict(train, mid)
    for k, tt in enumerate(mid.time):
        vals = train[train.time == tt].pm25
        assert vals.min() <= p[k] <= vals.max()


def test_random_forest_shape_and_reproducible():
    train = _obs()
    feats = ["x", "y"]
    a = random_forest_predict(train, train, feats, seed=0)
    b = random_forest_predict(train, train, feats, seed=0)
    assert a.shape == (len(train),)
    np.testing.assert_array_equal(a, b)
