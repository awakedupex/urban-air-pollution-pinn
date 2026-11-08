"""Tests for the data pipeline (src/data).

Requirements under test:
  * invalid concentrations (sentinels, negatives, out-of-range, NaN) are removed;
  * units other than µg/m³ are rejected;
  * duplicate (station, hour) records are collapsed to one;
  * output is sorted by station then time, timestamps are UTC hourly;
  * stuck sensors (flat-lines) are removed;
  * coordinates outside the study box are rejected;
  * the projection to local km is metrically correct;
  * wind (speed, meteorological direction) -> (u, v) follows the "blowing-from" convention;
  * the station split is disjoint, exhaustive, reproducible and keeps co-located
    stations in the same subset (no spatial leakage).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_observations,
    filter_coverage,
    project_to_local,
    split_stations,
    validate_coordinates,
    wind_to_uv,
)

BBOX = dict(lat_min=28.40, lat_max=28.88, lon_min=76.84, lon_max=77.35)


def _raw(values, times=None, loc=1, lat=28.6, lon=77.2, units="µg/m³", parameter="pm25"):
    n = len(values)
    if times is None:
        times = pd.date_range("2024-11-10T00:00", periods=n, freq="h", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "location_id": loc,
            "location": f"S{loc}",
            "datetime": [t.isoformat() for t in times],
            "lat": lat,
            "lon": lon,
            "parameter": parameter,
            "units": units,
            "value": values,
        }
    )


def test_invalid_concentrations_removed():
    df = _raw([50.0, -999.0, -3.0, np.nan, 5000.0, 80.0])
    out = clean_observations(df, valid_min=0, valid_max=2000, flatline_hours=6)
    assert out["pm25"].tolist() == [50.0, 80.0]


def test_other_parameters_ignored():
    df = pd.concat([_raw([10.0, 20.0]), _raw([999.0, 999.0], parameter="pm10")])
    out = clean_observations(df, valid_min=0, valid_max=2000, flatline_hours=6)
    assert out["pm25"].tolist() == [10.0, 20.0]


def test_wrong_units_rejected():
    df = _raw([10.0, 20.0], units="ppm")
    with pytest.raises(ValueError, match="units"):
        clean_observations(df, valid_min=0, valid_max=2000, flatline_hours=6)


def test_duplicate_timestamps_collapsed():
    t = pd.Timestamp("2024-11-10T05:00", tz="Asia/Kolkata")
    df = _raw([10.0, 30.0, 40.0], times=[t, t, t + pd.Timedelta(hours=1)])
    out = clean_observations(df, valid_min=0, valid_max=2000, flatline_hours=6)
    assert len(out) == 2
    assert not out.duplicated(["location_id", "time"]).any()
    assert out["pm25"].iloc[0] == pytest.approx(20.0)  # mean of duplicates


def test_sorted_utc_hourly():
    times = pd.to_datetime(
        ["2024-11-10T03:00+05:30", "2024-11-10T01:00+05:30", "2024-11-10T02:00+05:30"]
    )
    df = _raw([3.0, 1.0, 2.0], times=list(times))
    out = clean_observations(df, valid_min=0, valid_max=2000, flatline_hours=6)
    assert out["time"].is_monotonic_increasing
    assert str(out["time"].dt.tz) == "UTC"
    assert out["time"].iloc[0] == pd.Timestamp("2024-11-09T19:30", tz="UTC").floor("h")
    assert out["pm25"].tolist() == [1.0, 2.0, 3.0]


def test_flatline_removed():
    vals = [10.0, 11.0] + [55.0] * 6 + [12.0]
    out = clean_observations(_raw(vals), valid_min=0, valid_max=2000, flatline_hours=6)
    assert 55.0 not in out["pm25"].tolist()
    assert out["pm25"].tolist() == [10.0, 11.0, 12.0]


def test_short_repeat_kept():
    vals = [10.0] + [55.0] * 5 + [12.0]
    out = clean_observations(_raw(vals), valid_min=0, valid_max=2000, flatline_hours=6)
    assert out["pm25"].tolist().count(55.0) == 5


def test_coordinates_outside_box_rejected():
    df = pd.DataFrame(
        {"location_id": [1, 2, 3, 4], "lat": [28.6, 30.0, np.nan, 28.5], "lon": [77.2, 77.2, 77.2, 80.0]}
    )
    out = validate_coordinates(df, **BBOX)
    assert out["location_id"].tolist() == [1]


def test_projection_metric():
    # 1 degree of latitude ~ 111.19 km on a sphere of radius 6371 km
    x, y = project_to_local(np.array([29.63]), np.array([77.15]), 28.63, 77.15)
    assert x[0] == pytest.approx(0.0, abs=1e-9)
    assert y[0] == pytest.approx(111.195, rel=1e-4)
    # 1 degree of longitude shrinks by cos(lat0)
    x, y = project_to_local(np.array([28.63]), np.array([78.15]), 28.63, 77.15)
    assert x[0] == pytest.approx(111.195 * np.cos(np.deg2rad(28.63)), rel=1e-4)


def test_wind_convention():
    # Wind FROM the north (0 deg) blows towards the south: u = 0, v < 0.
    u, v = wind_to_uv(np.array([10.0]), np.array([0.0]))
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert v[0] == pytest.approx(-10.0)
    # Wind FROM the west (270 deg) blows towards the east: u > 0.
    u, v = wind_to_uv(np.array([4.0]), np.array([270.0]))
    assert u[0] == pytest.approx(4.0)
    assert v[0] == pytest.approx(0.0, abs=1e-12)


def test_filter_coverage():
    t = pd.date_range("2024-11-10", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"location_id": [1] * 10 + [2] * 3, "time": list(t) + list(t[:3]), "pm25": 1.0}
    )
    out = filter_coverage(df, t[0], t[-1] + pd.Timedelta(hours=1), min_coverage=0.75)
    assert set(out["location_id"]) == {1}


def _stations(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"location_id": np.arange(n), "x": rng.uniform(-20, 20, n), "y": rng.uniform(-20, 20, n)})


def test_split_disjoint_exhaustive_reproducible():
    st = _stations()
    s1 = split_stations(st, train=0.7, val=0.15, test=0.15, seed=42)
    s2 = split_stations(st, train=0.7, val=0.15, test=0.15, seed=42)
    assert s1 == s2
    tr, va, te = s1["train"], s1["val"], s1["test"]
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(st["location_id"])
    assert len(va) >= 1 and len(te) >= 1


def test_split_keeps_colocated_stations_together():
    st = _stations()
    # Station 100 sits 300 m from station 0: they must land in the same subset.
    extra = pd.DataFrame({"location_id": [100], "x": [st.x[0] + 0.3], "y": [st.y[0]]})
    st = pd.concat([st, extra], ignore_index=True)
    for seed in range(10):
        s = split_stations(st, train=0.7, val=0.15, test=0.15, seed=seed, min_sep_km=1.0)
        for part in s.values():
            assert (0 in part) == (100 in part)
