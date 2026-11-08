"""Cleaning, projection and splitting of station observations.

Conventions
-----------
* Time is UTC, floored to the hour (column ``time``).
* Concentration column is ``pm25`` in µg/m³ (mass concentration).
* Space is a local tangent plane in km: x east, y north, origin (lat0, lon0).
* Wind (u, v) in m/s is the velocity the air moves TOWARDS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
ACCEPTED_UNITS = {"µg/m³", "μg/m³", "ug/m3", "µg/m3"}


def clean_observations(
    df: pd.DataFrame,
    valid_min: float,
    valid_max: float,
    flatline_hours: int,
    parameter: str = "pm25",
) -> pd.DataFrame:
    """Turn raw OpenAQ long-format records into clean hourly station series.

    Steps: select parameter -> check units -> drop invalid values -> UTC hourly
    timestamps -> collapse duplicates (mean) -> drop flat-lined runs -> sort.
    Returns columns: location_id, location, lat, lon, time, pm25.
    """
    df = df[df["parameter"] == parameter].copy()
    bad_units = set(df["units"].unique()) - ACCEPTED_UNITS
    if bad_units:
        raise ValueError(f"Unexpected units for {parameter}: {sorted(bad_units)}")

    v = pd.to_numeric(df["value"], errors="coerce")
    df = df[v.notna() & (v >= valid_min) & (v <= valid_max)].copy()
    df["value"] = v[df.index]

    df["time"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("h")
    df = (
        df.groupby(["location_id", "time"], as_index=False)
        .agg(location=("location", "first"), lat=("lat", "first"), lon=("lon", "first"), pm25=("value", "mean"))
        .sort_values(["location_id", "time"])
    )

    # Flat-line detection: runs of identical consecutive values per station.
    run_id = (df["pm25"].diff() != 0) | (df["location_id"].diff() != 0)
    run_id = run_id.cumsum()
    run_len = df.groupby(run_id)["pm25"].transform("size")
    df = df[run_len < flatline_hours]

    cols = ["location_id", "location", "lat", "lon", "time", "pm25"]
    return df[cols].reset_index(drop=True)


def validate_coordinates(df, lat_min, lat_max, lon_min, lon_max) -> pd.DataFrame:
    """Keep rows with finite coordinates inside the study bounding box."""
    ok = df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)
    return df[ok.fillna(False)].reset_index(drop=True)


def project_to_local(lat, lon, lat0, lon0):
    """Equirectangular projection to km about (lat0, lon0).

    Distortion over a 50 km domain is < 0.1 %, far below other model errors.
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = EARTH_RADIUS_KM * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon - lon0)
    y = EARTH_RADIUS_KM * np.deg2rad(lat - lat0)
    return x, y


def wind_to_uv(speed, direction_deg):
    """Meteorological wind (speed, direction blowing FROM, degrees) -> (u, v)."""
    rad = np.deg2rad(np.asarray(direction_deg, float))
    speed = np.asarray(speed, float)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def filter_coverage(df, start, end, min_coverage) -> pd.DataFrame:
    """Drop stations with fewer than ``min_coverage`` of hours in [start, end) observed."""
    df = df[(df["time"] >= start) & (df["time"] < end)]
    n_hours = (end - start) / pd.Timedelta(hours=1)
    frac = df.groupby("location_id")["time"].nunique() / n_hours
    keep = frac[frac >= min_coverage].index
    return df[df["location_id"].isin(keep)].reset_index(drop=True)


def _colocated_groups(xy: np.ndarray, min_sep_km: float) -> np.ndarray:
    """Single-linkage groups: stations closer than min_sep_km share a group label."""
    n = len(xy)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    for i, j in zip(*np.where(np.triu(d < min_sep_km, k=1))):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def split_stations(stations: pd.DataFrame, train, val, test, seed, min_sep_km=1.0) -> dict:
    """Split STATIONS (not observations) into train/val/test sets of location_id.

    Co-located stations (closer than ``min_sep_km``) are kept in one subset so a
    test station never has a near-duplicate in training (spatial leakage).
    """
    assert abs(train + val + test - 1.0) < 1e-9
    ids = stations["location_id"].to_numpy()
    groups = _colocated_groups(stations[["x", "y"]].to_numpy(float), min_sep_km)
    uniq = np.unique(groups)
    np.random.default_rng(seed).shuffle(uniq)

    n = len(ids)
    targets = {"test": max(1, round(test * n)), "val": max(1, round(val * n))}
    out = {"train": set(), "val": set(), "test": set()}
    for g in uniq:
        members = {int(i) for i in ids[groups == g]}
        for part in ("test", "val"):
            if len(out[part]) < targets[part]:
                out[part] |= members
                break
        else:
            out["train"] |= members
    return out
