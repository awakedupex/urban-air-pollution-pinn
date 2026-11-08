"""Offline tests for download helpers (no network)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.download import openaq_s3_key, parse_openmeteo, select_locations


def test_parse_openmeteo_units_and_uv():
    payload = {
        "hourly_units": {"wind_speed_10m": "m/s", "wind_direction_10m": "°", "boundary_layer_height": "m"},
        "hourly": {
            "time": ["2024-11-10T00:00", "2024-11-10T01:00"],
            "wind_speed_10m": [2.0, 3.0],
            "wind_direction_10m": [270.0, 0.0],
            "boundary_layer_height": [150.0, 900.0],
            "temperature_2m": [15.0, 16.0],
            "relative_humidity_2m": [80.0, 70.0],
            "surface_pressure": [990.0, 991.0],
        },
    }
    df = parse_openmeteo(payload)
    assert str(df["time"].dt.tz) == "UTC"
    assert df["u"].tolist() == pytest.approx([2.0, 0.0], abs=1e-12)
    assert df["v"].tolist() == pytest.approx([0.0, -3.0], abs=1e-12)
    assert df["blh"].tolist() == [150.0, 900.0]


def test_parse_openmeteo_rejects_kmh():
    payload = {
        "hourly_units": {"wind_speed_10m": "km/h", "wind_direction_10m": "°", "boundary_layer_height": "m"},
        "hourly": {"time": ["2024-11-10T00:00"], "wind_speed_10m": [2.0], "wind_direction_10m": [0.0],
                   "boundary_layer_height": [100.0]},
    }
    with pytest.raises(ValueError, match="m/s"):
        parse_openmeteo(payload)


def test_select_locations_keeps_active_regulatory_monitors():
    locs = pd.DataFrame({
        "location_id": [1, 2, 3, 4, 5],
        "is_monitor": [True, True, False, True, True],
        "first": ["2018-01-01T00:00:00Z", "2018-01-01T00:00:00Z", "2018-01-01T00:00:00Z",
                  "2025-02-18T00:00:00Z", None],
        "last": ["2026-11-05T00:00:00Z", "2022-10-31T00:00:00Z", "2026-11-05T00:00:00Z",
                 "2026-11-05T00:00:00Z", None],
    })
    start, end = pd.Timestamp("2024-11-10", tz="UTC"), pd.Timestamp("2024-11-20", tz="UTC")
    # 1 active monitor; 2 retired; 3 low-cost sensor; 4 started later; 5 no dates.
    assert select_locations(locs, start, end)["location_id"].tolist() == [1]


def test_openaq_s3_key():
    k = openaq_s3_key(8118, dt.date(2024, 11, 1))
    assert k == "records/csv.gz/locationid=8118/year=2024/month=11/location-8118-20241101.csv.gz"
