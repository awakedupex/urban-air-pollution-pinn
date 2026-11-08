"""Guards against train/test leakage."""
from __future__ import annotations


def assert_no_station_leakage(train, test):
    """Held-out stations must never appear in the training table."""
    shared = set(train["location_id"]) & set(test["location_id"])
    assert not shared, f"stations in both train and test: {sorted(shared)}"
