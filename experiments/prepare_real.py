"""Clean the raw OpenAQ PM2.5 records into per-window station tables (OBSERVED data).

Input:  data/raw/openaq/pm_{window}.csv.gz, data/raw/openmeteo/era5_{window}.csv
Output: data/processed/real/{window}_observations.csv, {window}_forcing.csv, {window}_stations.csv,
        and a cleaning report (records removed at each step) in results/metrics/real_cleaning.json.

Usage: .venv/bin/python -m experiments.prepare_real
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.preprocessing import clean_observations, filter_coverage, project_to_local, validate_coordinates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "real"


def main():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    dom, d = cfg["domain"], cfg["data"]
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, w in cfg["windows"].items():
        raw = pd.read_csv(ROOT / "data" / "raw" / "openaq" / f"pm_{name}.csv.gz")
        start = pd.Timestamp(w["start"], tz=d["timezone"]).tz_convert("UTC")
        end = pd.Timestamp(w["end"], tz=d["timezone"]).tz_convert("UTC")

        n_raw = int((raw["parameter"] == d["pollutant"]).sum())
        obs = clean_observations(raw, d["valid_min"], d["valid_max"], d["flatline_hours"], d["pollutant"])
        n_clean = len(obs)
        obs = validate_coordinates(obs, dom["lat_min"], dom["lat_max"], dom["lon_min"], dom["lon_max"])
        n_box = len(obs)
        obs = filter_coverage(obs, start, end, d["min_coverage"])

        x, y = project_to_local(obs["lat"], obs["lon"], dom["lat0"], dom["lon0"])
        obs = obs.assign(x=x, y=y, t_rel=(obs["time"] - start).dt.total_seconds())
        obs.to_csv(OUT / f"{name}_observations.csv", index=False)
        obs.drop_duplicates("location_id")[["location_id", "location", "lat", "lon", "x", "y"]].to_csv(
            OUT / f"{name}_stations.csv", index=False)

        met = pd.read_csv(ROOT / "data" / "raw" / "openmeteo" / f"era5_{name}.csv", parse_dates=["time"])
        met = met[(met.time >= start) & (met.time <= end)]
        met.assign(t_rel=(met.time - start).dt.total_seconds()).to_csv(OUT / f"{name}_forcing.csv", index=False)

        report[name] = dict(raw_pm25_records=n_raw, after_cleaning=n_clean, inside_box=n_box,
                            after_coverage=len(obs), stations=int(obs["location_id"].nunique()),
                            start_utc=str(start), end_utc=str(end))
        print(name, report[name])
    (ROOT / "results" / "metrics" / "real_cleaning.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
