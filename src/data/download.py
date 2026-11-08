"""Reproducible download of the raw data.

Sources (all public):
  * PM2.5 — OpenAQ open-data archive on AWS S3 (bucket ``openaq-data-archive``),
    which republishes CPCB/DPCC regulatory monitor data. The station list is
    obtained from the OpenAQ v3 API, which needs a free key in the environment
    variable OPENAQ_API_KEY (or in a ``.env`` file at the project root).
  * Meteorology — Open-Meteo historical archive (ERA5 reanalysis), no key.

Usage:  .venv/bin/python -m src.data.download
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import gzip
import io
import json
import re
import time
import os
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.data.preprocessing import wind_to_uv

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
S3_BASE = "https://openaq-data-archive.s3.amazonaws.com/"
OPENAQ_API = "https://api.openaq.org/v3"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
]
PM25_PARAMETER_ID = 2  # OpenAQ parameter id for PM2.5 (µg/m³)


# ---------------------------------------------------------------- Open-Meteo
def parse_openmeteo(payload: dict) -> pd.DataFrame:
    """Convert an Open-Meteo hourly JSON payload to a UTC DataFrame with u, v (m/s)."""
    units = payload["hourly_units"]
    if units["wind_speed_10m"] != "m/s":
        raise ValueError(f"wind speed must be in m/s, got {units['wind_speed_10m']}")
    if units["boundary_layer_height"] != "m":
        raise ValueError("boundary layer height must be in m")
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    df["u"], df["v"] = wind_to_uv(df["wind_speed_10m"], df["wind_direction_10m"])
    return df.rename(columns={"boundary_layer_height": "blh"})


def download_openmeteo(lat, lon, start: str, end: str) -> pd.DataFrame:
    params = dict(
        latitude=lat, longitude=lon, start_date=start, end_date=end,
        hourly=",".join(HOURLY_VARS), wind_speed_unit="ms", timezone="GMT",
    )
    r = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=60)
    r.raise_for_status()
    return parse_openmeteo(r.json())


# ---------------------------------------------------------------- OpenAQ
def openaq_s3_key(location_id: int, day: dt.date) -> str:
    return (
        f"records/csv.gz/locationid={location_id}/year={day:%Y}/month={day:%m}/"
        f"location-{location_id}-{day:%Y%m%d}.csv.gz"
    )


def _api_key() -> str:
    key = os.environ.get("OPENAQ_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("OPENAQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("Set OPENAQ_API_KEY in the environment or in .env")
    return key


def list_openaq_locations(bbox: dict) -> pd.DataFrame:
    """Monitoring locations measuring PM2.5 inside the bounding box."""
    params = dict(
        bbox=f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}",
        parameters_id=PM25_PARAMETER_ID, limit=1000,
    )
    r = requests.get(f"{OPENAQ_API}/locations", params=params, headers={"X-API-Key": _api_key()}, timeout=60)
    r.raise_for_status()
    rows = []
    for loc in r.json()["results"]:
        rows.append(dict(
            location_id=loc["id"], name=loc["name"],
            lat=loc["coordinates"]["latitude"], lon=loc["coordinates"]["longitude"],
            provider=(loc.get("provider") or {}).get("name"),
            owner=(loc.get("owner") or {}).get("name"),
            is_monitor=loc.get("isMonitor"),
            first=(loc.get("datetimeFirst") or {}).get("utc"),
            last=(loc.get("datetimeLast") or {}).get("utc"),
        ))
    return pd.DataFrame(rows)


def select_locations(locs: pd.DataFrame, start, end) -> pd.DataFrame:
    """Regulatory monitors (not low-cost sensors) whose reporting period covers [start, end]."""
    first = pd.to_datetime(locs["first"], utc=True)
    last = pd.to_datetime(locs["last"], utc=True)
    ok = locs["is_monitor"].fillna(False).astype(bool) & (first <= start) & (last >= end)
    return locs[ok.fillna(False)].reset_index(drop=True)


def _s3_get(url, params=None, attempts=4):
    """GET with a short timeout and retries: the archive intermittently stalls under load."""
    for a in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(1 + 2 * a)
    raise RuntimeError(f"S3 request failed after {attempts} attempts: {url}")


def download_openaq_location(location_id: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily archive files for [start - 1 day, end] (padding covers the IST/UTC offset).

    The archive has multi-year gaps for some stations, so the existing files are listed
    first (one request per month) and only those are fetched.
    """
    days = {start - dt.timedelta(days=1) + dt.timedelta(days=k) for k in range((end - start).days + 2)}
    wanted = {openaq_s3_key(location_id, d) for d in days}
    existing = set()
    for y, m in sorted({(d.year, d.month) for d in days}):
        prefix = f"records/csv.gz/locationid={location_id}/year={y}/month={m:02d}/"
        existing |= set(re.findall("<Key>([^<]*)</Key>", _s3_get(S3_BASE, {"list-type": "2", "prefix": prefix}).text))
    frames = []
    for key in sorted(wanted & existing):
        r = _s3_get(S3_BASE + key)
        if r.status_code == 200:
            frames.append(pd.read_csv(io.BytesIO(gzip.decompress(r.content))))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- main
def main():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    dom = cfg["domain"]
    (RAW / "openmeteo").mkdir(parents=True, exist_ok=True)
    (RAW / "openaq").mkdir(parents=True, exist_ok=True)
    manifest = {"downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "files": []}

    for name, w in cfg["windows"].items():
        # Pad by one day either side so the UTC window is fully covered.
        s = (pd.Timestamp(w["start"]) - pd.Timedelta(days=1)).date().isoformat()
        e = pd.Timestamp(w["end"]).date().isoformat()
        met = download_openmeteo(dom["lat0"], dom["lon0"], s, e)
        path = RAW / "openmeteo" / f"era5_{name}.csv"
        met.to_csv(path, index=False)
        manifest["files"].append(dict(file=str(path.relative_to(ROOT)), source=OPENMETEO_ARCHIVE, rows=len(met)))
        print(f"weather {name}: {len(met)} rows")

    try:
        locs = list_openaq_locations(dom)
    except RuntimeError as err:
        print(f"OpenAQ skipped: {err}")
    else:
        locs.to_csv(RAW / "openaq" / "locations.csv", index=False)
        print(f"OpenAQ: {len(locs)} PM2.5 locations in bbox")
        for name, w in cfg["windows"].items():
            s, e = pd.Timestamp(w["start"]).date(), pd.Timestamp(w["end"]).date()
            tz = cfg["data"]["timezone"]
            sel = select_locations(locs, pd.Timestamp(w["start"], tz=tz), pd.Timestamp(w["end"], tz=tz))
            print(f"{name}: {len(sel)} active regulatory monitors")
            with ThreadPoolExecutor(max_workers=6) as pool:
                frames = list(pool.map(lambda i: download_openaq_location(int(i), s, e), sel["location_id"]))
            raw = pd.concat([f for f in frames if len(f)], ignore_index=True)
            path = RAW / "openaq" / f"pm_{name}.csv.gz"
            raw.to_csv(path, index=False)
            manifest["files"].append(dict(file=str(path.relative_to(ROOT)), source=S3_BASE, rows=len(raw)))
            print(f"pm {name}: {len(raw)} raw records from {raw['location_id'].nunique()} locations")

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
