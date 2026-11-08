"""Generate the SYNTHETIC-TWIN data set.

Truth: the ADR equation solved by src/physics/solver.py with known D, lambda,
emission field E(x, y), diurnal profile g and inflow concentration, driven by the
REAL ERA5 wind and boundary-layer height of the winter window. The truth is sampled
hourly at station locations with multiplicative noise.

Station locations: real Delhi monitor coordinates from data/raw/openaq/locations.csv
when available, otherwise a seeded random PLACEHOLDER layout (recorded in the output).

Usage: .venv/bin/python -m experiments.make_synthetic_twin
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocessing import project_to_local, validate_coordinates
from src.data.synthetic import bilinear_sample, diurnal_profile
from src.physics.solver import Grid, solve_adr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "twin"


def domain_km(dom):
    x0, y0 = project_to_local([dom["lat_min"]], [dom["lon_min"]], dom["lat0"], dom["lon0"])
    x1, y1 = project_to_local([dom["lat_max"]], [dom["lon_max"]], dom["lat0"], dom["lon0"])
    return float(x0[0]), float(x1[0]), float(y0[0]), float(y1[0])


def emission_field(X_km, Y_km, tw):
    E = np.full_like(X_km, tw["E_background"])
    for xc, yc, s, peak in tw["hotspots"]:
        E += peak * np.exp(-0.5 * ((X_km - xc) ** 2 + (Y_km - yc) ** 2) / s**2)
    return E


def stations(cfg, xr, yr):
    dom, tw = cfg["domain"], cfg["twin"]
    # The monitors that passed cleaning and coverage in the real winter window.
    path = ROOT / "data" / "processed" / "real" / "winter_stations.csv"
    if path.exists():
        st = validate_coordinates(pd.read_csv(path), dom["lat_min"], dom["lat_max"], dom["lon_min"], dom["lon_max"])
        x, y = project_to_local(st["lat"], st["lon"], dom["lat0"], dom["lon0"])
        return pd.DataFrame({"location_id": st["location_id"], "x": x, "y": y}), "real_winter_monitor_coordinates"
    rng = np.random.default_rng(cfg["seed"])
    n = tw["placeholder_stations"]
    # Keep a 3 km margin from the boundary, like a real urban network.
    x = rng.uniform(xr[0] + 3, xr[1] - 3, n)
    y = rng.uniform(yr[0] + 3, yr[1] - 3, n)
    return pd.DataFrame({"location_id": np.arange(n), "x": x, "y": y}), "placeholder_random_layout"


def simulate(cfg, D=None, lam=None, H_min=None):
    """Run the reference solver for the twin window. Returns grid, hourly t_rel, fields, E, forcing, start."""
    tw, ph = cfg["twin"], cfg["physics"]
    D = tw["D_true"] if D is None else D
    lam = tw["lam_true"] if lam is None else lam
    H_min = ph["H_min"] if H_min is None else H_min
    xmin, xmax, ymin, ymax = domain_km(cfg["domain"])
    h = tw["grid_km"]
    grid = Grid(x_min=xmin * 1e3, x_max=xmax * 1e3, y_min=ymin * 1e3, y_max=ymax * 1e3,
                nx=int(round((xmax - xmin) / h)) + 1, ny=int(round((ymax - ymin) / h)) + 1)

    met = pd.read_csv(ROOT / "data" / "raw" / "openmeteo" / "era5_twin_2024-11.csv", parse_dates=["time"])
    start = pd.Timestamp(tw["start"])
    t_spin = start - pd.Timedelta(hours=tw["spinup_hours"])
    end = start + pd.Timedelta(days=tw["days"])
    met = met[(met.time >= t_spin) & (met.time <= end)].reset_index(drop=True)
    ts = (met.time - t_spin).dt.total_seconds().to_numpy()
    H = np.maximum(met.blh.to_numpy(), H_min)
    local_hour0 = (t_spin.hour + 5.5) % 24

    u = lambda t: np.interp(t, ts, met.u)
    v = lambda t: np.interp(t, ts, met.v)
    E = emission_field(grid.X / 1e3, grid.Y / 1e3, tw)
    S = lambda X, Y, t: E * diurnal_profile(local_hour0 + t / 3600.0) / np.interp(t, ts, H)

    sol = solve_adr(grid, C0=np.full(grid.shape, tw["C_inflow"]), t_end=ts[-1],
                    u=u, v=v, D=D, lam=lam, S=S,
                    C_bc=lambda t: tw["C_inflow"], save_every=3600.0)
    keep = sol["t"] >= tw["spinup_hours"] * 3600.0 - 1e-6
    t_rel = sol["t"][keep] - tw["spinup_hours"] * 3600.0
    return grid, t_rel, sol["C"][keep], E, met, start


def main():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    tw = cfg["twin"]
    grid, t_rel, fields, E, met, start = simulate(cfg)
    xmin, xmax, ymin, ymax = (v / 1e3 for v in (grid.x_min, grid.x_max, grid.y_min, grid.y_max))

    st, layout = stations(cfg, (xmin, xmax), (ymin, ymax))
    rng = np.random.default_rng(cfg["seed"] + 1)
    rows = []
    for k, (tr, F) in enumerate(zip(t_rel, fields)):
        c = bilinear_sample(grid, F, st.x * 1e3, st.y * 1e3)
        noisy = c * (1 + tw["noise_rel"] * rng.standard_normal(len(c)))
        for sid, x, y, ct, cn in zip(st.location_id, st.x, st.y, c, noisy):
            rows.append(dict(location_id=sid, time=start + pd.Timedelta(seconds=tr), t_rel=tr,
                             x=x, y=y, pm25_true=ct, pm25=cn))
    obs = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT / "observations.csv", index=False)
    np.savez_compressed(OUT / "truth.npz", t_rel=t_rel, C=fields.astype(np.float32),
                        x=grid.x, y=grid.y, E=E)
    w = met[met.time >= start].reset_index(drop=True)
    w.assign(t_rel=(w.time - start).dt.total_seconds()).to_csv(OUT / "forcing.csv", index=False)
    meta = dict(kind="SYNTHETIC", station_layout=layout, n_stations=len(st), grid=[grid.nx, grid.ny],
                domain_km=[xmin, xmax, ymin, ymax], start=str(start), hours=len(t_rel),
                **{k: tw[k] for k in ("D_true", "lam_true", "C_inflow", "noise_rel")})
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"truth C range {fields.min():.1f}–{fields.max():.1f} µg/m³, station mean {obs.pm25.mean():.1f}")


if __name__ == "__main__":
    main()
