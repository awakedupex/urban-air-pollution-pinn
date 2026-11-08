"""Real-data experiment (OBSERVED PM2.5): reconstruct held-out Delhi monitors.

Each 10-day window is split into consecutive chunks (time-domain decomposition,
default 80 h) and a separate data-only NN and PINN are trained per chunk, so the
dimensionless advection coefficient stays comparable to the synthetic twin.

Models (identical station split for all):
  training-station mean (spatially uniform field), IDW, random forest,
  data-only NN (w_p = 0), PINN.

Usage: .venv/bin/python -m experiments.run_real --window winter [--wp 0.01] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.make_synthetic_twin import domain_km
from experiments.run_twin import add_features
from src.data.preprocessing import split_stations
from src.evaluation.metrics import evaluate
from src.evaluation.validation import assert_no_station_leakage
from src.models.baselines import idw_predict, random_forest_predict
from src.models.pinn import PINN, Forcing, Scales
from src.training.trainer import TrainConfig, config_dict, train

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "processed" / "real"


def setup(window, seed=None):
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    if seed is not None:
        cfg["seed"] = seed
    obs = pd.read_csv(REAL / f"{window}_observations.csv", parse_dates=["time"])
    forcing = pd.read_csv(REAL / f"{window}_forcing.csv", parse_dates=["time"])
    stations = obs.drop_duplicates("location_id")[["location_id", "location", "x", "y"]]
    split = split_stations(stations, **{k: cfg["split"][k] for k in ("train", "val", "test")}, seed=cfg["seed"])
    part = {k: obs[obs.location_id.isin(v)].reset_index(drop=True) for k, v in split.items()}
    assert_no_station_leakage(part["train"], part["test"])
    assert_no_station_leakage(part["val"], part["test"])
    xmin, xmax, ymin, ymax = (v * 1e3 for v in domain_km(cfg["domain"]))
    start = pd.Timestamp(cfg["windows"][window]["start"], tz=cfg["data"]["timezone"]).tz_convert("UTC")
    return dict(cfg=cfg, obs=obs, forcing=forcing, stations=stations, split=split, part=part,
                box=(xmin, xmax, ymin, ymax), start=start)


def chunk_setup(s, c0_h, c1_h):
    """Scales, forcing and per-split tables for hours [c0_h, c1_h) of the window."""
    ph = s["cfg"]["physics"]
    xmin, xmax, ymin, ymax = s["box"]
    sel = lambda d: d[(d.t_rel >= c0_h * 3600) & (d.t_rel < c1_h * 3600)].assign(
        t_rel=lambda q: q.t_rel - c0_h * 3600).reset_index(drop=True)
    part = {k: sel(v) for k, v in s["part"].items()}
    f = s["forcing"]
    f = f[(f.t_rel >= (c0_h - 1) * 3600) & (f.t_rel <= c1_h * 3600)].assign(t_rel=lambda q: q.t_rel - c0_h * 3600)
    sc = Scales(x_c=(xmin + xmax) / 2, y_c=(ymin + ymax) / 2, L=max(xmax - xmin, ymax - ymin) / 2,
                t_start=s["start"] + pd.Timedelta(hours=c0_h), T=float((c1_h - c0_h) * 3600),
                C_ref=float(part["train"].pm25.mean()))
    fz = Forcing(f.t_rel, f.u, f.v, f.blh, H_min=ph["H_min"], H_ref=ph["H_ref"])
    return part, sc, fz


def new_model(s, sc, fz):
    ph = s["cfg"]["physics"]
    return PINN(sc, fz, D_min=ph["D_min"], D_max=ph["D_max"], D_init=ph["D_init"], lam=ph["lam"],
                seed=s["cfg"]["seed"])


def run(args):
    s = setup(args.window, args.seed)
    cfg, part = s["cfg"], s["part"]
    seed = cfg["seed"]
    forcing = s["forcing"]
    test = add_features(part["test"], forcing)
    train_f = add_features(part["train"], forcing)

    preds = {}
    hourly_mean = part["train"].groupby("time").pm25.mean()
    preds["StationMean"] = test.time.map(hourly_mean).to_numpy()
    preds["IDW"] = idw_predict(part["train"], test)
    preds["RandomForest"] = random_forest_predict(train_f, test, ["x", "y", "t_rel", "hs", "hc", "u", "v", "blh"], seed=seed)

    # IST windows start at hh:30 UTC while observations sit on whole UTC hours, so t_rel
    # values are offset by half an hour; the window spans ceil(max t_rel) hours.
    n_hours = int(np.ceil(s["obs"].t_rel.max() / 3600))
    edges = list(range(0, n_hours, args.chunk_hours)) + [n_hours]
    tag = args.tag or f"{args.window}_wp{args.wp:g}_it{args.iters}_s{seed}"
    ck = ROOT / "results" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    nn_pred = {n: np.full(len(test), np.nan) for n in ("DataOnlyNN", "PINN")}
    histories, D_learned, tcfgs = {}, [], {}
    for c, (c0, c1) in enumerate(zip(edges[:-1], edges[1:])):
        cp, sc, fz = chunk_setup(s, c0, c1)
        tens = lambda d: (sc.inputs(d.x * 1e3, d.y * 1e3, d.t_rel),
                          torch.as_tensor(d.pm25.to_numpy() / sc.C_ref, dtype=torch.float32))
        Xtr, ctr = tens(cp["train"])
        val = tens(cp["val"])
        rows = ((test.t_rel >= c0 * 3600) & (test.t_rel < c1 * 3600)).to_numpy()
        Xte = sc.inputs(test.x[rows] * 1e3, test.y[rows] * 1e3, test.t_rel[rows] - c0 * 3600)
        for name, wp in (("DataOnlyNN", 0.0), ("PINN", args.wp)):
            model = new_model(s, sc, fz)
            tcfg = TrainConfig(n_iters=args.iters, seed=seed, w_p=wp,
                               w_b=0.1 if wp > 0 else 0.0, w_s=1e-3 if wp > 0 else 0.0)
            t0 = time.time()
            histories[f"{name}_chunk{c}"] = train(model, Xtr, ctr, tcfg, val=val)
            with torch.no_grad():
                nn_pred[name][rows] = model(Xte).numpy() * sc.C_ref
            torch.save(model.state_dict(), ck / f"real_{tag}_chunk{c}_{name}.pt")
            tcfgs[name] = config_dict(tcfg)
            if name == "PINN":
                D_learned.append(model.D().item())
            print(f"chunk {c} ({c0}-{c1} h) {name}: {time.time() - t0:.0f}s, D={model.D().item():.0f}", flush=True)
    preds.update(nn_pred)

    results = {n: evaluate(test.pm25, p) for n, p in preds.items()}
    results["PINN"]["D_learned_per_chunk"] = D_learned
    out = ROOT / "results" / "metrics"
    (out / f"real_{tag}.json").write_text(json.dumps(dict(
        kind="OBSERVED", window=args.window, chunks=[[a, b] for a, b in zip(edges[:-1], edges[1:])],
        split={k: sorted(map(int, v)) for k, v in s["split"].items()},
        train_config=tcfgs, results=results, histories=histories), indent=2))
    pd.DataFrame({"location_id": test.location_id, "time": test.time, "t_rel": test.t_rel, "obs": test.pm25,
                  **preds}).to_csv(out / f"real_{tag}_preds.csv", index=False)

    print(f"\n{args.window} ({len(s['split']['test'])} held-out stations)")
    print(f"{'model':14s} {'MAE':>7s} {'RMSE':>7s} {'R2':>6s} {'bias':>7s}")
    for n, r in results.items():
        print(f"{n:14s} {r['mae']:7.2f} {r['rmse']:7.2f} {r['r2']:6.3f} {r['bias']:7.2f}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--window", choices=["winter", "spring"], required=True)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--wp", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--chunk_hours", type=int, default=80)
    p.add_argument("--tag", default=None)
    run(p.parse_args())
