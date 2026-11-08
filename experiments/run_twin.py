"""Synthetic-twin experiment: reconstruct PM2.5 at held-out stations.

Models (identical station split, identical network size and seed for NN/PINN):
  IDW, random forest, data-only NN (w_p = 0), PINN.
Scores: test stations vs noisy observations and vs noise-free truth, plus
full-field error against the known truth (possible only in the synthetic twin).

Usage: .venv/bin/python -m experiments.run_twin [--iters 6000] [--wp 1.0] [--tag name]
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

from src.data.preprocessing import split_stations
from src.evaluation.metrics import evaluate
from src.evaluation.validation import assert_no_station_leakage
from src.models.baselines import idw_predict, random_forest_predict
from src.models.pinn import PINN, Forcing, Scales
from src.training.trainer import TrainConfig, config_dict, train

ROOT = Path(__file__).resolve().parents[1]
TWIN = ROOT / "data" / "processed" / "twin"


def load():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    obs = pd.read_csv(TWIN / "observations.csv", parse_dates=["time"])
    forcing = pd.read_csv(TWIN / "forcing.csv", parse_dates=["time"])
    meta = json.loads((TWIN / "meta.json").read_text())
    truth = dict(np.load(TWIN / "truth.npz"))
    return cfg, obs, forcing, meta, truth


def add_features(df, forcing):
    f = forcing.set_index("t_rel")
    out = df.copy()
    for c in ("u", "v", "blh"):
        out[c] = np.interp(out.t_rel, f.index, f[c])
    hour = (out.time.dt.hour + 5.5) % 24
    out["hs"], out["hc"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    return out


def setup(seed=None):
    """Data, station split, scales and forcing exactly as used for training (shared with figures)."""
    cfg, obs, forcing, meta, truth = load()
    if seed is not None:
        cfg["seed"] = seed  # changes both the station split and the network initialisation
    ph = cfg["physics"]
    stations = obs.drop_duplicates("location_id")[["location_id", "x", "y"]]
    split = split_stations(stations, **{k: cfg["split"][k] for k in ("train", "val", "test")}, seed=cfg["seed"])
    part = {k: obs[obs.location_id.isin(v)].reset_index(drop=True) for k, v in split.items()}
    assert_no_station_leakage(part["train"], part["test"])
    assert_no_station_leakage(part["val"], part["test"])

    xmin, xmax, ymin, ymax = [v * 1e3 for v in meta["domain_km"]]
    sc = Scales(x_c=(xmin + xmax) / 2, y_c=(ymin + ymax) / 2, L=max(xmax - xmin, ymax - ymin) / 2,
                t_start=pd.Timestamp(meta["start"]), T=float(obs.t_rel.max()),
                C_ref=float(part["train"].pm25.mean()))
    fz = Forcing(forcing.t_rel, forcing.u, forcing.v, forcing.blh, H_min=ph["H_min"], H_ref=ph["H_ref"])
    return dict(cfg=cfg, obs=obs, forcing=forcing, meta=meta, truth=truth, split=split, part=part, sc=sc, fz=fz)


def new_model(s):
    ph = s["cfg"]["physics"]
    return PINN(s["sc"], s["fz"], D_min=ph["D_min"], D_max=ph["D_max"], D_init=ph["D_init"], lam=ph["lam"],
                seed=s["cfg"]["seed"])


def run(args):
    s = setup(args.seed)
    cfg, obs, forcing, meta, truth = s["cfg"], s["obs"], s["forcing"], s["meta"], s["truth"]
    split, part, sc, fz = s["split"], s["part"], s["sc"], s["fz"]
    tens = lambda d: (sc.inputs(d.x * 1e3, d.y * 1e3, d.t_rel), torch.as_tensor(d.pm25.to_numpy() / sc.C_ref, dtype=torch.float32))
    Xtr, ctr = tens(part["train"])
    val = tens(part["val"])

    results, preds, histories = {}, {}, {}
    test = add_features(part["test"], forcing)
    train_f = add_features(part["train"], forcing)

    preds["IDW"] = idw_predict(part["train"], test)
    feats = ["x", "y", "t_rel", "hs", "hc", "u", "v", "blh"]
    preds["RandomForest"] = random_forest_predict(train_f, test, feats, seed=cfg["seed"])

    nets = {}
    for name, wp in (("DataOnlyNN", 0.0), (f"PINN", args.wp)):
        model = new_model(s)
        tcfg = TrainConfig(n_iters=args.iters, seed=cfg["seed"], w_p=wp,
                           w_b=0.1 if wp > 0 else 0.0, w_s=1e-3 if wp > 0 else 0.0,
                           adaptive=args.adaptive and wp > 0)
        t0 = time.time()
        histories[name] = train(model, Xtr, ctr, tcfg, val=val)
        print(f"{name}: {time.time() - t0:.0f}s, best val MSE {min(h['val'] for h in histories[name]):.4f}, D={model.D().item():.0f}")
        with torch.no_grad():
            preds[name] = model(sc.inputs(test.x * 1e3, test.y * 1e3, test.t_rel)).numpy() * sc.C_ref
        nets[name] = (model, tcfg)

    for name, p in preds.items():
        results[name] = {"vs_obs": evaluate(test.pm25, p), "vs_truth": evaluate(test.pm25_true, p)}

    # Full-field error against the synthetic truth, every 3 h on the solver grid.
    gx, gy = np.meshgrid(truth["x"], truth["y"], indexing="ij")
    idx = np.arange(0, len(truth["t_rel"]), 3)
    for name in ("DataOnlyNN", "PINN"):
        model = nets[name][0]
        err, ref = [], []
        with torch.no_grad():
            for k in idx:
                X = sc.inputs(gx.ravel(), gy.ravel(), np.full(gx.size, truth["t_rel"][k]))
                err.append(model(X).numpy() * sc.C_ref - truth["C"][k].ravel())
                ref.append(truth["C"][k].ravel())
        err, ref = np.concatenate(err), np.concatenate(ref)
        results[name]["field"] = {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))),
                                  "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(ref))}
    results["PINN"]["D_learned"] = nets["PINN"][0].D().item()
    results["PINN"]["D_true"] = meta["D_true"]

    tag = args.tag or f"wp{args.wp:g}{'_adapt' if args.adaptive else ''}_it{args.iters}_s{cfg['seed']}"
    out = ROOT / "results" / "metrics" / f"twin_{tag}.json"
    out.write_text(json.dumps(dict(kind="SYNTHETIC", meta=meta, split={k: sorted(map(int, v)) for k, v in split.items()},
                                   train_config={n: config_dict(c) for n, (_, c) in nets.items()},
                                   results=results, histories=histories), indent=2))
    ck = ROOT / "results" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    for name, (m, _) in nets.items():
        torch.save(m.state_dict(), ck / f"twin_{tag}_{name}.pt")
    pd.DataFrame({"location_id": test.location_id, "t_rel": test.t_rel, "obs": test.pm25, "truth": test.pm25_true,
                  **{k: v for k, v in preds.items()}}).to_csv(ROOT / "results" / "metrics" / f"twin_{tag}_preds.csv", index=False)

    print(f"\n{'model':14s} {'MAE':>7s} {'RMSE':>7s} {'R2':>6s} | vs truth RMSE | field RMSE")
    for name, r in results.items():
        f = r.get("field", {}).get("rmse", float("nan"))
        print(f"{name:14s} {r['vs_obs']['mae']:7.2f} {r['vs_obs']['rmse']:7.2f} {r['vs_obs']['r2']:6.3f} | "
              f"{r['vs_truth']['rmse']:13.2f} | {f:10.2f}")
    print(f"PINN D learned {results['PINN']['D_learned']:.0f} (true {meta['D_true']})")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--wp", type=float, default=1.0)
    p.add_argument("--tag", default=None)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    run(p.parse_args())
