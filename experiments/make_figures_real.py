"""Figures for the real-data (OBSERVED) experiment.

Writes results/figures/real_*.png and report/figures/real_*.eps; captions are appended
to results/figures/captions_real.md.

Usage: .venv/bin/python -m experiments.make_figures_real
"""
from __future__ import annotations

import json

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from experiments.make_figures_twin import (AXIS, COL, GRID, INK, INK2, IST, LABEL, METRICS, PNG, ROOT, SEQ,
                                          CAPTIONS, save, to_ist)
from experiments.run_real import chunk_setup, new_model, setup
from src.training.sampling import sample_collocation

torch.set_num_threads(4)
WINDOWS = ("winter", "spring")
WTITLE = {"winter": "Winter, 10–20 Nov 2025", "spring": "Spring, 1–11 Mar 2025"}
SEEDS = (1, 2, 42)
COL = {**COL, "StationMean": "#6250d6"}
LABEL = {**LABEL, "StationMean": "Station mean"}
ORDER = ["StationMean", "IDW", "RandomForest", "DataOnlyNN", "PINN"]


def res(window, wp="0.01", seed=42):
    return json.loads((METRICS / f"real_{window}_wp{wp}_it6000_s{seed}.json").read_text())


def preds(window, wp="0.01", seed=42):
    return pd.read_csv(METRICS / f"real_{window}_wp{wp}_it6000_s{seed}_preds.csv", parse_dates=["time"])


def fig_map():
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
    for ax, w in zip(axs, WINDOWS):
        s = setup(w, 42)
        m = s["obs"].groupby("location_id").pm25.mean()
        st = s["stations"].set_index("location_id").assign(mean=m)
        xmin, xmax, ymin, ymax = (v / 1e3 for v in s["box"])
        mk = {"train": "o", "val": "s", "test": "^"}
        sc_ = None
        for part, marker in mk.items():
            d = st.loc[sorted(s["split"][part])]
            sc_ = ax.scatter(d.x, d.y, c=d["mean"], cmap=SEQ, vmin=m.min() * 0.8, vmax=m.max(), marker=marker,
                             s=46, edgecolors=INK, linewidths=0.8, zorder=3)
        cb = fig.colorbar(sc_, ax=ax, shrink=0.8)
        cb.set_label("Window-mean PM2.5 (µg/m³)")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", title=f"{WTITLE[w]}: {len(st)} monitors",
               xlabel="x, east (km)")
    axs[0].set_ylabel("y, north (km)")
    for mk_, lab in (("o", "training"), ("s", "validation"), ("^", "held-out test")):
        axs[0].scatter([], [], marker=mk_, c="white", edgecolors=INK, label=lab)
    axs[0].legend(loc="lower left", fontsize=7.5, frameon=True, facecolor="white", edgecolor=AXIS)
    save(fig, "real_station_map",
         "OBSERVED: CPCB/DPCC/IMD/IITM regulatory monitors (from OpenAQ) passing the data-quality checks in each "
         "window, coloured by window-mean PM2.5 and marked by role in the seed-42 split. Axes are the local tangent "
         "plane centred at 28.63°N, 77.15°E. No spatial interpolation is shown.")


def fig_series():
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.0), gridspec_kw=dict(width_ratios=[2.4, 1]))
    for row, w in enumerate(WINDOWS):
        o = setup(w, 42)["obs"]
        o = o.assign(t=to_ist(o.time))
        g = o.groupby("t").pm25
        q = g.quantile([0.1, 0.5, 0.9]).unstack()
        axs[row, 0].fill_between(q.index, q[0.1], q[0.9], color="#cde2fb", lw=0)
        axs[row, 0].plot(q.index, q[0.5], color=COL["PINN"], lw=1.6)
        axs[row, 0].set(ylabel="PM2.5 (µg/m³)", title=f"{WTITLE[w]}: median and 10–90 % range across monitors")
        axs[row, 0].xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axs[row, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        h = o.assign(h=o.t.dt.hour).groupby("h").pm25.quantile([0.25, 0.5, 0.75]).unstack()
        axs[row, 1].fill_between(h.index, h[0.25], h[0.75], color="#cde2fb", lw=0)
        axs[row, 1].plot(h.index, h[0.5], color=COL["PINN"], lw=1.6)
        axs[row, 1].set(title="Diurnal cycle (median, IQR)", xticks=[0, 6, 12, 18, 23], ylabel="µg/m³")
    axs[1, 0].set_xlabel("Local time (IST)")
    axs[1, 1].set_xlabel("Hour of day (IST)")
    save(fig, "real_pm_series",
         "OBSERVED hourly PM2.5. Left: median over monitors with the 10–90 % range; right: diurnal median and "
         "interquartile range. Note the different vertical scales: the November 2025 window is a severe "
         "post-monsoon episode, the March 2025 window a moderate one.")


def summary_table():
    rows = []
    for w in WINDOWS:
        for sd in SEEDS:
            r = res(w, seed=sd)["results"]
            for m in ORDER:
                rows.append(dict(window=w, seed=sd, model=m, rmse=r[m]["rmse"], mae=r[m]["mae"], r2=r[m]["r2"],
                                 bias=r[m]["bias"]))
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "real_seed_summary.csv", index=False)
    return df


def fig_comparison(df):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for ax, w in zip(axs, WINDOWS):
        d = df[df.window == w]
        g = d.groupby("model").rmse.agg(["mean", "std"]).loc[ORDER]
        xs = np.arange(len(ORDER))
        ax.bar(xs, g["mean"], width=0.62, color=[COL[m] for m in ORDER], edgecolor="white", linewidth=2)
        ax.errorbar(xs, g["mean"], yerr=g["std"], fmt="none", ecolor=INK, capsize=3, lw=1)
        for x, m in zip(xs, ORDER):
            v = d[d.model == m].rmse
            ax.scatter(np.full(len(v), x) + np.linspace(-0.1, 0.1, len(v)), v, s=9, color=INK, zorder=3)
            top = max(g.loc[m, "mean"] + g.loc[m, "std"], v.max())
            ax.text(x, top * 1.03, f"{g.loc[m, 'mean']:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xs, [LABEL[m].replace(" ", "\n", 1) for m in ORDER], fontsize=7.5)
        ax.set(title=f"{WTITLE[w]}\nRMSE at held-out monitors", ylabel="µg/m³", ylim=(0, d.rmse.max() * 1.2))
        ax.grid(axis="x", visible=False)
    save(fig, "real_comparison",
         "OBSERVED: RMSE at held-out monitors, mean ± one standard deviation over three station splits "
         "(seeds 1, 2, 42; dots are individual splits). The physics weight w_p = 0.01 was fixed from the synthetic "
         "twin before any real-data run. 'Station mean' predicts the hourly mean of the training monitors everywhere.")


def fig_parity():
    fig, axs = plt.subplots(2, 3, figsize=(8.0, 5.4))
    for row, w in enumerate(WINDOWS):
        p = preds(w)
        r = res(w)["results"]
        lim = (0, np.nanpercentile(p.obs, 99.5) * 1.1)
        for ax, m in zip(axs[row], ("IDW", "DataOnlyNN", "PINN")):
            ax.plot(lim, lim, color=AXIS, lw=1)
            ax.scatter(p.obs, p[m], s=4, color=COL[m], rasterized=True)
            ax.set(xlim=lim, ylim=lim, aspect="equal", title=f"{LABEL[m]} ({w})")
            ax.text(0.04, 0.96, f"RMSE {r[m]['rmse']:.1f}\nR² {r[m]['r2']:.2f}", transform=ax.transAxes,
                    va="top", fontsize=7.5)
        axs[row, 0].set_ylabel("Predicted (µg/m³)")
    for ax in axs[1]:
        ax.set_xlabel("Observed (µg/m³)")
    save(fig, "real_parity",
         "OBSERVED: predicted against measured hourly PM2.5 at the held-out monitors (seed 42). Grey: 1:1 line.")


def fig_station_series():
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 5.0))
    for ax, w in zip(axs, WINDOWS):
        p = preds(w)
        s = setup(w, 42)
        names = s["stations"].set_index("location_id")["location"]
        sid = p.groupby("location_id").obs.count().idxmax()
        d = p[p.location_id == sid].sort_values("time")
        t = to_ist(d.time)
        ax.scatter(t, d.obs, s=6, color=INK, label="measured", zorder=3)
        for m in ("IDW", "DataOnlyNN", "PINN"):
            ax.plot(t, d[m], color=COL[m], lw=1.3, label=LABEL[m])
        ax.set(ylabel="PM2.5 (µg/m³)", title=f"{WTITLE[w]}: held-out monitor {names[sid].split(' - ')[0]}")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axs[0].legend(ncol=4, fontsize=7.5, loc="upper right")
    axs[1].set_xlabel("Local time (IST)")
    save(fig, "real_station_series",
         "OBSERVED: hourly PM2.5 at one held-out monitor per window (the test monitor with most valid hours, seed 42) "
         "and the reconstructions from the training monitors. The networks are trained on consecutive 80-hour chunks.")


def fig_wp():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, w in zip(axs, WINDOWS):
        wps = ["0.003", "0.01", "0.03"]
        rm = [res(w, wp)["results"]["PINN"]["rmse"] for wp in wps]
        nn = np.mean([res(w, wp)["results"]["DataOnlyNN"]["rmse"] for wp in wps])
        val = []
        for wp in wps:
            h = res(w, wp)["histories"]
            val.append(np.mean([min(r["val"] for r in v) for k, v in h.items() if k.startswith("PINN")]))
        x = [float(v) for v in wps]
        ax.plot(x, rm, color=COL["PINN"], marker="o", ms=6, mec="white", mew=1.5)
        ax.axhline(nn, color=COL["DataOnlyNN"], ls="--", lw=1.4)
        ax.text(x[-1], nn, " data-only NN", va="bottom", ha="right", fontsize=7.5)
        for xi, ri, vi in zip(x, rm, val):
            ax.text(xi, ri, f"  val MSE {vi:.3f}", fontsize=7, va="center")
        ax.set(xscale="log", xlabel="Physics-loss weight w_p", ylabel="Test RMSE (µg/m³)", title=WTITLE[w],
               xlim=(0.002, 0.08))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(x, wps)
    save(fig, "real_wp_sweep",
         "OBSERVED: sensitivity of the PINN to the physics weight (seed 42). Labels give the validation-station MSE "
         "(dimensionless, mean over chunks) that a practitioner without access to the test monitors would see. "
         "The reported configuration (w_p = 0.01) was fixed in advance from the synthetic twin.")


def fig_residual():
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 2.9), sharey=True)
    bins = np.logspace(-3, 2.5, 50)
    for ax, w in zip(axs, WINDOWS):
        s = setup(w, 42)
        _, sc, fz = chunk_setup(s, 0, 80)
        X = sample_collocation(20000, seed=7).requires_grad_(True)
        for m in ("DataOnlyNN", "PINN"):
            model = new_model(s, sc, fz)
            model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"real_{w}_wp0.01_it6000_s42_chunk0_{m}.pt", weights_only=True))
            if m == "DataOnlyNN":
                from experiments.make_figures_twin import fit_source_posthoc
                model = fit_source_posthoc(model)
            r = model.residual(X).detach().abs().numpy()
            lab = f"{LABEL[m]}{' (source, D fitted afterwards)' if m == 'DataOnlyNN' else ''}: median {np.median(r):.2f}"
            ax.hist(r, bins=bins, histtype="step", color=COL[m], lw=1.6, label=lab)
        ax.set(xscale="log", xlabel="|PDE residual| (dimensionless)", title=f"{WTITLE[w]}, first chunk")
        ax.legend(fontsize=6.8, loc="upper left")
    axs[0].set_ylabel("Collocation points")
    save(fig, "real_residual",
         "OBSERVED-data models: PDE residual at 20 000 random points in the first 80-hour chunk (seed 42). For a fair "
         "comparison the data-only network's field is frozen and its emission model and D are fitted to it afterwards.")


def fig_wind_check():
    """Do the monitors' own wind measurements support a spatially uniform ERA5 wind?"""
    # Only the November 2025 archive files carry the stations' wind channels (March 2025 has none).
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    out = {}
    for w in ("winter",):
        raw = pd.read_csv(ROOT / "data" / "raw" / "openaq" / f"pm_{w}.csv.gz")
        ws = raw[(raw.parameter == "wind_speed") & (raw.units == "m/s")].copy()
        ws["value"] = pd.to_numeric(ws.value, errors="coerce")
        ws = ws[(ws.value >= 0) & (ws.value < 30)]
        ws["time"] = pd.to_datetime(ws.datetime, utc=True).dt.floor("h")
        hourly = ws.groupby(["location_id", "time"]).value.mean().unstack(0)
        met = pd.read_csv(ROOT / "data" / "processed" / "real" / f"{w}_forcing.csv", parse_dates=["time"]).set_index("time")
        idx = hourly.index.intersection(met.index)
        med = hourly.loc[idx].median(axis=1)
        q1, q3 = hourly.loc[idx].quantile(0.25, axis=1), hourly.loc[idx].quantile(0.75, axis=1)
        t = idx.tz_convert(IST).tz_localize(None)
        ax.fill_between(t, q1, q3, color="#e1e0d9", lw=0)
        ax.plot(t, med, color=INK2, lw=1.2, label=f"monitors, median (IQR band), n = {hourly.shape[1]}")
        ax.plot(t, met.loc[idx, "wind_speed_10m"], color=INK, lw=1.6, label="ERA5 10 m")
        r = np.corrcoef(med, met.loc[idx, "wind_speed_10m"])[0, 1]
        out[w] = dict(n_monitors=int(hourly.shape[1]), r=float(r), era5_mean=float(met.loc[idx, "wind_speed_10m"].mean()),
                      monitor_median_mean=float(med.mean()))
        ax.set(title=f"{WTITLE[w]}: r = {r:.2f}", ylabel="Wind speed (m/s)")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.legend(fontsize=6.8, loc="upper right")
    save(fig, "real_wind_check",
         "Check of the spatially uniform wind assumption: hourly wind speed reported by the monitoring stations "
         "(OBSERVED; median and interquartile range across stations) against the single ERA5 value used in the PDE. "
         f"Winter window only: the March 2025 archive files contain no station wind data. Window means: monitors "
         f"{out['winter']['monitor_median_mean']:.1f} m/s, ERA5 {out['winter']['era5_mean']:.1f} m/s; correlation of "
         f"hourly values r = {out['winter']['r']:.2f}. Station anemometers sit within the urban canopy, so a lower "
         "reading than the 10 m reanalysis is expected; the timing agreement is what supports using ERA5 as the "
         "transport wind.")
    return out


def main():
    CAPTIONS.clear()
    fig_map()
    fig_series()
    df = summary_table()
    fig_comparison(df)
    fig_parity()
    fig_station_series()
    fig_wp()
    fig_residual()
    wind = fig_wind_check()
    with open(PNG / "captions_real.md", "w") as f:
        for k, v in CAPTIONS.items():
            f.write(f"### {k}\n\n{v}\n\n")
    print(df.groupby(["window", "model"])[["rmse", "mae", "r2", "bias"]].agg(["mean", "std"]).round(1).to_string())
    print("wind check", json.dumps(wind, indent=1))


if __name__ == "__main__":
    main()
