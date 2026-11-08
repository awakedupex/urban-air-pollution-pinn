"""Figures for the synthetic-twin study (and the ERA5 forcing, which is real).

Every figure is written twice: results/figures/<name>.png (notebook) and
report/figures/<name>.eps (LaTeX). Captions live in results/figures/captions.md.

Usage: .venv/bin/python -m experiments.make_figures_twin
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from experiments.make_synthetic_twin import simulate
from experiments.run_twin import setup, new_model
from src.data.synthetic import bilinear_sample, diurnal_profile
from src.training.sampling import sample_collocation

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
PNG, EPS = ROOT / "results" / "figures", ROOT / "report" / "figures"
METRICS = ROOT / "results" / "metrics"
CK = ROOT / "results" / "checkpoints"
SEEDS = (1, 2, 3, 42)
BEST = "wp0.01_it6000"

# Validated categorical palette (fixed order; colour follows the model everywhere).
COL = {"PINN": "#2a78d6", "DataOnlyNN": "#eb6834", "IDW": "#1baf7a", "RandomForest": "#eda100"}
LABEL = {"PINN": "PINN", "DataOnlyNN": "Data-only NN", "IDW": "IDW", "RandomForest": "Random forest"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SEQ = "Blues"
IST = "Asia/Kolkata"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight", "ps.fonttype": 42, "pdf.fonttype": 42,
})
CAPTIONS = {}


def save(fig, name, caption):
    PNG.mkdir(parents=True, exist_ok=True)
    EPS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG / f"{name}.png", dpi=200)
    fig.savefig(EPS / f"{name}.eps")
    fig.savefig(EPS / f"{name}.pdf")  # tectonic includes PDF; EPS is kept for submission/pdflatex
    plt.close(fig)
    CAPTIONS[name] = caption
    print("saved", name)


def predict(model, sc, x_km, y_km, t_rel):
    with torch.no_grad():
        return model(sc.inputs(np.asarray(x_km) * 1e3, np.asarray(y_km) * 1e3, np.asarray(t_rel))).numpy() * sc.C_ref


def load_models(s, tag, seed):
    out = {}
    for name in ("DataOnlyNN", "PINN"):
        m = new_model(s)
        m.load_state_dict(torch.load(CK / f"twin_{tag}_s{seed}_{name}.pt", weights_only=True))
        m.eval()
        out[name] = m
    return out


def results(tag, seed):
    return json.loads((METRICS / f"twin_{tag}_s{seed}.json").read_text())


def to_ist(t):
    return pd.to_datetime(t).dt.tz_convert(IST).dt.tz_localize(None) if hasattr(t, "dt") else t


# ---------------------------------------------------------------- figures
def fig_study_area(s):
    tw = s["truth"]
    sc_split = s["split"]
    st = s["obs"].drop_duplicates("location_id")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    X, Y = np.meshgrid(tw["x"] / 1e3, tw["y"] / 1e3, indexing="ij")
    cs = ax.contourf(X, Y, tw["E"], levels=12, cmap=SEQ)
    cb = fig.colorbar(cs, ax=ax, shrink=0.85)
    cb.set_label("Synthetic emission flux E (µg m⁻² s⁻¹)")
    marker = {"train": ("o", "Training station"), "val": ("s", "Validation station"), "test": ("^", "Held-out test station")}
    for part, (mk, lab) in marker.items():
        d = st[st.location_id.isin(sc_split[part])]
        ax.scatter(d.x, d.y, marker=mk, s=42 if part != "train" else 26, c=INK if part == "train" else "white",
                   edgecolors=INK, linewidths=1.0, label=lab, zorder=3)
    f = s["forcing"]
    ub, vb = f.u.mean(), f.v.mean()
    ax.annotate("", xy=(-6 + 5 * ub / np.hypot(ub, vb), 24 + 5 * vb / np.hypot(ub, vb)), xytext=(-6, 24),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.5, 24, f"window-mean ERA5 wind\n{np.hypot(ub, vb):.1f} m/s", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.set(xlabel="x, east of origin (km)", ylabel="y, north of origin (km)", aspect="equal",
           title="Synthetic twin: domain, emission field and station split")
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=1, frameon=True, facecolor="white", edgecolor=AXIS)
    save(fig, "twin_study_area",
         "Model domain (local tangent plane centred at 28.63°N, 77.15°E, covering the Delhi NCT bounding box). "
         "Shading: the SYNTHETIC emission flux used to generate the twin (background plus three illustrative hotspots, "
         "not an inventory). Markers: the 45 real CPCB/DPCC/IMD monitor locations that pass data-quality checks in "
         "the winter window, and their train/validation/test split (seed 42). Arrow: window-mean ERA5 10 m wind.")


def fig_forcing(cfg):
    # The twin is driven by the Nov 2024 ERA5 forcing (kept separately from the real-data windows).
    met = pd.read_csv(ROOT / "data" / "raw" / "openmeteo" / "era5_twin_2024-11.csv", parse_dates=["time"])
    start = pd.Timestamp("2024-11-10", tz=IST)
    end = pd.Timestamp("2024-11-20", tz=IST)
    met = met[(met.time >= start) & (met.time < end)]
    t = to_ist(met.time)
    tw0 = pd.Timestamp(cfg["twin"]["start"]).tz_convert(IST).tz_localize(None)
    tw1 = tw0 + pd.Timedelta(days=cfg["twin"]["days"])
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True)
    axs[0].plot(t, met.wind_speed_10m, color=INK)
    axs[0].set(ylabel="Wind speed (m/s)", title="ERA5 forcing over Delhi, 10–20 Nov 2024 (reanalysis estimate)")
    axs[1].scatter(t, met.wind_direction_10m, s=6, color=INK)
    axs[1].set(ylabel="Wind from (°)", yticks=[0, 90, 180, 270, 360], ylim=(0, 360))
    axs[2].plot(t, met.blh, color=INK)
    axs[2].axhline(cfg["physics"]["H_min"], color=INK2, lw=1, ls="--")
    axs[2].text(t.iloc[-1], cfg["physics"]["H_min"] + 260, f"floor H_min = {cfg['physics']['H_min']:.0f} m",
                ha="right", fontsize=7.5, color=INK2)
    axs[2].set(ylabel="Boundary-layer\nheight (m)", xlabel="Local time (IST)")
    for ax in axs:
        ax.axvspan(tw0, tw1, color="#f0efec", zorder=0, lw=0)
    axs[2].text(tw0 + (tw1 - tw0) / 2, met.blh.max() * 1.02, "synthetic-twin window",
                ha="center", va="bottom", fontsize=7.5, color=INK2)
    axs[2].set_ylim(0, met.blh.max() * 1.15)
    axs[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(fig, "era5_forcing",
         "Hourly ERA5 reanalysis (via Open-Meteo) at the domain centre: 10 m wind speed, direction the wind blows "
         "from, and boundary-layer height. These ESTIMATED quantities enter the PDE as the spatially uniform "
         "advection velocity (u, v) and the mixing height H(t). Grey band: the 3-day window used for the synthetic twin.")


def fig_pm_series(s):
    obs = s["obs"].copy()
    obs["t"] = to_ist(obs.time)
    obs["hour"] = obs.t.dt.hour
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for _, g in obs.groupby("location_id"):
        axs[0].plot(g.t, g.pm25, color="#c3c2b7", lw=0.6)
    m = obs.groupby("t").pm25.mean()
    axs[0].plot(m.index, m.values, color=COL["PINN"], lw=2, label="mean over stations")
    axs[0].annotate(f"mean over {obs.location_id.nunique()} stations", xy=(m.index[30], m.iloc[30]), xytext=(m.index[18], 520),
                    fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axs[0].set(ylabel="PM2.5 (µg/m³)", xlabel="Local time (IST)", title="Synthetic station series")
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    q = obs.groupby("hour").pm25.quantile([0.25, 0.5, 0.75]).unstack()
    axs[1].fill_between(q.index, q[0.25], q[0.75], color="#cde2fb", lw=0)
    axs[1].plot(q.index, q[0.5], color=COL["PINN"], lw=2)
    axs[1].set(xlabel="Hour of day (IST)", ylabel="PM2.5 (µg/m³)", title="Diurnal cycle", xticks=[0, 6, 12, 18, 23])
    axs[1].text(12, q[0.75].max() * 0.98, "median and\ninterquartile range", ha="center", va="top", fontsize=7.5)
    save(fig, "twin_pm_series",
         "SYNTHETIC hourly PM2.5 sampled at the 45 real monitor locations (grey) and their mean (blue), and the diurnal "
         "median with interquartile range. The night-time maximum and afternoon minimum come from the ERA5 "
         "boundary-layer height entering the source term as E·g/H.")


def fig_training(s):
    h = results(BEST, 42)["histories"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for name in ("DataOnlyNN", "PINN"):
        it = [r["iter"] for r in h[name]]
        axs[0].semilogy(it, [r["data"] for r in h[name]], color=COL[name], label=f"{LABEL[name]}: training data loss")
        axs[0].semilogy(it, [r["val"] for r in h[name]], color=COL[name], ls="--", label=f"{LABEL[name]}: validation MSE")
    axs[0].set(xlabel="Iteration", ylabel="MSE (dimensionless C/C_ref)", title="Data fit (seed 42)")
    axs[0].legend(fontsize=7)
    styles = {"data": "-", "physics": "--", "boundary": ":", "source_reg": "-."}
    names = {"data": "data", "physics": "PDE residual", "boundary": "boundary", "source_reg": "source regulariser"}
    it = [r["iter"] for r in h["PINN"]]
    for k, ls in styles.items():
        y = [r[k] for r in h["PINN"]]
        axs[1].semilogy(it, y, color=INK, ls=ls)
        axs[1].text(it[-1] * 1.02, y[-1], names[k], fontsize=7.5, va="center")
    axs[1].set(xlabel="Iteration", ylabel="Unweighted loss term", title="PINN loss components (w_p = 0.01)",
               xlim=(0, it[-1] * 1.35))
    save(fig, "twin_training",
         "Training history for seed 42. Left: training data loss (solid) and validation-station MSE (dashed) for the "
         "data-only network and the PINN; the data-only network overfits after ~2000 iterations (validation error "
         "rises) whereas the PINN's validation error keeps falling. Right: unweighted PINN loss components. "
         "Validation stations are used only to select the best iterate, never for gradient steps.")


def fig_parity(s):
    p = pd.read_csv(METRICS / f"twin_{BEST}_s42_preds.csv")
    r = results(BEST, 42)["results"]
    fig, axs = plt.subplots(1, 4, figsize=(8.4, 2.5), sharex=True, sharey=True)
    lim = (0, max(p.obs.max(), p[list(COL)].max().max()) * 1.05)
    for ax, name in zip(axs, ("IDW", "RandomForest", "DataOnlyNN", "PINN")):
        ax.plot(lim, lim, color=AXIS, lw=1)
        ax.scatter(p.obs, p[name], s=5, color=COL[name], rasterized=True)
        m = r[name]["vs_obs"]
        ax.set(title=LABEL[name], xlabel="Observed (µg/m³)", xlim=lim, ylim=lim, aspect="equal")
        ax.text(0.04, 0.96, f"RMSE {m['rmse']:.1f}\nMAE {m['mae']:.1f}\nR² {m['r2']:.2f}", transform=ax.transAxes,
                va="top", fontsize=7.5)
    axs[0].set_ylabel("Predicted (µg/m³)")
    save(fig, "twin_parity",
         "Predicted against observed PM2.5 at the five held-out test stations (seed 42, 73 hours each; SYNTHETIC data). "
         "Grey line: perfect agreement.")


def fig_comparison():
    rows = []
    for seed in SEEDS:
        r = results(BEST, seed)["results"]
        for name in COL:
            rows.append(dict(seed=seed, model=name, rmse=r[name]["vs_obs"]["rmse"], mae=r[name]["vs_obs"]["mae"],
                             field=r[name].get("field", {}).get("rmse", np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "twin_seed_summary.csv", index=False)
    order = ["IDW", "RandomForest", "DataOnlyNN", "PINN"]
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.8), gridspec_kw=dict(width_ratios=[4, 4, 2.4]))
    for ax, col, title, models in ((axs[0], "rmse", "Held-out stations: RMSE", order),
                                   (axs[1], "mae", "Held-out stations: MAE", order),
                                   (axs[2], "field", "Whole field: RMSE", ["DataOnlyNN", "PINN"])):
        g = df.groupby("model")[col].agg(["mean", "std"]).loc[models]
        xs = np.arange(len(models))
        ax.bar(xs, g["mean"], width=0.62, color=[COL[m] for m in models], edgecolor="white", linewidth=2)
        ax.errorbar(xs, g["mean"], yerr=g["std"], fmt="none", ecolor=INK, capsize=3, lw=1)
        for x, m in zip(xs, models):
            ax.scatter(np.full(4, x) + np.linspace(-0.12, 0.12, 4), df[df.model == m][col], s=9, color=INK, zorder=3)
            top = max(g.loc[m, "mean"] + g.loc[m, "std"], df[df.model == m][col].max())
            ax.text(x, top * 1.03 + 0.5, f"{g.loc[m, 'mean']:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0, df[df.model.isin(models)][col].max() * 1.18)
        ax.set_xticks(xs, [LABEL[m].replace(" ", "\n", 1) for m in models], fontsize=7.5)
        ax.set(title=title, ylabel="µg/m³")
        ax.grid(axis="x", visible=False)
    save(fig, "twin_comparison",
         "Model comparison on the SYNTHETIC twin: mean ± one standard deviation over four station splits "
         "(seeds 1, 2, 3, 42; dots are individual splits). Whole-field RMSE is computed against the known truth "
         "on the 1 km solver grid every 3 h, which is possible only in the synthetic experiment.")
    return df


def fig_error_dist():
    errs = {m: [] for m in COL}
    for seed in SEEDS:
        p = pd.read_csv(METRICS / f"twin_{BEST}_s{seed}_preds.csv")
        for m in COL:
            errs[m].append(p[m] - p.obs)
    order = ["IDW", "RandomForest", "DataOnlyNN", "PINN"]
    data = [np.concatenate(errs[m]) for m in order]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, m in zip(bp["boxes"], order):
        patch.set(facecolor=COL[m], edgecolor=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(1, 5), [LABEL[m] for m in order])
    ax.set(xlabel="Prediction error, predicted − observed (µg/m³)",
           title="Error distribution at held-out stations (4 splits pooled)")
    ax.grid(axis="y", visible=False)
    save(fig, "twin_error_distribution",
         "Distribution of prediction errors at held-out stations, pooled over four splits (boxes: interquartile range; "
         "whiskers: 1.5 IQR; outliers not drawn). SYNTHETIC data.")


def fit_source_posthoc(model, n_iters=2000, w_s=1e-3, seed=0):
    """Best case for the data-only NN: freeze its concentration field and fit the emission
    model and D to minimise the PDE residual (same regulariser as the PINN)."""
    from src.training.losses import source_regulariser
    for p in model.net.parameters():
        p.requires_grad_(False)
    params = list(model.source.parameters()) + [model.D_raw]
    opt = torch.optim.Adam(params, lr=1e-2)
    for it in range(n_iters):
        if it % 250 == 0:
            Xc = sample_collocation(4096, seed=seed + it).requires_grad_(True)
        loss = torch.mean(model.residual(Xc) ** 2) + w_s * source_regulariser(model, Xc)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def fig_residual(s):
    models = load_models(s, BEST, 42)
    fitted = fit_source_posthoc(load_models(s, BEST, 42)["DataOnlyNN"])
    X = sample_collocation(20000, seed=123).requires_grad_(True)
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    bins = np.logspace(-4, 2.5, 60)
    meds = {}
    cases = (("PINN", models["PINN"], COL["PINN"], "-"),
             ("Data-only NN, source and D fitted afterwards", fitted, COL["DataOnlyNN"], "-"),
             ("Data-only NN, untrained source", models["DataOnlyNN"], COL["DataOnlyNN"], ":"))
    for label, m, c, ls in cases:
        r = m.residual(X).detach().abs().numpy()
        meds[label] = float(np.median(r))
        ax.hist(r, bins=bins, histtype="step", color=c, lw=1.8, ls=ls, label=f"{label} (median {meds[label]:.2f})")
    ax.set_xscale("log")
    ax.set(xlabel="|PDE residual| (dimensionless)", ylabel="Collocation points",
           title="Physical consistency: PDE residual at 20 000 random points")
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, -0.22), ncol=1)
    a, b = meds["Data-only NN, source and D fitted afterwards"], meds["PINN"]
    save(fig, "twin_residual",
         "Absolute dimensionless PDE residual at 20 000 Latin-hypercube points (seed 42; SYNTHETIC). To compare "
         "fairly, the data-only network's field is frozen and the emission model and D are then fitted to it "
         f"(2000 Adam steps, same regulariser): even in this best case its median residual ({a:.2f}) is "
         f"{a / b:.1f}× the PINN's ({b:.2f}). No admissible source makes the data-only field consistent with "
         "transport by the observed wind; the dotted curve shows the same network with the untrained source.")
    return meds


def fig_wp_sweep():
    tags = {0.003: "wp0.003_it6000", 0.01: "wp0.01_it6000", 0.03: "wp0.03_it6000", 0.1: "wp0.1_it6000", 1.0: "wp1_it6000"}
    rows = []
    for wp, tag in tags.items():
        r = results(tag, 42)["results"]
        rows.append(dict(wp=wp, rmse=r["PINN"]["vs_obs"]["rmse"], field=r["PINN"]["field"]["rmse"],
                         nn_rmse=r["DataOnlyNN"]["vs_obs"]["rmse"], nn_field=r["DataOnlyNN"]["field"]["rmse"]))
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
    for ax, col, nn, title in ((axs[0], "rmse", "nn_rmse", "Held-out stations: RMSE"),
                               (axs[1], "field", "nn_field", "Whole field: RMSE")):
        ax.plot(df.wp, df[col], color=COL["PINN"], marker="o", ms=6, mec="white", mew=1.5)
        ref = df[nn].mean()
        ax.axhline(ref, color=COL["DataOnlyNN"], lw=1.4, ls="--")
        ax.text(df.wp.max(), ref - 2, "data-only NN (w_p = 0)", ha="right", va="top", fontsize=7.5)
        ax.text(df.wp.iloc[3] * 0.8, df[col].iloc[3] + 2, "PINN", ha="right", va="bottom", fontsize=7.5, color=INK)
        ax.set(xscale="log", xlabel="Physics-loss weight w_p", ylabel="µg/m³", title=title)
    save(fig, "twin_wp_sweep",
         "Sensitivity of the PINN to the physics-loss weight (seed 42, 6000 Adam iterations; SYNTHETIC data). "
         "Small weights (0.003–0.01) improve on the data-only network; at w_p ≥ 0.1 the PDE term dominates the "
         "gradient, the optimiser under-fits the data, and accuracy collapses.")
    return df


def fig_fields(s):
    tw, sc = s["truth"], s["sc"]
    models = load_models(s, BEST, 42)
    var = tw["C"].reshape(len(tw["t_rel"]), -1).std(axis=1)
    k = int(np.argmax(var))
    t = tw["t_rel"][k]
    gx, gy = np.meshgrid(tw["x"] / 1e3, tw["y"] / 1e3, indexing="ij")
    pred = {n: predict(m, sc, gx.ravel(), gy.ravel(), np.full(gx.size, t)).reshape(gx.shape) for n, m in models.items()}
    truth = tw["C"][k]
    vmax = np.percentile(truth, 99.5)
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.4), sharex=True, sharey=True)
    st = s["obs"].drop_duplicates("location_id")
    tr = st[st.location_id.isin(s["split"]["train"])]
    te = st[st.location_id.isin(s["split"]["test"])]
    panels = [("Truth (solver)", truth), ("Data-only NN", pred["DataOnlyNN"]), ("PINN", pred["PINN"])]
    for ax, (title, F) in zip(axs[0], panels):
        im = ax.pcolormesh(gx, gy, F, cmap=SEQ, vmin=0, vmax=vmax, shading="auto", rasterized=True)
        ax.set(title=title, aspect="equal")
    snap = {}
    for ax, (title, F) in zip(axs[1, 1:], panels[1:]):
        im2 = ax.pcolormesh(gx, gy, np.abs(F - truth), cmap="Oranges", vmin=0, vmax=vmax / 2, shading="auto", rasterized=True)
        rm = np.sqrt(np.mean((F - truth) ** 2))
        snap[title] = rm
        ax.set(title=f"{title}: |error|\nRMSE {rm:.0f} µg/m³", aspect="equal")
    # Field RMSE at every hour (bottom-left): the snapshot alone would be anecdotal.
    ax = axs[1, 0]
    ax.remove()
    ax = fig.add_subplot(2, 3, 4)
    hours_t = pd.Timestamp(s["meta"]["start"]).tz_convert(IST).tz_localize(None) + pd.to_timedelta(tw["t_rel"], unit="s")
    hourly = {}
    for name, m in models.items():
        rm = []
        for kk, tt in enumerate(tw["t_rel"]):
            F = predict(m, sc, gx.ravel(), gy.ravel(), np.full(gx.size, tt))
            rm.append(np.sqrt(np.mean((F - tw["C"][kk].ravel()) ** 2)))
        hourly[name] = np.array(rm)
        ax.plot(hours_t, rm, color=COL[name], lw=1.4, label=LABEL[name])
    ax.legend(loc="upper left", fontsize=7)
    ax.axvline(hours_t[k], color=INK2, lw=1, ls=":")
    ax.set(ylabel="Field RMSE (µg/m³)", title="Field error, every hour")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_aspect("auto")
    for ax in list(axs[0]) + list(axs[1, 1:]):
        ax.scatter(tr.x, tr.y, s=8, c=INK, linewidths=0)
        ax.scatter(te.x, te.y, s=26, marker="^", c="white", edgecolors=INK, linewidths=0.9)
        ax.grid(False)
    c1 = fig.colorbar(im, ax=axs[0], shrink=0.8)
    c1.set_label("PM2.5 (µg/m³)")
    c2 = fig.colorbar(im2, ax=axs[1, 1:], shrink=0.8)
    c2.set_label("Absolute error (µg/m³)")
    when = (sc.t_start + pd.Timedelta(seconds=float(t))).tz_convert(IST)
    fig.suptitle(f"Snapshot {when:%d %b %Y %H:%M} IST (hour of largest spatial variance) · x, y in km · "
                 "● training station  △ held-out station", fontsize=8.5, color=INK2, y=0.995)
    save(fig, "twin_fields",
         f"Top: solver truth and the two network reconstructions at {when:%d %b %H:%M} IST, the hour of largest "
         "spatial variance, chosen by that rule rather than by result (seed 42; SYNTHETIC). Bottom right: absolute "
         f"error (snapshot RMSE: data-only NN {snap['Data-only NN']:.0f}, PINN {snap['PINN']:.0f} µg/m³). Bottom left: "
         "whole-field RMSE at every hour (dotted line marks the snapshot); the PINN has the lower field error in "
         f"{(hourly['PINN'] < hourly['DataOnlyNN']).sum()} of {len(hourly['PINN'])} hours. Each panel is a model "
         "evaluated on the grid; no station data are interpolated.")


def fig_identifiability(s):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.6]))
    tw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    ph = tw["physics"]
    Ds = [results(BEST, sd)["results"]["PINN"]["D_learned"] for sd in SEEDS]
    axs[0].scatter(np.arange(4), Ds, s=40, color=COL["PINN"], edgecolors="white", linewidths=1.5, zorder=3)
    axs[0].axhline(tw["twin"]["D_true"], color=INK, lw=1.2)
    axs[0].text(1.5, tw["twin"]["D_true"] + 12, "true D = initial value", ha="center", fontsize=7.5)
    axs[0].text(1.5, np.mean(Ds) + 25, "learned D", ha="center", fontsize=7.5, color=INK)
    axs[0].set(xticks=range(4), xticklabels=[f"split {sd}" for sd in SEEDS], ylabel="D (m²/s)",
               title="Learned eddy diffusivity", ylim=(0, 420))
    hours = np.arange(24)
    axs[1].plot(hours, diurnal_profile(hours), color=INK, lw=2)
    axs[1].annotate("true g", xy=(14, diurnal_profile(14)), xytext=(14, 0.45), ha="center", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    for sd in SEEDS:
        s2 = setup(sd)
        m = load_models(s2, BEST, sd)["PINN"]
        axs[1].plot(hours, m.source.g_nodes().detach().numpy(), color=COL["PINN"], lw=1.2)
    axs[1].annotate("learned g (4 splits)", xy=(15, 0.88), xytext=(15.5, 1.25), ha="center", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axs[1].set_ylim(0.4, 1.8)
    axs[1].set(xlabel="Hour of day (IST)", ylabel="Relative emission", title="Diurnal emission profile",
               xticks=[0, 6, 12, 18, 23])
    save(fig, "twin_identifiability",
         "Parameter recovery on the SYNTHETIC twin. Left: although D was initialised at its true value, every split "
         f"drives it to {min(Ds):.0f}–{max(Ds):.0f} m²/s, far below the true 300 m²/s, as expected for an advection-dominated "
         "flow (Pe ≈ 10²–10³) in which diffusion barely affects the data. Right: learned versus true diurnal emission profile.")
    return Ds


def fig_sensitivity(cfg, s):
    st = s["obs"].drop_duplicates("location_id")
    base = cfg["twin"]
    out = []
    for p, values in (("lam", base["lam_true"] * np.array([0.25, 0.5, 1, 2, 4])),
                      ("D", base["D_true"] * np.array([0.25, 0.5, 1, 2, 4])),
                      ("H_min", np.array([25.0, 50.0, 100.0, 200.0]))):
        for val in values:
            grid, t_rel, F, *_ = simulate(cfg, **{p: float(val)})
            c = np.array([bilinear_sample(grid, f, st.x * 1e3, st.y * 1e3) for f in F])
            out.append(dict(param=p, value=float(val), mean=c.mean(), p95=np.percentile(c, 95),
                            spatial_sd=c.std(axis=1).mean()))
    df = pd.DataFrame(out)
    df.to_csv(METRICS / "twin_forward_sensitivity.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.7), sharey=True)
    ref = {"lam": base["lam_true"], "D": base["D_true"], "H_min": cfg["physics"]["H_min"]}
    names = {"lam": "Removal rate λ (s⁻¹)", "D": "Eddy diffusivity D (m²/s)", "H_min": "Mixing-height floor H_min (m)"}
    for ax, p in zip(axs, ("lam", "D", "H_min")):
        d = df[df.param == p]
        b = d[np.isclose(d.value, ref[p])].iloc[0]
        ax.plot(d.value, d["mean"] / b["mean"], color=INK, marker="o", ms=5, label="station-mean C")
        ax.plot(d.value, d["spatial_sd"] / b["spatial_sd"], color=INK2, ls="--", marker="s", ms=4, label="spatial s.d.")
        ax.axvline(ref[p], color=AXIS, lw=1)
        ax.set(xscale="log", xlabel=names[p], title=f"Sensitivity to {p.replace('lam', 'λ')}")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(d.value.to_numpy()[::2] if p != "H_min" else d.value.to_numpy())
        ax.set_xticklabels([f"{v:.0f}" if v >= 1 else f"{v:.1e}" for v in (d.value.to_numpy()[::2] if p != "H_min" else d.value.to_numpy())])
        lo, hi = d.value.min(), d.value.max()
        el = np.log(d["mean"].iloc[-1] / d["mean"].iloc[0]) / np.log(hi / lo)
        ax.text(0.04, 0.06, f"elasticity of mean: {el:+.2f}", transform=ax.transAxes, fontsize=7.5)
    axs[0].set_ylabel("Relative to baseline")
    axs[0].legend(loc="upper right", fontsize=7)
    save(fig, "twin_forward_sensitivity",
         "Forward-model sensitivity: the reference solver re-run with one parameter varied (others at baseline), "
         "sampled at the station locations. Values are relative to the baseline run; elasticity is "
         "d ln C̄ / d ln p over the range shown. λ and D have weak influence on station concentrations over a "
         "3-day window, consistent with the Damköhler and Péclet numbers; the mixing-height floor matters more.")
    return df


def fig_station_series(s):
    p = pd.read_csv(METRICS / f"twin_{BEST}_s42_preds.csv")
    sid = p.groupby("location_id").truth.std().idxmax()
    d = p[p.location_id == sid].sort_values("t_rel")
    t = pd.Timestamp(s["meta"]["start"]).tz_convert(IST).tz_localize(None) + pd.to_timedelta(d.t_rel, unit="s")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.scatter(t, d.obs, s=8, color=INK2, label="noisy observation", zorder=3)
    ax.plot(t, d.truth, color=INK, lw=1.2, label="truth")
    for m in ("IDW", "DataOnlyNN", "PINN"):
        ax.plot(t, d[m], color=COL[m], lw=1.6, label=LABEL[m])
    ax.set(ylabel="PM2.5 (µg/m³)", xlabel="Local time (IST)",
           title=f"Held-out station {sid}: reconstruction from the other stations")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7.5)
    save(fig, "twin_station_series",
         f"Time series at held-out station {sid} (seed 42, the test station with the largest variability; "
         "SYNTHETIC data). None of the models saw this station during training.")


def main():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    s = setup(42)
    fig_study_area(s)
    fig_forcing(cfg)
    fig_pm_series(s)
    fig_training(s)
    fig_parity(s)
    summ = fig_comparison()
    fig_error_dist()
    meds = fig_residual(s)
    sweep = fig_wp_sweep()
    fig_fields(s)
    Ds = fig_identifiability(s)
    sens = fig_sensitivity(cfg, s)
    fig_station_series(s)
    with open(PNG / "captions.md", "w") as f:
        for k, v in CAPTIONS.items():
            f.write(f"### {k}\n\n{v}\n\n")
    print(summ.groupby("model")[["rmse", "mae", "field"]].agg(["mean", "std"]).round(1))
    print("residual medians", meds)
    print(sweep.round(1))
    print("D learned", np.round(Ds, 1))
    print(sens.round(2))


if __name__ == "__main__":
    main()
