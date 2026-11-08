"""Non-physics baselines for spatial reconstruction at held-out stations.

Tables use columns: location_id, time (UTC), x, y (km) and, for training, pm25.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def idw_predict(train: pd.DataFrame, target: pd.DataFrame, power: float = 2.0) -> np.ndarray:
    """Inverse-distance weighting at each hour, using the training stations observed that hour.

    C(x0, t) = sum_i w_i C_i(t) / sum_i w_i,  w_i = |x0 - x_i|^-p.
    A target at a station location returns that station's value exactly.
    """
    out = np.full(len(target), np.nan)
    by_time = {t: g for t, g in train.groupby("time")}
    for k, (t, x, y) in enumerate(zip(target["time"], target["x"], target["y"])):
        g = by_time.get(t)
        if g is None:
            continue
        d = np.hypot(g["x"].to_numpy() - x, g["y"].to_numpy() - y)
        c = g["pm25"].to_numpy()
        if np.any(d < 1e-9):
            out[k] = c[np.argmin(d)]
        else:
            w = d ** -power
            out[k] = np.sum(w * c) / np.sum(w)
    return out


def random_forest_predict(train, target, features, seed=0, n_estimators=300) -> np.ndarray:
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    rf.fit(train[features].to_numpy(), train["pm25"].to_numpy())
    return rf.predict(target[features].to_numpy())
