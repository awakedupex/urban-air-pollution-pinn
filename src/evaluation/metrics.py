"""Point-wise error metrics (concentrations in µg/m³)."""
from __future__ import annotations

import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def evaluate(y, p) -> dict:
    """MAE, RMSE, R², mean bias (p - y) and Pearson r over pairs where both are finite."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return {
        "n": int(ok.sum()),
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "r2": r2(y, p),
        "bias": float(np.mean(p - y)),
        "pearson_r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan"),
    }
