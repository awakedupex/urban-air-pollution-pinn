"""Training loop shared by the PINN and the data-only network (w_p = w_b = w_i = w_s = 0)."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.training.losses import boundary_loss, data_loss, initial_loss, physics_loss, source_regulariser
from src.training.sampling import sample_boundary, sample_collocation


@dataclass
class TrainConfig:
    n_iters: int = 5000
    lr: float = 1e-3
    lr_final: float = 1e-4  # exponential decay from lr to lr_final
    w_d: float = 1.0
    w_p: float = 1.0
    w_b: float = 0.1
    w_i: float = 0.0
    w_s: float = 1e-3
    n_colloc: int = 4096
    n_bc: int = 512
    resample_every: int = 250
    log_every: int = 100
    seed: int = 0
    # Adaptive physics weight (gradient-norm balancing, after Wang, Teng & Perdikaris 2021):
    # w_p <- (1 - a) w_p + a * w_p0 * mean|grad L_data| / mean|grad L_phys|, over the network weights.
    adaptive: bool = False
    adapt_every: int = 100
    adapt_alpha: float = 0.1


def train(model, X_obs, c_obs, cfg: TrainConfig, val=None, ic=None, c_bc=None):
    """Minimise the weighted loss with Adam.

    X_obs, c_obs: dimensionless training observations. val: optional (X, c) used to
    keep the best weights (early-stopping selection, never for gradient steps).
    ic: optional (X0, c0) initial field. c_bc: optional inflow value (dimensionless).
    Returns the logged history; the model is left at the best-validation weights.
    """
    torch.manual_seed(cfg.seed)
    physics = cfg.w_p > 0 or cfg.w_b > 0 or cfg.w_i > 0 or cfg.w_s > 0
    params = model.parameters() if physics else model.net.parameters()
    opt = torch.optim.Adam(params, lr=cfg.lr)
    gamma = (cfg.lr_final / cfg.lr) ** (1 / max(cfg.n_iters, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)

    history, best = [], (np.inf, None)
    w_p = cfg.w_p
    shared = list(model.net.parameters())
    for it in range(cfg.n_iters + 1):
        if it % cfg.resample_every == 0:
            Xc = sample_collocation(cfg.n_colloc, seed=cfg.seed + it).requires_grad_(True)
            Xb, nb = sample_boundary(cfg.n_bc, seed=cfg.seed + it)
            Xb.requires_grad_(True)

        parts = {"data": data_loss(model, X_obs, c_obs)}
        if cfg.w_p > 0:
            parts["physics"] = physics_loss(model, Xc)
        if cfg.w_b > 0:
            parts["boundary"] = boundary_loss(model, Xb, nb, c_bc)
        if cfg.w_i > 0 and ic is not None:
            parts["initial"] = initial_loss(model, *ic)
        if cfg.w_s > 0:
            parts["source_reg"] = source_regulariser(model, Xc)
        if cfg.adaptive and "physics" in parts and it > 0 and it % cfg.adapt_every == 0:
            g_d = _flat_grad(parts["data"], shared)
            g_p = _flat_grad(parts["physics"], shared)
            w_hat = cfg.w_p * g_d.abs().mean() / (g_p.abs().mean() + 1e-12)
            w_p = (1 - cfg.adapt_alpha) * w_p + cfg.adapt_alpha * w_hat.item()
        w = {"data": cfg.w_d, "physics": w_p, "boundary": cfg.w_b, "initial": cfg.w_i, "source_reg": cfg.w_s}
        total = sum(w[k] * v for k, v in parts.items())

        if it % cfg.log_every == 0 or it == cfg.n_iters:
            rec = {"iter": it, "total": total.item(), "w_p": w_p, **{k: v.item() for k, v in parts.items()},
                   "D": model.D().item() if hasattr(model, "D") else float("nan")}
            if val is not None:
                with torch.no_grad():
                    rec["val"] = torch.mean((model(val[0]) - val[1]) ** 2).item()
                if rec["val"] < best[0]:
                    best = (rec["val"], copy.deepcopy(model.state_dict()))
            history.append(rec)
        if it == cfg.n_iters:
            break
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

    if best[1] is not None:
        model.load_state_dict(best[1])
    return history


def _flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
