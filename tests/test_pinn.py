"""Tests for the PINN model, losses and trainer."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.pinn import PINN, Forcing, Scales
from src.physics.advection_diffusion import gaussian_puff
from src.training.losses import boundary_loss, physics_loss
from src.training.sampling import sample_boundary, sample_collocation
from src.training.trainer import TrainConfig, train

T0 = pd.Timestamp("2024-11-10T00:00", tz="UTC")


def _scales():
    return Scales(x_c=0.0, y_c=0.0, L=20_000.0, t_start=T0, T=43_200.0, C_ref=100.0)


def _forcing(u=2.0, v=-1.0, H=500.0, hours=25):
    t = np.arange(hours) * 3600.0
    return Forcing(t, np.full(hours, u), np.full(hours, v), np.full(hours, H), H_min=50.0, H_ref=500.0)


def _model(seed=0, **kw):
    return PINN(_scales(), _forcing(**kw), D_min=10.0, D_max=2000.0, D_init=300.0, lam=5e-6, seed=seed)


def _X(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (2 * torch.rand(n, 3, generator=g) - 1).to(torch.get_default_dtype())


def test_forward_shape_and_positive():
    m = _model()
    c = m(_X())
    assert c.shape == (64,)
    assert torch.all(c > 0)


def test_seeded_init_reproducible():
    a, b = _model(seed=3), _model(seed=3)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_D_within_bounds():
    m = _model()
    for raw in (-100.0, 0.0, 100.0):
        with torch.no_grad():
            m.D_raw.fill_(raw)
        assert 10.0 <= m.D().item() <= 2000.0
    m2 = _model()
    assert m2.D().item() == pytest.approx(300.0, rel=1e-4)


def test_diurnal_profile_positive_mean_one_and_periodic():
    m = _model()
    with torch.no_grad():
        m.source.g_raw.copy_(torch.randn(24))
    g = m.source.g_nodes()
    assert torch.all(g > 0)
    assert g.mean().item() == pytest.approx(1.0, rel=1e-6)
    hours = torch.tensor([0.0, 5.0, 24.0, 23.5])
    v = m.source.g(hours)
    torch.testing.assert_close(v[:2], g[[0, 5]])
    torch.testing.assert_close(v[2], g[0])  # periodic
    torch.testing.assert_close(v[3], 0.5 * (g[23] + g[0]))


def test_forcing_linear_interpolation():
    t = np.array([0.0, 3600.0, 7200.0])
    f = Forcing(t, np.array([0.0, 2.0, 4.0]), np.zeros(3), np.array([20.0, 100.0, 300.0]), H_min=50.0, H_ref=100.0)
    u, v, h = f.at(torch.tensor([0.0, 1800.0, 7200.0]))
    torch.testing.assert_close(u, torch.tensor([0.0, 1.0, 4.0]))
    torch.testing.assert_close(h, torch.tensor([0.5, 0.75, 3.0]))  # floored at H_min, scaled by H_ref


def test_residual_zero_for_exact_puff():
    """Swap the network for the exact puff (and switch off the source): residual must vanish.

    Checks the coupling of scaling, wind interpolation, D and lambda inside the PINN.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        m = _model()
        sc = m.scales
        D, lam = m.D().item(), m.lam

        class Puff(torch.nn.Module):
            def forward(self, X):
                x = X[:, 0] * sc.L + sc.x_c
                y = X[:, 1] * sc.L + sc.y_c
                t = X[:, 2] * sc.T / 2 + sc.T / 2 + 7200.0  # released 2 h before the window
                return gaussian_puff(x, y, t, M=1e10, x0=-5000.0, y0=3000.0, u=2.0, v=-1.0, D=D, lam=lam) / sc.C_ref

        m.net = Puff()
        m.source.forward = lambda X: torch.zeros(len(X))
        X = _X(256).requires_grad_(True)
        r = m.residual(X)
        assert r.abs().max().item() < 1e-8
    finally:
        torch.set_default_dtype(previous)


def test_boundary_loss_neumann_and_dirichlet():
    m = _model()

    class Const(torch.nn.Module):
        def forward(self, X):
            return 0.0 * X[:, 0] + 0.7

    m.net = Const()
    Xb, n = sample_boundary(200, seed=0)
    Xb.requires_grad_(True)
    assert boundary_loss(m, Xb, n).item() == pytest.approx(0.0, abs=1e-12)
    Xb2, n2 = sample_boundary(200, seed=0)
    val = boundary_loss(m, Xb2.requires_grad_(True), n2, c_bc=0.2).item()
    # u = 2, v = -1: inflow faces are x' = -1 and y' = +1. Each inflow point
    # contributes (0.7 - 0.2)^2, outflow points contribute 0; the mean is over all points.
    inflow = (2.0 * n2[:, 0] - 1.0 * n2[:, 1]) <= 0
    assert val == pytest.approx(0.25 * inflow.double().mean().item(), rel=1e-5)


def test_sampling_ranges():
    Xc = sample_collocation(1000, seed=0)
    assert Xc.shape == (1000, 3) and Xc.min() >= -1 and Xc.max() <= 1
    Xb, n = sample_boundary(400, seed=0)
    on_edge = (Xb[:, :2].abs() - 1).abs().min(dim=1).values < 1e-6
    assert on_edge.all()
    torch.testing.assert_close(n.norm(dim=1), torch.ones(400))


def test_physics_loss_gradients_reach_all_parameters():
    m = _model()
    loss = physics_loss(m, sample_collocation(128, seed=1).requires_grad_(True))
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert m.D_raw.grad.abs().item() > 0


def _toy_data(n=200, seed=0):
    X = _X(n, seed)
    c = 1.0 + 0.5 * torch.sin(np.pi * X[:, 0]) * torch.cos(np.pi * X[:, 2])
    return X, c


def test_training_decreases_loss_and_updates_parameters():
    m = _model()
    before = copy.deepcopy(m.state_dict())
    X, c = _toy_data()
    cfg = TrainConfig(n_iters=300, n_colloc=256, n_bc=64, log_every=50, seed=0, w_p=0.1, w_b=0.1)
    hist = train(m, X, c, cfg)
    assert hist[-1]["data"] < 0.5 * hist[0]["data"]
    assert all(np.isfinite(h["total"]) for h in hist)
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_data_only_mode_leaves_physics_parameters_untouched():
    m = _model()
    D0 = m.D_raw.detach().clone()
    src0 = copy.deepcopy(m.source.state_dict())
    X, c = _toy_data()
    train(m, X, c, TrainConfig(n_iters=50, seed=0, w_p=0.0, w_b=0.0, w_i=0.0, w_s=0.0))
    assert torch.equal(m.D_raw.detach(), D0)
    for k, v in m.source.state_dict().items():
        assert torch.equal(v, src0[k])


def test_adaptive_weight_balances_gradient_norms():
    """After an adaptive update, w_p * mean|grad L_phys| ~= w_p0 * mean|grad L_data| (smoothed)."""
    m = _model()
    X, c = _toy_data()
    cfg = TrainConfig(n_iters=200, n_colloc=128, n_bc=32, log_every=50, seed=0,
                      w_p=1.0, adaptive=True, adapt_every=50, adapt_alpha=1.0)
    hist = train(m, X, c, cfg)
    ws = [h["w_p"] for h in hist]
    assert all(np.isfinite(w) and w > 0 for w in ws)
    assert ws[-1] != ws[0]


def test_training_reproducible():
    X, c = _toy_data()
    cfg = TrainConfig(n_iters=40, n_colloc=128, n_bc=32, log_every=10, seed=7)
    h1 = train(_model(seed=1), X, c, cfg)
    h2 = train(_model(seed=1), X, c, cfg)
    assert [h["total"] for h in h1] == [h["total"] for h in h2]
