"""Tests for synthetic-twin helpers."""
import numpy as np
import pytest

from src.data.synthetic import bilinear_sample, diurnal_profile
from src.physics.solver import Grid


def test_bilinear_exact_for_linear_field():
    g = Grid(x_min=0, x_max=10, y_min=-5, y_max=5, nx=11, ny=21)
    F = 3.0 + 2.0 * g.X - 0.5 * g.Y
    x, y = np.array([0.3, 9.9, 4.25]), np.array([-4.9, 2.2, 0.0])
    np.testing.assert_allclose(bilinear_sample(g, F, x, y), 3.0 + 2.0 * x - 0.5 * y)


def test_bilinear_outside_raises():
    g = Grid(x_min=0, x_max=1, y_min=0, y_max=1, nx=3, ny=3)
    with pytest.raises(ValueError):
        bilinear_sample(g, np.zeros(g.shape), np.array([2.0]), np.array([0.5]))


def test_diurnal_profile_mean_one_positive():
    g = diurnal_profile(np.arange(24))
    assert g.mean() == pytest.approx(1.0) and (g > 0).all()
    assert g[9] > g[3] and g[20] > g[14]  # morning and evening peaks
