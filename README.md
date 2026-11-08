# Physics-Informed Neural Networks for Urban Air Pollution Modelling

A Mathematical Modelling group project. A 2D advection–diffusion–reaction PINN models mixing-layer-averaged PM2.5 over Delhi. It is checked first on a synthetic twin with a known truth, then on real CPCB/DPCC monitor data with entire stations held out.

## Governing equation

$$
C_t + u(t)\,C_x + v(t)\,C_y = D\,\nabla^2 C - \lambda C + \frac{q(x,y)\,g(\tau)}{\max(H(t),H_{\min})}
$$

| Symbol | Meaning | Kind |
|---|---|---|
| $u, v$ | Wind components | Estimated: ERA5 reanalysis |
| $H$ | Boundary-layer (mixing) height | Estimated: ERA5 reanalysis |
| $\lambda$ | Removal coefficient | Fixed, literature value |
| $D$ | Eddy diffusivity | Learned, within prior bounds |
| $q \ge 0,\ g$ | Emission field and diurnal profile | Learned, regularised |

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest                       # all tests
.venv/bin/python -m src.data.download            # ERA5 weather (+ OpenAQ PM2.5 if OPENAQ_API_KEY is set)
.venv/bin/python -m experiments.make_synthetic_twin
.venv/bin/python -m experiments.run_twin --iters 6000 --wp 1 --adaptive
```

The report is compiled with `tectonic` (`brew install tectonic`). Data provenance is documented in [data/README.md](data/README.md), and every configuration value and seed is in [config/config.yaml](config/config.yaml).

## Layout

| Path | Contents |
|---|---|
| `src/data/` | Download, cleaning, projection, station split, synthetic-twin helpers |
| `src/physics/` | ADR residual (autograd), Gaussian puff exact solution, finite-difference reference solver |
| `src/models/` | PINN (concentration net, source model, bounded D), baselines (IDW, random forest) |
| `src/training/` | Losses, sampling, trainer (fixed or adaptive physics weight) |
| `src/evaluation/` | Metrics, leakage guards |
| `experiments/` | Runnable experiments; outputs go to `results/` |
| `tests/` | pytest suite (data, physics, PINN, metrics) |
