### twin_study_area

Model domain (local tangent plane centred at 28.63°N, 77.15°E, covering the Delhi NCT bounding box). Shading: the SYNTHETIC emission flux used to generate the twin (background plus three illustrative hotspots, not an inventory). Markers: the 45 real CPCB/DPCC/IMD monitor locations that pass data-quality checks in the winter window, and their train/validation/test split (seed 42). Arrow: window-mean ERA5 10 m wind.

### era5_forcing

Hourly ERA5 reanalysis (via Open-Meteo) at the domain centre: 10 m wind speed, direction the wind blows from, and boundary-layer height. These ESTIMATED quantities enter the PDE as the spatially uniform advection velocity (u, v) and the mixing height H(t). Grey band: the 3-day window used for the synthetic twin.

### twin_pm_series

SYNTHETIC hourly PM2.5 sampled at the 45 real monitor locations (grey) and their mean (blue), and the diurnal median with interquartile range. The night-time maximum and afternoon minimum come from the ERA5 boundary-layer height entering the source term as E·g/H.

### twin_training

Training history for seed 42. Left: training data loss (solid) and validation-station MSE (dashed) for the data-only network and the PINN; the data-only network overfits after ~2000 iterations (validation error rises) whereas the PINN's validation error keeps falling. Right: unweighted PINN loss components. Validation stations are used only to select the best iterate, never for gradient steps.

### twin_parity

Predicted against observed PM2.5 at the five held-out test stations (seed 42, 73 hours each; SYNTHETIC data). Grey line: perfect agreement.

### twin_comparison

Model comparison on the SYNTHETIC twin: mean ± one standard deviation over four station splits (seeds 1, 2, 3, 42; dots are individual splits). Whole-field RMSE is computed against the known truth on the 1 km solver grid every 3 h, which is possible only in the synthetic experiment.

### twin_error_distribution

Distribution of prediction errors at held-out stations, pooled over four splits (boxes: interquartile range; whiskers: 1.5 IQR; outliers not drawn). SYNTHETIC data.

### twin_residual

Absolute dimensionless PDE residual at 20 000 Latin-hypercube points (seed 42; SYNTHETIC). To compare fairly, the data-only network's field is frozen and the emission model and D are then fitted to it (2000 Adam steps, same regulariser): even in this best case its median residual (1.35) is 3.8× the PINN's (0.35). No admissible source makes the data-only field consistent with transport by the observed wind; the dotted curve shows the same network with the untrained source.

### twin_wp_sweep

Sensitivity of the PINN to the physics-loss weight (seed 42, 6000 Adam iterations; SYNTHETIC data). Small weights (0.003–0.01) improve on the data-only network; at w_p ≥ 0.1 the PDE term dominates the gradient, the optimiser under-fits the data, and accuracy collapses.

### twin_fields

Top: solver truth and the two network reconstructions at 14 Nov 13:30 IST, the hour of largest spatial variance, chosen by that rule rather than by result (seed 42; SYNTHETIC). Bottom right: absolute error (snapshot RMSE: data-only NN 53, PINN 30 µg/m³). Bottom left: whole-field RMSE at every hour (dotted line marks the snapshot); the PINN has the lower field error in 72 of 73 hours. Each panel is a model evaluated on the grid; no station data are interpolated.

### twin_identifiability

Parameter recovery on the SYNTHETIC twin. Left: although D was initialised at its true value, every split drives it to 58–60 m²/s, far below the true 300 m²/s, as expected for an advection-dominated flow (Pe ≈ 10²–10³) in which diffusion barely affects the data. Right: learned versus true diurnal emission profile.

### twin_forward_sensitivity

Forward-model sensitivity: the reference solver re-run with one parameter varied (others at baseline), sampled at the station locations. Values are relative to the baseline run; elasticity is d ln C̄ / d ln p over the range shown. λ and D have weak influence on station concentrations over a 3-day window, consistent with the Damköhler and Péclet numbers; the mixing-height floor matters more.

### twin_station_series

Time series at held-out station 5634 (seed 42, the test station with the largest variability; SYNTHETIC data). None of the models saw this station during training.

