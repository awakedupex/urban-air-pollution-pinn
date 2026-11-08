# Likely viva questions, with answers

Answers use only the project's own results. Report sections are cited as §n.

## Data (presenter 1)

**Why Delhi?**
Data density, not popularity. It has one of the densest regulatory networks in India: 45 usable monitors in a ~50 km box. A spatial model needs that.

**Where do the data come from, and can you trust them?**
CPCB, DPCC, IMD and IITM regulatory monitors, republished by OpenAQ in a public archive; low-cost sensors were excluded. The weather is ERA5 reanalysis via Open-Meteo. Every quantity is labelled observed, estimated, calibrated or synthetic (Table 1).

**What did cleaning remove?**
- `-999` sentinels, negative values, values above 2000 µg/m³ and wrong units.
- IST timestamps converted to UTC; 15-minute readings averaged to hourly means, which also merges duplicates.
- Stuck sensors (6 or more identical hours).
- Stations with less than 75 % coverage.

Every rule has a unit test.

**Why November 2025 and not 2024?**
The archive holds no regulatory files for November 2024 (1 of 55 monitors). We moved to the same dates a year later. This was a data constraint, decided before any model was run.

**How do you avoid train/test leakage?**
- We split stations, not hours.
- Stations closer than 1 km are kept in the same subset.
- Leakage is asserted in code, and every model uses the same split.
- Validation stations only pick the best iterate; they never contribute gradients.

## Model (presenter 2)

**Derive the equation.**
Mass conservation in a control volume: ∂C/∂t + ∇·(uC) = −∇·F − λC + S. Take the wind uniform in space (so ∇·(uC) = u·∇C) and close the turbulent flux with K-theory, F = −D∇C. Average over a well-mixed layer of depth H: an emission flux q·g injected into that layer is a volume source q·g/H.

**Why 2D and not 3D?**
We have no vertical measurements. Assumption A1 (a well-mixed boundary layer) turns the surface concentration into the layer average.

**Why not a Gaussian plume?**
It is the steady-state, point-source solution of this same equation. It needs known stack locations and emission rates, which we don't have for city-wide PM2.5.

**Why divide the source by H(t)?**
The emission is a flux per unit area (µg m⁻² s⁻¹). Spread over a layer of depth H, it becomes a volume rate (µg m⁻³ s⁻¹). This is how the night-time collapse of the boundary layer produces the night-time peaks.

**Why floor H at 50 m?**
ERA5 reports 10–25 m on calm nights, which would make the source term blow up. 50 m is a typical lower bound for a nocturnal urban layer. It's the most sensitive choice we made (elasticity −0.35), so we list it as a limitation.

**What do Pe and Da tell you?**
- Pe ≈ 75–750: the wind dominates diffusion, so D is hard to identify.
- Da ≈ 0.17: removal is slow during one crossing of the city.
- Crossing time 7–11 h: hourly data can see transport between stations.

**Why non-dimensionalise?**
It puts the inputs in [−1, 1] and the coefficients at O(10⁻²)–O(10). Without it, the residual terms differ by orders of magnitude and training fails.

## Analysis and numerics (presenter 3)

**Prove stability.**
Substitute C = ĉ·exp(ik·x + σt). This gives σ = −iu·k − D|k|² − λ, so Re σ = −(D|k|² + λ) < 0 for every k. Diffusion damps short waves fastest, and removal damps all waves. Advection only changes the phase.

**What is the equilibrium?**
For a uniform constant source, dC/dt = S₀ − λC, so C → C* = S₀/λ. It is globally asymptotically stable, with e-folding time 1/λ ≈ 56 h. Periodic forcing through g(τ)/H(t) gives a daily cycle instead.

**Is mass conserved? Can C go negative?**
- Integrating over the domain: total mass changes only through sources, removal and the boundary flux. The solver conserves mass to within 10⁻³ in a test.
- The maximum principle gives C ≥ 0. The PINN enforces C ≥ 0 by construction with softplus.

**How do you know the PINN computes the right derivatives?**
- Autograd matches analytic derivatives to 4 × 10⁻¹⁵.
- The exact Gaussian-puff solution has a residual of 3 × 10⁻¹⁶.
- A test inside the PINN replaces the network with the puff and confirms the scaling and wind wiring give zero residual.

**Describe the reference solver and its stability limit.**
- Method of lines: third-order upwind-biased advection, central diffusion, SSP-RK3 in time.
- Von Neumann analysis: diffusion needs D·Δt·(Δx⁻² + Δy⁻²) ≲ 0.63 (we use 0.25); advection Courant number ≤ 0.5.
- It converges to the exact puff at observed order 2.6–3.7.

**Which parameters can the data identify?**
- The source must be restricted: an arbitrary S(x,y,t) could absorb any residual. Ours is a smooth spatial map times a 24-value daily profile divided by H.
- D is weakly identifiable (Pe ≫ 1, elasticity −0.03). On the twin, D started at the true 300 m²/s and still drifted to about 60. The daily profile is recovered well.

## PINN (presenter 4)

**Why tanh rather than ReLU?**
ReLU has zero second derivative almost everywhere, so the diffusion term ∇²Ĉ would vanish.

**Why isn't the wind a network input?**
If C depended on wind(t) as an input, the network's ∂/∂t would no longer be the PDE's partial derivative. The weather enters only through the PDE coefficients.

**What are the loss terms?**
- Data misfit at the training stations.
- Mean squared PDE residual at 4096 Latin-hypercube points, resampled every 250 steps.
- A zero-normal-gradient condition on outflow faces (weight 0.1).
- A smoothness penalty on q and g (weight 10⁻³).

There is no initial-condition loss on real data: the initial field is known only at stations, which the data loss already covers.

**Why no inflow boundary condition?**
The inflow concentration (regional background) is unknown. Imposing a made-up value would be fabrication. This is limitation A6.

**How did you choose w_p? Isn't that tuning on the test set?**
- At w_p = 1 the PDE term starts about 20× larger than the data term and the network under-fits (gradient imbalance, Wang et al. 2021).
- We swept w_p on the synthetic twin only, chose 0.01, and fixed it before any real-data run.
- On real data, w_p from 0.003 to 0.03 changes the RMSE by a few µg/m³ and doesn't change the conclusion.

**Did adaptive weighting help?**
We implemented gradient-norm balancing. Starting from w_p = 1 it adapted too slowly to compete, so we report fixed weights.

**Why train in 80-hour chunks?**
Over 10 days the advection coefficient grows and the network would have to represent ten daily cycles. Both are known PINN failure modes (Krishnapriyan et al. 2021). Chunking keeps the problem comparable to the 73-hour twin.

**What exactly is the synthetic twin, and isn't it circular?**
- Our solver, run with known D, λ and emissions under the real ERA5 wind and mixing height, sampled at the 45 real monitor sites with 5 % noise.
- It shares the PINN's model structure, so it's a best case. That's deliberate: it tests whether the method works when its assumptions hold, and lets us score the whole field.
- We say this explicitly in the report.

**How big is the improvement on the twin, and is it robust?**
- Station RMSE 29.9 → 22.1 (−26 %); field RMSE 54.5 → 31.3 (−43 %), over 4 splits.
- The PINN wins on the field in 4 of 4 splits and at the stations in 3 of 4.
- An earlier random 36-station layout gave the same ordering.

## Real results and limitations (presenter 5)

**So does the PINN work on real data?**
- It beats the identical network without physics in all 6 runs.
- Its fields satisfy the PDE 3.7–8× better.
- It does not beat simple interpolation: in winter it ties IDW and the station mean (83.3 vs 84.4 vs 84.8), and in spring the station mean is best. It is the best model in only 1 of 6 runs.

**Why is the training-station mean so hard to beat?**
Stations move together: the median inter-station correlation is about 0.7. A spatially uniform estimate already explains most of what happens at an unseen station.

**Then why bother with physics?**
- The data-only network is unstable on real data (winter RMSE 65–182 across splits); the PINN is not.
- The PINN's fields are physically consistent.
- The twin shows the gain is real when the physics is right, so the remaining gap points at specific missing physics.

**What would you change first?**
1. A regional boundary condition, for inflow during the post-monsoon season.
2. A mixing-height entrainment term, for morning dilution.
3. Spatially varying or station-informed wind.

**Is the uniform-wind assumption reasonable?**
Partly. The monitors' anemometers correlate with ERA5 at r = 0.50 and read lower (0.55 vs 1.38 m/s), as expected inside the urban canopy. The timing agrees moderately, but the transporting wind is uncertain. (March 2025 has no station wind data, so this check is winter only.)

**How significant are the differences?**
With 3 splits and 6–8 test stations per window, differences of a few µg/m³ are not significant; the split-to-split spread is 4–25 µg/m³. That's why we report mean ± s.d. and individual splits.

**Is the work reproducible?**
- All seeds and settings are in `config/config.yaml`; 50 pytest tests pass.
- Download, cleaning, experiments and figures are one command each.
- The notebook runs top to bottom in about a minute, loading the full saved results.
- One caveat: repeat runs on different CPU thread counts differ in the last digit.
