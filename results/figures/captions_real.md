### real_station_map

OBSERVED: CPCB/DPCC/IMD/IITM regulatory monitors (from OpenAQ) passing the data-quality checks in each window, coloured by window-mean PM2.5 and marked by role in the seed-42 split. Axes are the local tangent plane centred at 28.63°N, 77.15°E. No spatial interpolation is shown.

### real_pm_series

OBSERVED hourly PM2.5. Left: median over monitors with the 10–90 % range; right: diurnal median and interquartile range. Note the different vertical scales: the November 2025 window is a severe post-monsoon episode, the March 2025 window a moderate one.

### real_comparison

OBSERVED: RMSE at held-out monitors, mean ± one standard deviation over three station splits (seeds 1, 2, 42; dots are individual splits). The physics weight w_p = 0.01 was fixed from the synthetic twin before any real-data run. 'Station mean' predicts the hourly mean of the training monitors everywhere.

### real_parity

OBSERVED: predicted against measured hourly PM2.5 at the held-out monitors (seed 42). Grey: 1:1 line.

### real_station_series

OBSERVED: hourly PM2.5 at one held-out monitor per window (the test monitor with most valid hours, seed 42) and the reconstructions from the training monitors. The networks are trained on consecutive 80-hour chunks.

### real_wp_sweep

OBSERVED: sensitivity of the PINN to the physics weight (seed 42). Labels give the validation-station MSE (dimensionless, mean over chunks) that a practitioner without access to the test monitors would see. The reported configuration (w_p = 0.01) was fixed in advance from the synthetic twin.

### real_residual

OBSERVED-data models: PDE residual at 20 000 random points in the first 80-hour chunk (seed 42). For a fair comparison the data-only network's field is frozen and its emission model and D are fitted to it afterwards.

### real_wind_check

Check of the spatially uniform wind assumption: hourly wind speed reported by the monitoring stations (OBSERVED; median and interquartile range across stations) against the single ERA5 value used in the PDE. Winter window only: the March 2025 archive files contain no station wind data. Window means: monitors 0.5 m/s, ERA5 1.4 m/s; correlation of hourly values r = 0.50. Station anemometers sit within the urban canopy, so a lower reading than the 10 m reanalysis is expected; the timing agreement is what supports using ERA5 as the transport wind.

