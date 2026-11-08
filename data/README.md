# Data provenance

Every quantity used in the project is one of four kinds. The report and notebook label each figure accordingly.

| Kind | Quantity | Source |
|---|---|---|
| **Observed** | Hourly PM2.5 (µg/m³) at Delhi monitoring stations | OpenAQ open-data archive (AWS S3 `openaq-data-archive`), republishing CPCB/DPCC regulatory monitors |
| **Estimated (reanalysis)** | 10 m wind speed (m/s) and direction (°), boundary-layer height (m), 2 m temperature, RH, surface pressure | ERA5 via Open-Meteo Historical Weather API (`archive-api.open-meteo.com`) |
| **Calibrated** | Eddy diffusivity D, emission field q(x,y), diurnal profile g | Learned by the PINN within stated bounds |
| **Synthetic** | Synthetic-twin concentration fields | `src/physics/solver.py` with known parameters, sampled at real station locations |

## Reproduce

```bash
echo 'OPENAQ_API_KEY=your-key' > .env    # free key from https://explore.openaq.org (account settings)
.venv/bin/python -m src.data.download
```

This writes:
- `raw/openmeteo/era5_{winter,spring}.csv`
- `raw/openaq/locations.csv` and `raw/openaq/pm_{winter,spring}.csv.gz`
- `raw/manifest.json`, which records the download time, URLs and row counts.

## Windows (config/config.yaml)

- Winter: 2024-11-10 to 2024-11-20 (local time)
- Spring: 2025-03-01 to 2025-03-11 (local time)

## Known raw-data artefacts

These are handled in `src/data/preprocessing.py` and tested in `tests/test_data.py`:
- `-999` sentinels
- negative or implausible values (>2000 µg/m³)
- duplicate timestamps
- stuck sensors (≥6 identical consecutive hours)
- stations outside the study box
- timestamps in IST, which are converted to UTC
