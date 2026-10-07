# weathernext3-forecast

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23224416.svg)](https://doi.org/10.5281/zenodo.23224416)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Download a **Google WeatherNext 3** weather forecast for a bounding box and save it as a
CF-style **NetCDF** grid — up to 14 days of hourly steps, precipitation and 2 m temperature.

Access is through **Google Earth Engine only**: no Cloud Storage bucket, no Requester-Pays,
no billing account. An Earth Engine project (free for noncommercial use) and one browser
login are all that is needed.

## What you get

A NetCDF file with dimensions `(time, latitude, longitude)` and one variable per
requested variable/statistic pair:

| Variable | Units | Notes |
|---|---|---|
| `total_precipitation_1hr_mean`, `_p10`, `_p90` | mm | 1-hour accumulation, converted from metres |
| `temperature_2m_mean`, `_p10`, `_p90` | °C | converted from kelvin |

Coordinates: `time` (valid time, UTC), `lead_hours`, `latitude` (ascending °N),
`longitude` (°E), scalar `init_time`. Global attributes record the collection, run time,
bounding box, resolution and whether any resampling was applied.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Use a dedicated environment. These packages pull a recent numpy, which will break an
Anaconda base environment that holds TensorFlow, numba or streamlit.

## Authenticate (once)

```bash
earthengine authenticate
```

This opens a browser and caches a token; the Google Cloud CLI is not required. Your Earth
Engine project must be registered at <https://code.earthengine.google.com/register> and have
access to the WeatherNext collections.

## Use

```bash
# 14 days of hourly precipitation + 2 m temperature over northern Italy
python weathernext3_forecast.py --project MY_EE_PROJECT --bbox WEST SOUTH EAST NORTH

# Which runs are published right now?
python weathernext3_forecast.py --project MY_EE_PROJECT --list-inits

# A specific run, 5 days, ensemble mean only
python weathernext3_forecast.py --project MY_EE_PROJECT --bbox WEST SOUTH EAST NORTH \
    --init 2026-09-20T00:00:00Z --days 5 --stats mean

# Station-calibrated temperature on the genuine 5 km grid
python weathernext3_forecast.py --project MY_EE_PROJECT --res 0p05 \
    --vars station_head_temperature_2m --stats mean
```

| Option | Default | Meaning |
|---|---|---|
| `--bbox W S E N` | `W S E N` | *required* | area of interest, degrees (Earth Engine order) |
| `--days` | `14` | forecast length; synoptic runs reach 15 days |
| `--res` | `0p1` | `0p1` = 0.1° surface grid, `0p05` = 0.05° station-head grid |
| `--vars` | `total_precipitation_1hr temperature_2m` | base variable names |
| `--stats` | `mean p10 p90` | from `mean p10 p25 p50 p75 p90` |
| `--init` | `latest` | newest 00/06/12/18 UTC run, or an exact `start_time` |
| `--allow-interim` | off | permit interim hourly runs (48 h horizon) |
| `--interp-scale` | off | resample onto a finer mesh (interpolation only) |
| `--out` | auto | output path |

A Colab version is in [`notebooks/WeatherNext3_Colab.ipynb`](notebooks/WeatherNext3_Colab.ipynb).

## Reference evapotranspiration (ET₀)

`weathernext3_eto.py` extends the downloader with FAO-56 reference evapotranspiration:

```bash
python weathernext3_eto.py --self-test                      # verify the physics, no EE needed
python weathernext3_eto.py --project MY_EE_PROJECT --bbox WEST SOUTH EAST NORTH --days 14
```

It downloads `temperature_2m`, `dewpoint_temperature_2m`, the 10 m wind components,
`surface_solar_radiation_downwards_1hr`, `mean_sea_level_pressure` and
`total_precipitation_1hr`, adds grid-cell mean elevation from a DEM (default
`USGS/SRTMGL1_003`), computes ET₀ **hourly** and sums to daily totals. Two files are written:
`*_hourly.nc` and `*_daily.nc`, both carrying precipitation, temperature and ET₀.

Three points where the standard recipe needs care, handled here:

1. **There is no `wind_speed_10m` variable.** WeatherNext publishes `u_component_of_wind_10m`
   and `v_component_of_wind_10m`; speed is `sqrt(u² + v²)`, then `u₂ = 0.748 u₁₀` (Eq. 47).
2. **The 900 / 0.34 coefficients are the daily form.** Hourly ET₀ uses FAO-56 Eq. 53 with
   `Cn = 37` and a non-negligible soil heat flux, `G = 0.1 Rₙ` by day and `0.5 Rₙ` at night.
   Applying 900 to hourly data inflates the aerodynamic term by roughly 24×.
3. **Net radiation needs clear-sky radiation**, hence extraterrestrial radiation for the hour
   (Eq. 28, from latitude, longitude, day of year and solar time). That is computed internally,
   so no cloud-cover variable is required. Night-time `Rs/Rso` is taken from the same day's
   well-lit hours, clipped to [0.3, 1.0].

Validation: `--self-test` reproduces FAO-56 Example 19 (hourly ET₀ at N'Diaye, Senegal) —
every intermediate term within 0.1 % of the published values and ET₀ 0.627 vs 0.63 mm h⁻¹.

Caveats: inputs are ensemble *statistics*, so this is ET₀ of the ensemble-mean weather rather
than the ensemble mean of ET₀ (the equation is nonlinear); `hours_in_day` in the daily file
flags partial days at the ends of the run; ET₀ is reference-grass demand, not catchment actual
evapotranspiration.

**NetCDF writer.** Colab images ship without `netCDF4`, so the script checks for a writer
*before* downloading anything (a missing writer used to surface only after the whole forecast
had been fetched) and falls back across `netcdf4` / `h5netcdf` / `scipy` at write time. In a
bare Colab session run `!pip install -q netcdf4` first, or clone the repo and
`pip install -r requirements.txt`.

Options: `--stat`, `--pressure-source elevation|mslp`, `--utc-offset` (daily aggregation
boundary), `--dem`, `--accum preceding|following`, plus the `--bbox` / `--days` / `--init`
options shared with `weathernext3_forecast.py`.

A Colab version is in [`notebooks/WeatherNext3_ETo_Colab.ipynb`](notebooks/WeatherNext3_ETo_Colab.ipynb).

Per-basin tables: feed either output file, plus a shapefile of your catchments, to
[`notebooks/WeatherNext3_SubBasin_Aggregation_Colab.ipynb`](notebooks/WeatherNext3_SubBasin_Aggregation_Colab.ipynb).
It masks the grid with each polygon, takes `cos(latitude)`-weighted basin means, and writes one
Excel workbook per NetCDF with a sheet per variable and a summary of totals and extremes.

## Things worth knowing

- **Run choice sets the horizon.** Only the 00/06/12/18 UTC runs extend to 360 h; interim
  hourly initializations (01–05, 07–11, …) stop at 48 h. `--init latest` therefore picks the
  newest *synoptic* run, not simply the newest run — asking for 14 days from an interim run
  would silently return 2 days.
- **The time axis is built from `forecast_hour`.** Every image in a run shares
  `system:time_start` (the run time), so a time axis taken from that property collapses to a
  single repeated timestamp. This tool sorts by `forecast_hour` and labels the steps from it.
- **There is no 5 km precipitation.** WeatherNext 3 is multi-resolution: the 0.1° collection
  (~11 km) holds 19 surface variables including precipitation, while the 0.05° collection
  (~5 km) holds station-calibrated 2 m temperature and dewpoint only. `--interp-scale 0.05`
  will put precipitation on a 5 km mesh, but that is interpolation and the output says so.
- **Ensemble statistics, not members.** Earth Engine publishes `mean`, `p10`, `p25`, `p50`,
  `p75`, `p90`. The 64 individual members are only in the Cloud Storage Zarr store, which
  does require a billing project.
- **Read window snapped to the model grid.** Deriving a grid from bounding-box corners can
  land the output half a cell off the native grid and make Earth Engine resample; the window
  here is aligned to the collection's own transform.

## Quick look

```python
import xarray as xr
ds = xr.open_dataset("weathernext3_0p1_20260920_00Z_336h.nc")
ds["total_precipitation_1hr_mean"].sum("time").plot()                       # 14-day total
ds["temperature_2m_mean"].mean(("latitude", "longitude")).plot()            # area-mean series
```

## Citation

Forecast data: Google DeepMind / Google Research, WeatherNext 3, distributed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode). See
<https://developers.google.com/weathernext>.

This tool: see [`CITATION.cff`](CITATION.cff).

## License

MIT for the code in this repository (see [LICENSE](LICENSE)); the forecast data carries
Google's CC-BY-4.0 terms.
