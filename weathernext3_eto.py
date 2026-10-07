#!/usr/bin/env python3
"""WeatherNext 3 -> hourly and daily FAO-56 reference evapotranspiration (ET0).

Downloads the forecast fields needed for FAO-56 from the WeatherNext 3 Earth Engine
collection (0.1 deg), adds elevation from a DEM, computes ET0 at HOURLY resolution and
aggregates to daily totals. Output: one NetCDF with hourly fields and one with daily fields,
both carrying precipitation, temperature and ET0.

    python weathernext3_eto.py --project MY_EE_PROJECT --bbox WEST SOUTH EAST NORTH --days 14

Earth Engine only: no Cloud Storage bucket, no billing account.

Three corrections to the usual recipe, applied here
---------------------------------------------------
1. `wind_speed_10m` is NOT a WeatherNext variable. The documented surface list has
   `u_component_of_wind_10m` and `v_component_of_wind_10m`; wind speed is computed as
   sqrt(u^2 + v^2) and then converted to 2 m with u2 = 0.748 * u10 (FAO-56 Eq. 47 at z=10).
2. The 900 / 0.34 coefficients belong to the DAILY equation (FAO-56 Eq. 6). The hourly form
   (FAO-56 Eq. 53) uses Cn = 37 and a soil-heat term that is not negligible:
   G = 0.1 Rn during daylight and 0.5 Rn at night. Using 900 on hourly data overstates the
   aerodynamic term by a factor of ~24.
3. Net radiation needs clear-sky radiation Rso, which needs extraterrestrial radiation Ra for
   the hour (FAO-56 Eq. 28) — a function of latitude, longitude, day of year and solar time.
   That is implemented here, so no cloud-cover variable is required.

Equations (FAO-56, Allen et al. 1998)
-------------------------------------
    es, ea    Eq. 11 applied to T and to Td          VPD = es - ea
    Delta     Eq. 13
    P         Eq. 7 from elevation (or MSLP reduced to station level)
    gamma     Eq. 8,  gamma = 0.000665 P
    u2        Eq. 47,  u2 = u10 * 4.87 / ln(67.8*10 - 5.42) = 0.748 u10
    Ra_hr     Eq. 28 with Eq. 29-33 (declination, seasonal correction, solar time angles)
    Rso       Eq. 37,  (0.75 + 2e-5 z) Ra
    Rns       Eq. 38,  (1 - 0.23) Rs
    Rnl       Eq. 39 with the hourly Stefan-Boltzmann constant 2.043e-10 MJ m-2 h-1 K-4
    ET0       Eq. 53 (hourly), summed to daily totals

Caveats
-------
* Earth Engine serves ensemble statistics. ET0 computed from the `_mean` fields is "ET0 of the
  ensemble-mean weather", which is not the ensemble mean of ET0 (the equation is nonlinear).
  For an ensemble-mean ET0 you would need the 64 members from the Cloud Storage store.
* Night-time Rs/Rso cannot be measured, so each day's well-lit hours supply the ratio used for
  that day's night hours (FAO-56 recommends a late-afternoon value); the ratio is clipped to
  [0.3, 1.0].
* Daily totals are summed over UTC days unless --utc-offset is given.
* ET0 is a reference-grass demand, not actual evapotranspiration from your catchment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

COLLECTION = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg"
NATIVE_DEG = 0.1
HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"
SYNOPTIC_HOURS = (0, 6, 12, 18)

# WeatherNext variables required by FAO-56 (plus precipitation for the water balance).
BASE_VARS = (
    "temperature_2m",                        # K
    "dewpoint_temperature_2m",               # K
    "u_component_of_wind_10m",               # m/s
    "v_component_of_wind_10m",               # m/s
    "surface_solar_radiation_downwards_1hr",  # J/m2 per hour
    "mean_sea_level_pressure",               # Pa
    "total_precipitation_1hr",               # m per hour
)

ALBEDO = 0.23           # FAO-56 reference grass
GSC = 0.0820            # solar constant, MJ m-2 min-1
SIGMA_HR = 2.043e-10    # Stefan-Boltzmann, MJ m-2 h-1 K-4
CN_HOURLY = 37.0        # FAO-56 Eq. 53 numerator constant
CD_HOURLY = 0.34        # FAO-56 Eq. 53 denominator constant


# NetCDF writers, in order of preference. Colab ships none of them by default.
NC_ENGINES = {"netcdf4": "netCDF4", "h5netcdf": "h5netcdf", "scipy": "scipy"}

# reduceResolution accepts at most 65536 input pixels per output pixel; stay well under.
MAX_PX_PER_STAGE = 4096


def pick_nc_engine(preferred: str | None = None) -> str:
    """First available NetCDF writer. Raises before any download if none is installed."""
    import importlib.util

    order = ([preferred] + [e for e in NC_ENGINES if e != preferred]) if preferred \
        else list(NC_ENGINES)
    for engine in order:
        if importlib.util.find_spec(NC_ENGINES[engine]) is not None:
            return engine
    raise SystemExit(
        "No NetCDF writer is installed, so the results could not be saved.\n"
        "Install one before running (in Colab: !pip install -q netcdf4):\n"
        "    pip install netcdf4        # preferred\n"
        "    pip install h5netcdf       # alternative"
    )


def dem_stages(dem_scale_m: float, target_scale_m: float,
               max_px: int = MAX_PX_PER_STAGE) -> list[int]:
    """Input pixels per output pixel for each reduceResolution stage.

    Going from 30 m SRTM to a 0.1 deg (~11 km) cell is ~137000 source pixels per target
    pixel, far above the per-call limit, so the aggregation is split into stages.
    """
    ratio = max(1.0, (target_scale_m / dem_scale_m) ** 2)
    stages = []
    while ratio > max_px:
        stages.append(max_px)
        ratio /= max_px
    stages.append(max(1, int(round(ratio))))
    return stages


# --------------------------------------------------------------------------- #
# FAO-56 building blocks (pure numpy — unit-testable without Earth Engine)
# --------------------------------------------------------------------------- #

def svp(t_c):
    """Saturation vapour pressure [kPa] at temperature t_c [degC] (FAO-56 Eq. 11)."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def delta_svp(t_c):
    """Slope of the vapour-pressure curve [kPa/degC] (FAO-56 Eq. 13)."""
    return 4098.0 * svp(t_c) / (t_c + 237.3) ** 2


def pressure_from_elevation(z_m):
    """Atmospheric pressure [kPa] from elevation [m] (FAO-56 Eq. 7)."""
    return 101.3 * ((293.0 - 0.0065 * z_m) / 293.0) ** 5.26


def pressure_from_mslp(mslp_pa, z_m):
    """Station pressure [kPa] by reducing mean sea level pressure to elevation z."""
    return (mslp_pa / 1000.0) * ((293.0 - 0.0065 * z_m) / 293.0) ** 5.26


def psychrometric(p_kpa):
    """Psychrometric constant [kPa/degC] (FAO-56 Eq. 8)."""
    return 0.000665 * p_kpa


def wind_2m(u10):
    """Wind speed at 2 m [m/s] from 10 m wind (FAO-56 Eq. 47)."""
    return u10 * 4.87 / math.log(67.8 * 10.0 - 5.42)


def ra_hourly(doy, t_mid_hours, lat_deg, lon_deg, lz_deg=0.0):
    """Extraterrestrial radiation for an hourly period [MJ m-2 h-1] (FAO-56 Eq. 28).

    doy          day of year
    t_mid_hours  clock time at the midpoint of the hour, in the time zone whose
                 central meridian is lz_deg (default 0 = UTC)
    lat_deg      latitude, north positive
    lon_deg      longitude, EAST positive (converted internally to FAO's west-positive Lm)
    """
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)

    b = 2.0 * np.pi * (doy - 81.0) / 364.0
    sc = 0.1645 * np.sin(2.0 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)

    lm = -lon_deg                     # FAO-56 measures longitude positive WEST of Greenwich
    omega = (np.pi / 12.0) * ((t_mid_hours + 0.06667 * (lz_deg - lm) + sc) - 12.0)
    omega = np.arctan2(np.sin(omega), np.cos(omega))      # wrap to [-pi, pi]

    w1 = omega - np.pi / 24.0         # one-hour period
    w2 = omega + np.pi / 24.0

    cos_ws = np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(cos_ws)            # sunset hour angle
    w1 = np.clip(w1, -ws, ws)
    w2 = np.clip(w2, -ws, ws)

    ra = (12.0 * 60.0 / np.pi) * GSC * dr * (
        (w2 - w1) * np.sin(phi) * np.sin(decl)
        + np.cos(phi) * np.cos(decl) * (np.sin(w2) - np.sin(w1))
    )
    return np.maximum(ra, 0.0)


def rso_hourly(ra, z_m):
    """Clear-sky radiation [MJ m-2 h-1] (FAO-56 Eq. 37)."""
    return (0.75 + 2e-5 * z_m) * ra


def rnl_hourly(t_c, ea_kpa, rs_rso_ratio):
    """Net outgoing longwave radiation [MJ m-2 h-1] (FAO-56 Eq. 39, hourly sigma)."""
    tk4 = (t_c + 273.16) ** 4
    return (SIGMA_HR * tk4
            * (0.34 - 0.14 * np.sqrt(np.maximum(ea_kpa, 0.0)))
            * (1.35 * rs_rso_ratio - 0.35))


def eto_hourly(delta, rn, g, gamma, t_c, u2, es, ea, cd=CD_HOURLY, cn=CN_HOURLY):
    """FAO-56 Eq. 53 — reference evapotranspiration for one hour [mm]."""
    num = 0.408 * delta * (rn - g) + gamma * (cn / (t_c + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + cd * u2)
    return num / den


# --------------------------------------------------------------------------- #
# ET0 on a gridded dataset
# --------------------------------------------------------------------------- #

def compute_eto(ds: xr.Dataset, elevation: xr.DataArray, *, stat: str = "mean",
                pressure_source: str = "elevation", accum: str = "preceding",
                night_ratio_bounds=(0.3, 1.0)) -> xr.Dataset:
    """Add hourly FAO-56 ET0 and its components to a WeatherNext dataset.

    ds must hold <variable>_<stat> bands in native units, with dims (time, latitude, longitude).
    """
    g = lambda name: ds[f"{name}_{stat}"]

    t_c = g("temperature_2m") - 273.15
    td_c = g("dewpoint_temperature_2m") - 273.15
    u10 = np.sqrt(g("u_component_of_wind_10m") ** 2 + g("v_component_of_wind_10m") ** 2)
    rs = g("surface_solar_radiation_downwards_1hr") / 1e6          # J/m2 -> MJ/m2 per hour
    precip = g("total_precipitation_1hr") * 1000.0                 # m -> mm

    es = svp(t_c)
    ea = svp(td_c)                                                  # Eq. 14: ea = es(Tdew)
    ea = xr.where(ea > es, es, ea)                                  # guard supersaturation
    vpd = es - ea
    delta = delta_svp(t_c)

    if pressure_source == "mslp":
        p_kpa = pressure_from_mslp(g("mean_sea_level_pressure"), elevation)
    else:
        p_kpa = pressure_from_elevation(elevation) + 0.0 * t_c      # broadcast over time
    gamma = psychrometric(p_kpa)
    u2 = wind_2m(u10)

    # Solar geometry at the midpoint of each accumulation hour.
    shift = -0.5 if accum == "preceding" else 0.5
    t_mid = ds["time"] + pd.Timedelta(hours=shift)
    doy = t_mid.dt.dayofyear
    hour_mid = t_mid.dt.hour + t_mid.dt.minute / 60.0

    ra = xr.apply_ufunc(
        ra_hourly,
        doy, hour_mid, ds["latitude"], ds["longitude"],
        kwargs={"lz_deg": 0.0},                                     # time axis is UTC
        output_dtypes=[float],
    ).transpose(*[d for d in ("time", "latitude", "longitude") if d in ds.dims])

    rso = rso_hourly(ra, elevation)
    daylight = rso > 0.05

    ratio = (rs / rso).where(daylight)
    day = ds["time"].dt.floor("D")
    lit = ratio.where(rso > 0.5 * rso.max("time"))      # well-lit hours only
    day_ratio = lit.assign_coords(day=day).groupby("day").mean("time")
    ratio_filled = xr.where(daylight, ratio, day_ratio.sel(day=day).drop_vars("day"))
    ratio_filled = ratio_filled.clip(*night_ratio_bounds).fillna(night_ratio_bounds[1])

    rnl = rnl_hourly(t_c, ea, ratio_filled)
    rns = (1.0 - ALBEDO) * rs
    rn = rns - rnl
    soil_g = xr.where(daylight, 0.1 * rn, 0.5 * rn)                 # FAO-56 Eq. 45/46

    et0 = eto_hourly(delta, rn, soil_g, gamma, t_c, u2, es, ea).clip(min=0.0)

    out = xr.Dataset({
        "eto": et0.astype("float32"),
        "precipitation": precip.astype("float32"),
        "temperature_2m": t_c.astype("float32"),
        "dewpoint_temperature_2m": td_c.astype("float32"),
        "vpd": vpd.astype("float32"),
        "wind_speed_2m": u2.astype("float32"),
        "net_radiation": rn.astype("float32"),
        "solar_radiation": rs.astype("float32"),
        "pressure": p_kpa.astype("float32"),
        "elevation": elevation.astype("float32"),
    })

    units = {"eto": "mm", "precipitation": "mm", "temperature_2m": "degC",
             "dewpoint_temperature_2m": "degC", "vpd": "kPa", "wind_speed_2m": "m s-1",
             "net_radiation": "MJ m-2 h-1", "solar_radiation": "MJ m-2 h-1",
             "pressure": "kPa", "elevation": "m"}
    long_names = {
        "eto": "FAO-56 hourly reference evapotranspiration",
        "precipitation": "1-hour total precipitation",
        "temperature_2m": "2 m air temperature",
        "dewpoint_temperature_2m": "2 m dewpoint temperature",
        "vpd": "vapour pressure deficit (es - ea)",
        "wind_speed_2m": "wind speed at 2 m (converted from 10 m)",
        "net_radiation": "net radiation over the reference surface",
        "solar_radiation": "downward shortwave radiation",
        "pressure": "atmospheric pressure at the surface",
        "elevation": "grid-cell mean elevation",
    }
    for name in out.data_vars:
        out[name].attrs.update(units=units[name], long_name=long_names[name])
    out["eto"].attrs.update(method="FAO-56 Eq. 53 (hourly), Cn=37, Cd=0.34, G=0.1Rn day / 0.5Rn night")
    return out


def to_daily(hourly: xr.Dataset, utc_offset: float = 0.0) -> xr.Dataset:
    """Daily aggregation: ET0 and precipitation summed, temperature mean/min/max."""
    shifted = hourly.assign_coords(time=hourly["time"] + pd.Timedelta(hours=utc_offset))
    grp = shifted.groupby("time.date")
    daily = xr.Dataset({
        "eto": grp.sum("time")["eto"],
        "precipitation": grp.sum("time")["precipitation"],
        "temperature_2m_mean": grp.mean("time")["temperature_2m"],
        "temperature_2m_min": grp.min("time")["temperature_2m"],
        "temperature_2m_max": grp.max("time")["temperature_2m"],
        "vpd_mean": grp.mean("time")["vpd"],
        "wind_speed_2m_mean": grp.mean("time")["wind_speed_2m"],
    })
    daily = daily.rename({"date": "time"})
    daily["time"] = pd.to_datetime(list(daily["time"].values))
    daily["elevation"] = hourly["elevation"]

    n_hours = grp.count("time")["eto"].rename({"date": "time"})
    daily["hours_in_day"] = ("time", np.asarray(n_hours.isel(
        {d: 0 for d in n_hours.dims if d != "time"}).values, dtype="int16"))
    daily["hours_in_day"].attrs.update(
        long_name="hourly steps contributing to this day (24 = complete)")

    for name, unit, long in (("eto", "mm", "FAO-56 daily reference evapotranspiration (sum of hourly)"),
                             ("precipitation", "mm", "daily total precipitation"),
                             ("temperature_2m_mean", "degC", "daily mean 2 m temperature"),
                             ("temperature_2m_min", "degC", "daily minimum 2 m temperature"),
                             ("temperature_2m_max", "degC", "daily maximum 2 m temperature"),
                             ("vpd_mean", "kPa", "daily mean vapour pressure deficit"),
                             ("wind_speed_2m_mean", "m s-1", "daily mean 2 m wind speed")):
        daily[name].attrs.update(units=unit, long_name=long)
    daily["elevation"].attrs.update(units="m", long_name="grid-cell mean elevation")
    daily.attrs.update(utc_offset_hours=utc_offset)
    return daily


# --------------------------------------------------------------------------- #
# Earth Engine access
# --------------------------------------------------------------------------- #

def available_runs(lookback_days: int = 3) -> list[str]:
    import ee

    now = dt.datetime.now(dt.timezone.utc)
    col = ee.ImageCollection(COLLECTION).filterDate(
        (now - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%S"),
        (now + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S"),
    )
    return sorted(set(col.aggregate_array("start_time").getInfo()))


def resolve_run(init: str, allow_interim: bool) -> str:
    if init != "latest":
        return init
    runs = available_runs()
    if not runs:
        raise SystemExit("No runs published in the last 3 days — check project access.")
    if not allow_interim:
        synoptic = [r for r in runs if pd.Timestamp(r).hour in SYNOPTIC_HOURS]
        if synoptic:
            return synoptic[-1]
        print("[warn] no 00/06/12/18 UTC run yet; using an interim run (48 h horizon).",
              file=sys.stderr)
    return runs[-1]


def native_window(ic, bbox) -> dict:
    """Read window snapped to the collection's own pixel grid (no resampling)."""
    from xee import helpers

    nat = helpers.extract_grid_params(ic)
    sx, _, x0, _, sy, y0 = [float(v) for v in nat["crs_transform"]]
    west, south, east, north = bbox
    c0, r0 = math.floor((west - x0) / sx), math.floor((north - y0) / sy)
    ox, oy = x0 + c0 * sx, y0 + r0 * sy
    return {"crs": nat["crs"],
            "crs_transform": (sx, 0.0, ox, 0.0, sy, oy),
            "shape_2d": (max(1, math.ceil((east - ox) / sx)),
                         max(1, math.ceil((south - oy) / sy)))}


def fetch_forecast(run: str, hours: int, bbox, stat: str):
    """Download the FAO-56 input bands. Returns (dataset, forecast_hours, grid)."""
    import ee

    bands = [f"{v}_{stat}" for v in BASE_VARS]
    ic = (ee.ImageCollection(COLLECTION)
          .filter(ee.Filter.eq("start_time", run))
          .filterBounds(ee.Geometry.BBox(*bbox))
          .select(bands))
    if hours:
        ic = ic.filter(ee.Filter.lte("forecast_hour", hours))

    # system:time_start is the RUN time for every step, so label the axis from forecast_hour.
    ic = ic.sort("forecast_hour")
    fh = np.asarray(ic.aggregate_array("forecast_hour").getInfo(), dtype="int64")
    if fh.size == 0:
        raise SystemExit(f"No images for start_time={run}. Try --list-inits.")

    grid = native_window(ic, bbox)
    nx, ny = grid["shape_2d"]
    print(f"[1/5] run {run} | lead {fh.min()}-{fh.max()} h ({fh.size} steps) | "
          f"grid {nx} x {ny} px | {len(bands)} bands", flush=True)

    print("[2/5] downloading forecast fields ...", flush=True)
    ds = xr.open_dataset(ic, engine="ee", **grid, chunks={}).load()
    if ds.sizes["time"] != fh.size:
        raise RuntimeError(f"{ds.sizes['time']} images but {fh.size} forecast hours listed.")

    ren = {a: b for a, b in (("lon", "longitude"), ("X", "longitude"), ("x", "longitude"),
                             ("lat", "latitude"), ("Y", "latitude"), ("y", "latitude"))
           if a in ds.dims}
    ds = ds.rename(ren)
    init_np = np.datetime64(pd.Timestamp(run.replace("Z", "+00:00")).tz_localize(None))
    ds = ds.assign_coords(
        time=("time", init_np + (fh * 3_600_000_000_000).astype("timedelta64[ns]")),
        lead_hours=("time", fh.astype("float32")),
    ).sortby("time").sortby("latitude").sortby("longitude")
    if not ds.indexes["time"].is_unique:
        raise RuntimeError("duplicate valid times — check the forecast_hour property")
    return ds, fh, grid


def fetch_elevation(grid: dict, dem_asset: str) -> xr.DataArray:
    """Grid-cell MEAN DEM elevation on the forecast grid (aggregated, not point-sampled)."""
    import ee

    dem = ee.Image(dem_asset).select(0).rename("elevation").unmask(0)
    dem_scale = float(dem.projection().nominalScale().getInfo())
    target_scale = abs(float(grid["crs_transform"][0])) * 111320.0      # deg -> m at equator
    stages = dem_stages(dem_scale, target_scale)
    print(f"[3/5] downloading elevation from {dem_asset} "
          f"({dem_scale:.0f} m -> {target_scale:.0f} m, {len(stages)} aggregation stage(s)) ...",
          flush=True)

    img = dem
    for px in stages[:-1]:                                              # intermediate stages
        factor = int(round(px ** 0.5))
        img = (img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=px)
               .reproject(crs=img.projection().scale(factor, factor)))
    coarse = (img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=stages[-1] * 4)
              .reproject(crs=grid["crs"], crsTransform=list(grid["crs_transform"])))

    ic = ee.ImageCollection([coarse.set("system:time_start", 0)])
    ds = xr.open_dataset(ic, engine="ee", **grid, chunks={}).load()
    ren = {a: b for a, b in (("lon", "longitude"), ("X", "longitude"), ("x", "longitude"),
                             ("lat", "latitude"), ("Y", "latitude"), ("y", "latitude"))
           if a in ds.dims}
    elev = ds.rename(ren)["elevation"]
    if "time" in elev.dims:
        elev = elev.isel(time=0, drop=True)
    elev = elev.sortby("latitude").sortby("longitude").fillna(0.0)
    print(f"      elevation {float(elev.min()):.0f}-{float(elev.max()):.0f} m", flush=True)
    return elev


# --------------------------------------------------------------------------- #
# Self-test: FAO-56 Example 19 (hourly ET0, N'Diaye, Senegal, 1 October)
# --------------------------------------------------------------------------- #

def self_test() -> int:
    """Reproduce FAO-56 Example 19 (14:00-15:00 h), published answer 0.63 mm h-1."""
    lat, lon_east, z = 16.217, -16.25, 8.0     # 16 13' N, 16 15' W, 8 m
    doy, t_mid = 274, 14.5                     # 1 October, midpoint of 14:00-15:00
    t_c, rh, u2_obs, rs = 38.0, 52.0, 3.3, 2.450

    es_v = svp(t_c)
    ea_v = es_v * rh / 100.0
    delta_v = delta_svp(t_c)
    gamma_v = psychrometric(pressure_from_elevation(z))
    ra = ra_hourly(doy, t_mid, lat, lon_east, lz_deg=15.0)   # local standard time, 15 W
    rso = rso_hourly(ra, z)
    rnl = rnl_hourly(t_c, ea_v, rs / rso)
    rn = (1 - ALBEDO) * rs - rnl
    g = 0.1 * rn
    et0 = eto_hourly(delta_v, rn, g, gamma_v, t_c, u2_obs, es_v, ea_v)

    ref = {"es": 6.625, "ea": 3.445, "delta": 0.358, "gamma": 0.0673,
           "Ra": 3.543, "Rso": 2.658, "Rn": 1.749, "ET0": 0.63}
    got = {"es": es_v, "ea": ea_v, "delta": delta_v, "gamma": gamma_v,
           "Ra": float(ra), "Rso": float(rso), "Rn": float(rn), "ET0": float(et0)}
    print("FAO-56 Example 19 (14:00-15:00, 1 Oct, N'Diaye)")
    ok = True
    for k, want in ref.items():
        diff = abs(got[k] - want) / max(abs(want), 1e-9)
        flag = "ok" if diff < 0.02 else "MISMATCH"
        ok &= diff < 0.02
        print(f"  {k:6s} computed {got[k]:8.4f}   published {want:8.4f}   {diff*100:5.1f}%  {flag}")
    print("\nDEM aggregation stages for a 0.1 deg grid (~11132 m)")
    for asset, scale in (("USGS/GMTED2010_FULL", 232.0), ("USGS/SRTMGL1_003", 30.0),
                         ("CGIAR/SRTM90_V4", 90.0), ("NOAA/NGDC/ETOPO1", 1855.0)):
        print(f"  {asset:22s} {scale:7.0f} m -> stages {dem_stages(scale, 11132.0)}")

    try:
        print(f"\nNetCDF writer available: {pick_nc_engine()}")
    except SystemExit as err:
        print(f"\n{err}")
        ok = False

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="WeatherNext 3 forecast -> hourly and daily FAO-56 ET0 as NetCDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                   help="Earth Engine project id (or set EE_PROJECT).")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   help="Area of interest in degrees, Earth Engine order (required).")
    p.add_argument("--days", type=float, default=14.0, help="Forecast length in days (max 15).")
    p.add_argument("--stat", default="mean", choices=("mean", "p10", "p25", "p50", "p75", "p90"),
                   help="Ensemble statistic used for every input field.")
    p.add_argument("--init", default="latest",
                   help="'latest' (newest 00/06/12/18 UTC run) or an exact start_time.")
    p.add_argument("--allow-interim", action="store_true",
                   help="Permit interim hourly runs (48 h horizon).")
    p.add_argument("--dem", default="USGS/GMTED2010_FULL",
                   help="Earth Engine DEM asset. GMTED2010_FULL (~232 m) is global and quick; "
                        "USGS/SRTMGL1_003 is finer but 60N-56S only; NOAA/NGDC/ETOPO1 for polar.")
    p.add_argument("--pressure-source", default="elevation", choices=("elevation", "mslp"),
                   help="FAO-56 Eq. 7 from elevation, or mean sea level pressure reduced to elevation.")
    p.add_argument("--accum", default="preceding", choices=("preceding", "following"),
                   help="Whether *_1hr accumulations cover the hour before or after the stamp.")
    p.add_argument("--utc-offset", type=float, default=0.0,
                   help="Hours to shift before daily aggregation (e.g. 1 for CET).")
    p.add_argument("--prefix", default=None, help="Output filename prefix.")
    p.add_argument("--list-inits", action="store_true", help="List recent runs and exit.")
    p.add_argument("--self-test", action="store_true",
                   help="Check the FAO-56 implementation against Example 19 and exit "
                        "(no Earth Engine access needed).")
    p.add_argument("--engine", default=None, choices=tuple(NC_ENGINES),
                   help="NetCDF writer to use (default: first one installed).")
    p.add_argument("--complevel", type=int, default=4)
    return p.parse_args(argv)


def write_nc(ds: xr.Dataset, path: str, complevel: int, engine: str) -> str:
    """Write NetCDF, falling back through the remaining engines if one fails."""
    candidates = [engine] + [e for e in NC_ENGINES if e != engine]
    last_err = None
    for eng in candidates:
        try:
            if eng == "scipy":                      # NETCDF3 only: no compression available
                enc = {}
            else:
                enc = {v: {"zlib": True, "complevel": complevel} for v in ds.data_vars}
                if "time" in ds.coords:
                    enc["time"] = {"units": "hours since 1970-01-01 00:00:00", "dtype": "int64"}
            ds.to_netcdf(path, engine=eng, encoding=enc)
            if eng != engine:
                print(f"      [note] wrote with the {eng} engine ({engine} failed)", flush=True)
            return path
        except Exception as err:                    # try the next writer rather than lose the run
            last_err = err
    raise SystemExit(f"Could not write {path} with any NetCDF engine: {last_err}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.project:
        raise SystemExit("--project is required (your Earth Engine project id).")

    # Checked BEFORE any download: a missing writer used to surface only after the
    # full forecast had been fetched and ET0 computed.
    engine = pick_nc_engine(args.engine)

    import ee

    ee.Initialize(project=args.project, opt_url=HIGH_VOLUME)

    if args.list_inits:
        for run in available_runs():
            tag = "synoptic (360 h)" if pd.Timestamp(run).hour in SYNOPTIC_HOURS else "interim (48 h)"
            print(f"{run}  {tag}")
        return 0

    run = resolve_run(args.init, args.allow_interim)
    ds, fh, grid = fetch_forecast(run, int(round(args.days * 24)), tuple(args.bbox), args.stat)
    if fh.max() < round(args.days * 24):
        print(f"[warn] run reaches only {fh.max()} h; asked for {int(round(args.days*24))} h.",
              file=sys.stderr)

    elevation = fetch_elevation(grid, args.dem)

    print("[4/5] computing hourly FAO-56 ET0 ...", flush=True)
    hourly = compute_eto(ds, elevation, stat=args.stat,
                         pressure_source=args.pressure_source, accum=args.accum)
    daily = to_daily(hourly, args.utc_offset)

    common = dict(
        title="WeatherNext 3 forecast with FAO-56 reference evapotranspiration",
        institution="Google DeepMind / Google Research (WeatherNext 3); ET0 after Allen et al. (1998)",
        source=COLLECTION,
        dem_source=args.dem,
        model="WeatherNext 3",
        initialization_time=run,
        ensemble_statistic=args.stat,
        ensemble_note="inputs are ensemble summary statistics; ET0 of the mean weather is not "
                      "the ensemble mean of ET0",
        eto_method="FAO-56 Allen et al. (1998) Eq. 53 (hourly), summed to daily totals",
        pressure_source=args.pressure_source,
        accumulation_convention=f"*_1hr values cover the hour {args.accum} the time stamp",
        spatial_resolution_deg=NATIVE_DEG,
        requested_bbox=f"west {args.bbox[0]}, south {args.bbox[1]}, "
                       f"east {args.bbox[2]}, north {args.bbox[3]}",
        Conventions="CF-1.8",
        license="Data: CC-BY-4.0 (Google WeatherNext 3)",
        history=f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} "
                "created by weathernext3_eto.py",
    )
    hourly.attrs.update(common, time_resolution="hourly")
    daily.attrs.update(common, time_resolution="daily")

    stamp = pd.Timestamp(run.replace("Z", "+00:00")).tz_localize(None).strftime("%Y%m%d_%HZ")
    prefix = args.prefix or f"weathernext3_eto_{stamp}"
    h_path = write_nc(hourly, f"{prefix}_hourly.nc", args.complevel, engine)
    d_path = write_nc(daily, f"{prefix}_daily.nc", args.complevel, engine)

    print(f"[5/5] wrote {h_path} ({os.path.getsize(h_path)/1e6:.1f} MB) and "
          f"{d_path} ({os.path.getsize(d_path)/1e6:.1f} MB)")
    area = daily.mean(("latitude", "longitude"))
    print("\nArea-mean daily summary")
    print(pd.DataFrame({
        "ET0 (mm)": np.round(area["eto"].values, 2),
        "Precip (mm)": np.round(area["precipitation"].values, 2),
        "Tmean (degC)": np.round(area["temperature_2m_mean"].values, 1),
        "hours": daily["hours_in_day"].values,
    }, index=pd.DatetimeIndex(daily["time"].values).date).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
