#!/usr/bin/env python3
"""Download a WeatherNext 3 forecast for a bounding box and save it as NetCDF.

Earth Engine only: no Cloud Storage bucket, no Requester-Pays, no billing account.
An Earth Engine project (free for noncommercial use) and one browser login are enough.

    python weathernext3_forecast.py --project MY_EE_PROJECT --bbox 6.0 44.0 14.0 48.0

Defaults give the next 14 days of hourly forecasts for precipitation and 2 m temperature
on the native 0.1 deg grid, as a CF-style NetCDF with dimensions (time, latitude, longitude).

Resolution note: WeatherNext 3 is multi-resolution. The 0.1 deg collection carries 19 surface
variables including precipitation; the 0.05 deg (~5 km) collection carries station-calibrated
2 m temperature and dewpoint only. There is no 5 km precipitation field.

Ensemble note: Earth Engine publishes ensemble summary statistics (mean, p10, p25, p50, p75,
p90), not the 64 individual members. Members live in the Cloud Storage Zarr store, which does
require a billing project.
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

COLLECTIONS = {
    "0p1": "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg",
    "0p05": "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p05deg",
}
NATIVE_DEG = {"0p1": 0.1, "0p05": 0.05}
STATS = ("mean", "p10", "p25", "p50", "p75", "p90")
PRECIP_VARS = ("total_precipitation_1hr", "imerg_tp_1hr", "experimental_tp_1hr")
VARS_0P05 = ("station_head_temperature_2m", "station_head_dewpoint_temperature_2m")

# Only the 6-hourly synoptic runs reach 360 h; interim hourly runs stop at 48 h.
SYNOPTIC_HOURS = (0, 6, 12, 18)
HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"


# --------------------------------------------------------------------------- #
# Run selection
# --------------------------------------------------------------------------- #

def available_runs(collection_id: str, lookback_days: int = 3) -> list[str]:
    """start_time values published in the recent past, oldest first."""
    import ee

    now = dt.datetime.now(dt.timezone.utc)
    col = ee.ImageCollection(collection_id).filterDate(
        (now - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%S"),
        (now + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S"),
    )
    return sorted(set(col.aggregate_array("start_time").getInfo()))


def resolve_run(collection_id: str, init: str, allow_interim: bool) -> str:
    if init != "latest":
        return init

    runs = available_runs(collection_id)
    if not runs:
        raise SystemExit(
            "No runs found in the last 3 days. Check that your project has access to the "
            "WeatherNext collections (https://developers.google.com/weathernext)."
        )
    if not allow_interim:
        synoptic = [r for r in runs if pd.Timestamp(r).hour in SYNOPTIC_HOURS]
        if synoptic:
            return synoptic[-1]
        print("[warn] no 00/06/12/18 UTC run published yet; falling back to an interim run "
              "(48 h horizon).", file=sys.stderr)
    return runs[-1]


# --------------------------------------------------------------------------- #
# Grid handling
# --------------------------------------------------------------------------- #

def native_window(ic, bbox) -> dict:
    """Read window snapped to the collection's own pixel grid.

    Deriving a grid from the bounding-box corners instead can land the output half a cell
    off the model grid, which silently makes Earth Engine resample the values.
    """
    from xee import helpers

    nat = helpers.extract_grid_params(ic)
    sx, _, x0, _, sy, y0 = [float(v) for v in nat["crs_transform"]]
    west, south, east, north = bbox
    c0 = math.floor((west - x0) / sx)
    r0 = math.floor((north - y0) / sy)          # sy is negative (north-up)
    origin_x, origin_y = x0 + c0 * sx, y0 + r0 * sy
    nx = max(1, math.ceil((east - origin_x) / sx))
    ny = max(1, math.ceil((south - origin_y) / sy))
    return {"crs": nat["crs"],
            "crs_transform": (sx, 0.0, origin_x, 0.0, sy, origin_y),
            "shape_2d": (int(nx), int(ny))}


def interp_window(bbox, scale_deg: float) -> dict:
    west, south, east, north = bbox
    return {"crs": "EPSG:4326",
            "crs_transform": (scale_deg, 0.0, west, 0.0, -scale_deg, north),
            "shape_2d": (max(1, math.ceil((east - west) / scale_deg)),
                         max(1, math.ceil((north - south) / scale_deg)))}


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #

def fetch(collection_id: str, bands: list[str], run: str, hours: int, bbox,
          interp_scale: float | None) -> tuple[xr.Dataset, np.ndarray]:
    import ee

    geom = ee.Geometry.BBox(*bbox)
    ic = (ee.ImageCollection(collection_id)
          .filter(ee.Filter.eq("start_time", run))
          .filterBounds(geom)
          .select(bands))
    if hours:
        ic = ic.filter(ee.Filter.lte("forecast_hour", hours))

    # Every image in a run shares system:time_start (the run time), so the axis xee builds
    # from it is constant. Sort by forecast_hour and label the steps from that property.
    ic = ic.sort("forecast_hour")
    fh = np.asarray(ic.aggregate_array("forecast_hour").getInfo(), dtype="int64")
    if fh.size == 0:
        raise SystemExit(f"No images for start_time={run}. Try --list-inits.")

    grid = interp_window(bbox, interp_scale) if interp_scale else native_window(ic, bbox)
    nx, ny = grid["shape_2d"]
    print(f"[1/3] run {run} | lead {fh.min()}-{fh.max()} h ({fh.size} steps) | "
          f"grid {nx} x {ny} px @ {abs(grid['crs_transform'][0])} deg | {len(bands)} band(s)",
          flush=True)

    print("[2/3] fetching pixels from Earth Engine ...", flush=True)
    ds = xr.open_dataset(ic, engine="ee", **grid, chunks={}).load()
    if ds.sizes["time"] != fh.size:
        raise RuntimeError(f"{ds.sizes['time']} images returned but {fh.size} forecast hours "
                           "listed — refusing to guess the time axis.")
    return ds, fh


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #

def tidy(ds: xr.Dataset, fh: np.ndarray, *, run: str, collection_id: str,
         native_deg: float, bbox, interp_scale: float | None) -> xr.Dataset:
    ren = {}
    for a, b in (("lon", "longitude"), ("X", "longitude"), ("x", "longitude"),
                 ("lat", "latitude"), ("Y", "latitude"), ("y", "latitude")):
        if a in ds.dims:
            ren[a] = b
    ds = ds.rename(ren)

    init_np = np.datetime64(pd.Timestamp(run.replace("Z", "+00:00")).tz_localize(None))
    ds = ds.assign_coords(
        time=("time", init_np + (fh * 3_600_000_000_000).astype("timedelta64[ns]")),
        lead_hours=("time", fh.astype("float32")),
    )
    if not ds.indexes["time"].is_unique:
        raise RuntimeError("duplicate valid times — check the forecast_hour property")

    ds = ds.sortby("time").sortby("latitude").sortby("longitude")
    ds = ds.transpose("time", "latitude", "longitude", ...)
    ds = ds.assign_coords(init_time=init_np)

    for name in list(ds.data_vars):
        base, _, stat = name.rpartition("_")
        if base in PRECIP_VARS:
            ds[name] = ds[name] * 1000.0                       # metres -> mm
            ds[name].attrs.update(units="mm", standard_name="precipitation_amount",
                                  cell_methods="time: sum (1-hour accumulation)")
        elif "temperature" in base or "dewpoint" in base:
            ds[name] = ds[name] - 273.15                       # K -> degC
            ds[name].attrs.update(units="degC")
        ds[name].attrs.update(long_name=base.replace("_", " "), source_variable=base,
                              ensemble_statistic=stat)
        ds[name] = ds[name].astype("float32")

    ds["latitude"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds["longitude"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    ds["time"].attrs.update(standard_name="time", axis="T",
                            description="valid time (UTC) = start_time + forecast_hour")
    ds["lead_hours"].attrs.update(units="hours", long_name="lead time since initialization")
    ds["init_time"].attrs.update(long_name="forecast initialization time (UTC)")

    west, south, east, north = bbox
    ds.attrs.update(
        title="WeatherNext 3 forecast subset",
        summary="Per-pixel WeatherNext 3 forecast grid read from Earth Engine. Ensemble "
                "summary statistics, not individual members.",
        institution="Google DeepMind / Google Research (WeatherNext 3)",
        source=collection_id,
        model="WeatherNext 3",
        initialization_time=run,
        native_resolution_deg=native_deg,
        spatial_resolution_deg=interp_scale or native_deg,
        resampling="none (native model grid)" if not interp_scale else
                   f"bilinear-style resampling from {native_deg} deg to {interp_scale} deg — "
                   "interpolated, adds no information",
        requested_bbox=f"west {west}, south {south}, east {east}, north {north}",
        Conventions="CF-1.8",
        license="Data: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/legalcode)",
        history=f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} "
                "created by weathernext3_forecast.py",
    )
    return ds


def write_netcdf(ds: xr.Dataset, path: str, complevel: int = 4) -> str:
    enc = {v: {"zlib": True, "complevel": complevel, "dtype": "float32",
               "_FillValue": np.float32(np.nan)} for v in ds.data_vars}
    enc["time"] = {"units": "hours since 1970-01-01 00:00:00", "dtype": "int64"}
    ds.to_netcdf(path, engine="netcdf4", encoding=enc)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a WeatherNext 3 forecast for a bounding box as NetCDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                   help="Earth Engine project id (or set EE_PROJECT).")
    p.add_argument("--bbox", nargs=4, type=float, required=False,
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   default=[6.0, 44.0, 14.0, 48.0],
                   help="Area of interest in degrees (Earth Engine order).")
    p.add_argument("--days", type=float, default=14.0,
                   help="Forecast length in days (max 15; synoptic runs reach 360 h).")
    p.add_argument("--res", choices=("0p1", "0p05"), default="0p1",
                   help="0p1 = 0.1 deg surface grid (has precipitation); "
                        "0p05 = 0.05 deg station-head temperature/dewpoint only.")
    p.add_argument("--vars", nargs="+", default=["total_precipitation_1hr", "temperature_2m"],
                   help="Base variable names, without the statistic suffix.")
    p.add_argument("--stats", nargs="+", default=["mean", "p10", "p90"], choices=STATS,
                   help="Ensemble statistics to download.")
    p.add_argument("--init", default="latest",
                   help="'latest' (newest 00/06/12/18 UTC run) or an exact start_time such as "
                        "2026-09-20T00:00:00Z.")
    p.add_argument("--allow-interim", action="store_true",
                   help="Permit interim hourly runs (01-05, 07-11, ... UTC), which stop at 48 h.")
    p.add_argument("--interp-scale", type=float, default=None,
                   help="Resample onto this pixel size in degrees instead of the native grid. "
                        "Interpolation only; adds no information.")
    p.add_argument("--out", default=None, help="Output NetCDF path.")
    p.add_argument("--list-inits", action="store_true",
                   help="Print the runs published in the last 3 days and exit.")
    p.add_argument("--complevel", type=int, default=4, help="NetCDF deflate level (0-9).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.project:
        raise SystemExit("--project is required (your Earth Engine project id).")

    if args.res == "0p05":
        bad = [v for v in args.vars if v not in VARS_0P05]
        if bad:
            raise SystemExit(
                f"{bad} are not in the 0.05 deg collection, which holds only {list(VARS_0P05)}. "
                "Precipitation is native 0.1 deg: use --res 0p1, optionally with "
                "--interp-scale 0.05 to interpolate onto a 5 km mesh."
            )

    import ee

    ee.Initialize(project=args.project, opt_url=HIGH_VOLUME)
    collection_id = COLLECTIONS[args.res]

    if args.list_inits:
        for run in available_runs(collection_id):
            tag = "synoptic (360 h)" if pd.Timestamp(run).hour in SYNOPTIC_HOURS else "interim (48 h)"
            print(f"{run}  {tag}")
        return 0

    run = resolve_run(collection_id, args.init, args.allow_interim)
    hours = int(round(args.days * 24))
    bands = [f"{v}_{s}" for v in args.vars for s in args.stats]

    ds, fh = fetch(collection_id, bands, run, hours, tuple(args.bbox), args.interp_scale)
    if fh.max() < hours:
        print(f"[warn] this run only extends to {fh.max()} h; asked for {hours} h. "
              "Interim hourly runs stop at 48 h — use a 00/06/12/18 UTC run for the full horizon.",
              file=sys.stderr)

    ds = tidy(ds, fh, run=run, collection_id=collection_id,
              native_deg=NATIVE_DEG[args.res], bbox=tuple(args.bbox),
              interp_scale=args.interp_scale)

    out = args.out or (f"weathernext3_{args.res}_"
                       f"{pd.Timestamp(run.replace('Z', '+00:00')).tz_localize(None):%Y%m%d_%HZ}"
                       f"_{int(fh.max())}h.nc")
    write_netcdf(ds, out, args.complevel)
    print(f"[3/3] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) | dims {dict(ds.sizes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
