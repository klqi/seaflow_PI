"""
PAR quality control, smoothing, and gap filling for 30-minute cruise PAR data.

Pipeline (per cruise or per cruise_day group):
  0. (optional) Detect and correct a clock offset: if the observed PAR peak
     is systematically shifted from solar noon, the timestamps are shifted
     onto the solar clock (flagged in `time_shift_h`).
  1. (optional) Recompute time_of_day from solar elevation.
  2. Remove the dark sensor offset (cruise nighttime median).
  3. Flag outliers conservatively: isolated spikes, clear-sky cap, and a
     wide-window Hampel filter on the clear-sky ratio (not raw PAR, so
     the dawn-to-noon trend is never flagged).
  4. Reindex to the full 30-minute grid; impute gaps with time
     interpolation (short gaps) or a day-scaled clear-sky model (long gaps).
  5. Smooth with Savitzky-Golay (polyorder 2 preserves the dawn-noon slope).
  6. Force all nighttime values to 0, before and after smoothing.

Cruises with no usable PAR (sensor missing or broken) get PAR entirely
from the scaled clear-sky astronomical model, flagged in `par_source`.

Requires only numpy, pandas, scipy.
Solar position: NOAA formulations (fractional-year equation of time and
declination, gml.noaa.gov/grad/solcalc/solareqns.PDF).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import warnings

# nanmedian on all-NaN windows (night, missing cruises) is expected here
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="divide by zero")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# ---------------------------------------------------------------------------
# Solar position (NOAA fractional-year formulation, vectorized)
# ---------------------------------------------------------------------------

def solar_elevation(times, lat, lon):
    """
    Solar elevation in degrees for UTC datetimes.

    times : DatetimeIndex-like, timezone-aware (converted to UTC).
    lat, lon : numeric, degrees (lon positive east).
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("UTC")

    doy = idx.dayofyear.to_numpy(dtype=float)
    hour_frac = (idx.hour + idx.minute / 60.0).to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour_frac - 12.0) / 24.0)

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    # Solar declination (radians -> degrees)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    decl = np.degrees(decl)

    # True solar time -> hour angle (degrees)
    minutes_utc = (idx.hour * 60.0 + idx.minute + idx.second / 60.0).to_numpy(dtype=float)
    tst = minutes_utc + eqtime + 4.0 * np.asarray(lon, dtype=float)
    ha = tst / 4.0 - 180.0

    phi = np.radians(np.asarray(lat, dtype=float))
    dec = np.radians(decl)
    cos_zen = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(np.radians(ha))
    zenith = np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))
    return 90.0 - zenith


def clearsky_par(elev_deg, coefficient=2200.0):
    """
    Clear-sky PAR envelope (same units as the observed PAR column).

    Calibrated so the peak matches a clear tropical midday PAR of roughly
    2200 umol m-2 s-1. Tune `coefficient` to your sensor/region if needed.
    """
    e = np.maximum(np.asarray(elev_deg, dtype=float), 0.0)
    return coefficient * np.sin(np.radians(e))


def time_of_day_from_elevation(elev_deg, elev_next_hour, night_elev=-6.0):
    """
    Recompute sunrise/day/sunset/night labels from solar elevation.

    night          : elevation < night_elev (default -6 deg, end of civil
                    twilight)
    sunrise/sunset : night_elev <= elevation < 0 (decided by whether the
                    sun is rising or setting, from `elev_next_hour`)
    day            : elevation >= 0
    """
    elev = np.asarray(elev_deg, dtype=float)
    rising = np.asarray(elev_next_hour, dtype=float) > elev
    labels = np.where(
        elev >= 0.0, "day",
        np.where(elev < night_elev, "night", "twilight"),
    ).astype(object)
    for i in range(len(labels)):
        if labels[i] == "twilight":
            labels[i] = "sunrise" if rising[i] else "sunset"
    return labels


# ---------------------------------------------------------------------------
# Cruise-level QC
# ---------------------------------------------------------------------------

def flag_cruise_par_quality(
    df,
    cruise_col="cruise",
    cruise_day_col="cruise_day",
    par_col="norm_par",
    time_of_day_col="time_of_day",
    lat_col="lat",
    lon_col="lon",
    datetime_col="time",
    broken_max_par=100.0,
    offset_threshold_h=2.0,
    clearsky_coefficient=2200.0,
):
    """
    One row per cruise with PAR quality flags:

    missing         : no PAR values at all
    suspect_broken   : max PAR below broken_max_par (sensor not working)
    clock_offset     : observed PAR peak systematically shifted from solar
                      noon by more than offset_threshold_h (timestamps
                      labeled with the wrong clock / timezone); the
                      detected offset is in `par_peak_offset_h`
    """
    out = []
    for cruise, d in df.groupby(cruise_col, sort=False):
        par = pd.to_numeric(d[par_col], errors="coerce")
        elev = solar_elevation(d[datetime_col], d[lat_col], d[lon_col])
        cs = clearsky_par(elev, clearsky_coefficient)
        daylight = elev >= 0
        valid = par.notna() & daylight & (par > 0)
        csi = float((par[valid] / cs[valid]).median()) if valid.sum() >= 3 else np.nan

        # median offset between observed PAR peak and solar noon, per day
        offs = []
        if par.notna().sum() >= 6 and par.max() >= broken_max_par:
            for cd, day in d.groupby(cruise_day_col, sort=False):
                dp = pd.to_numeric(day[par_col], errors="coerce")
                if dp.notna().sum() < 6 or dp.max() < broken_max_par:
                    continue
                de = solar_elevation(day[datetime_col], day[lat_col], day[lon_col])
                if len(de) < 6 or len(np.unique(de)) < 2:
                    continue
                t_peak = pd.DatetimeIndex(day[datetime_col])[int(dp.reset_index(drop=True).idxmax())]
                t_noon = pd.DatetimeIndex(day[datetime_col])[int(np.argmax(de))]
                offs.append((t_peak - t_noon).total_seconds() / 3600.0)
        offset = float(np.median(offs)) if offs else np.nan

        out.append({
            cruise_col: cruise,
            "n_rows": len(d),
            "n_par": int(par.notna().sum()),
            "max_par": float(par.max()) if par.notna().any() else np.nan,
            "clearsky_index": csi,
            "par_peak_offset_h": offset,
            "missing": bool(par.notna().sum() == 0),
            "suspect_broken": bool(par.notna().any() and par.max() < broken_max_par),
            "clock_offset": bool(np.isfinite(offset) and abs(offset) > offset_threshold_h),
        })
    qc = pd.DataFrame(out)
    qc["flag"] = np.where(
        qc["missing"], "missing",
        np.where(qc["suspect_broken"], "suspect_broken",
                 np.where(qc["clock_offset"], "clock_offset", "ok")))
    return qc


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _hampel_mask(y, window, n_mads):
    """Boolean outlier mask: rolling median +/- n_mads * scaled MAD."""
    n = len(y)
    if n == 0 or not np.isfinite(np.nanmedian(y)):
        return np.zeros(n, dtype=bool)
    half = window // 2
    med = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        med[i] = np.nanmedian(y[lo:hi])
    resid = np.abs(y - med)
    mad = 1.4826 * np.nanmedian(resid)
    if not np.isfinite(mad) or mad <= 0:
        # fall back to a small floor so constant segments don't explode
        mad = max(1e-3, float(np.nanmedian(np.abs(y))) * 0.01)
    return resid > n_mads * mad


def _detect_clock_offset(out, par_col, datetime_col, cruise_day_col, threshold_h=1.5):
    """
    Median offset (hours) between the observed PAR peak and solar noon.
    Returns np.nan when there is not enough data to measure it.
    """
    par = pd.to_numeric(out[par_col], errors="coerce")
    if par.notna().sum() < 6 or (par.max() if par.notna().any() else 0) < 100:
        return np.nan
    offs = []
    for cd, pos in out.groupby(cruise_day_col, sort=False).indices.items():
        day_par = par.to_numpy()[pos]
        if np.isnan(day_par).all():
            continue
        if np.nanmax(day_par) < 100:
            continue
        times = pd.DatetimeIndex(out[datetime_col].to_numpy()[pos])
        if len(times) < 6:
            continue
        elev = solar_elevation(times, out["lat"].astype(float).to_numpy()[pos],
                               out["lon"].astype(float).to_numpy()[pos])
        if len(np.unique(elev)) < 2:
            continue
        t_peak_par = times[int(np.nanargmax(day_par))]
        t_noon = times[int(np.argmax(elev))]
        offs.append((t_peak_par - t_noon).total_seconds() / 3600.0)
    if not offs:
        return np.nan
    return float(np.median(offs))



def smooth_cruise_par(
    df,
    cruise_col="cruise",
    cruise_day_col="cruise_day",
    datetime_col="time",
    norm_par_col="norm_par",
    par_col="par",
    lat_col="lat",
    lon_col="lon",
    output_col="norm_par_smooth",
    freq="30min",
    fill_grid=True,
    window_length=9,
    polyorder=2,
    remove_outliers=True,
    outlier_n_mads=6.0,
    min_outlier_points=20,
    min_clearsky_for_ratio=100.0,
    night_elevation_deg=-6.0,
    clearsky_coefficient=2200.0,
    clearsky_index_fallback=0.55,
    scale_model_to_observations=True,
):
    """
    Process one cruise's PAR time series.

    Processing order
    ----------------
    1. Use norm_par where available, otherwise par.
    2. Optionally remove high outliers using a cruise-wide
       median/MAD criterion on observed / clear-sky PAR.
    3. Fill missing/rejected values using the clear-sky model.
    4. Smooth the entire cruise using Savitzky-Golay.
    5. Set final smoothed PAR to zero at solar elevation <= -6 deg.

    Requirements
    ------------
    - One row per timestamp within a cruise.
    - Existing solar_elevation() and clearsky_par() helpers.
    - PAR observations and modeled PAR must use compatible units.
    - Naive timestamps are interpreted as UTC.
    - No automatic clock shift is applied.

    Notes
    -----
    Original norm_par and par measurements are not overwritten.
    time_of_day is neither modified nor used to force zeros.
    Only HIGH ratio outliers are removed; low cloud-attenuated
    observations are retained.
    """
    required = {
        cruise_col,
        datetime_col,
        lat_col,
        lon_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("The input dataframe is empty.")

    if (
        norm_par_col not in df.columns
        and par_col not in df.columns
    ):
        raise ValueError(
            f"At least one of {norm_par_col!r} or "
            f"{par_col!r} must be present."
        )

    if df[cruise_col].isna().any():
        raise ValueError("Cruise identifiers cannot be missing.")

    if df[cruise_col].nunique() != 1:
        raise ValueError(
            "Pass one cruise at a time to smooth_cruise_par()."
        )

    if (
        not isinstance(window_length, (int, np.integer))
        or window_length < 1
        or window_length % 2 == 0
    ):
        raise ValueError(
            "window_length must be a positive odd integer."
        )

    if (
        not isinstance(polyorder, (int, np.integer))
        or polyorder < 0
        or polyorder >= window_length
    ):
        raise ValueError(
            "polyorder must be a nonnegative integer "
            "smaller than window_length."
        )

    if outlier_n_mads <= 0:
        raise ValueError("outlier_n_mads must be positive.")

    if (
        not np.isfinite(clearsky_index_fallback)
        or clearsky_index_fallback < 0
    ):
        raise ValueError(
            "clearsky_index_fallback must be finite and nonnegative."
        )

    step = pd.Timedelta(freq)

    if step <= pd.Timedelta(0):
        raise ValueError("freq must be a positive fixed interval.")

    out = df.copy()
    cruise = out[cruise_col].iloc[0]

    out[datetime_col] = pd.to_datetime(
        out[datetime_col],
        utc=True,
        errors="raise",
    )

    if out[datetime_col].isna().any():
        raise ValueError("Missing timestamps found.")

    out = (
        out.sort_values(datetime_col)
        .reset_index(drop=True)
    )

    repeated = out[datetime_col].duplicated(keep=False)

    if repeated.any():
        raise ValueError(
            f"{cruise}: repeated timestamps found. "
            "Provide one PAR record per cruise/time:\n"
            + out.loc[
                repeated,
                [cruise_col, datetime_col],
            ].head(20).to_string(index=False)
        )

    # ----------------------------------------------------------
    # Regular time grid.
    # ----------------------------------------------------------
    first_time = out[datetime_col].min()
    last_time = out[datetime_col].max()

    off_grid = (
        (out[datetime_col] - first_time) % step
    ) != pd.Timedelta(0)

    if off_grid.any():
        raise ValueError(
            f"{cruise}: timestamps are not aligned to the "
            f"{freq} grid anchored at {first_time}."
        )

    out["_input_row_present"] = True

    if fill_grid:
        full_grid = pd.date_range(
            first_time,
            last_time,
            freq=freq,
            name=datetime_col,
        )

        out = (
            out.set_index(datetime_col)
            .reindex(full_grid)
            .reset_index()
        )

    elif len(out) > 1:
        intervals = out[datetime_col].diff().dropna()

        if not intervals.eq(step).all():
            raise ValueError(
                "Savitzky-Golay input must be regularly spaced. "
                "Use fill_grid=True to add missing timestamps."
            )

    out["grid_filled"] = out["_input_row_present"].isna()
    out = out.drop(columns="_input_row_present")

    out[cruise_col] = cruise

    # Preserve existing cruise_day values. Do not forward-fill
    # day labels into inserted rows across possible boundaries.
    # The smoother itself does not depend on cruise_day.

    # ----------------------------------------------------------
    # Coordinates for solar geometry.
    # Preserve original columns; use separate filled coordinates.
    # ----------------------------------------------------------
    for column in [lat_col, lon_col]:
        values = pd.to_numeric(
            out[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        coordinate_series = pd.Series(
            values.to_numpy(),
            index=pd.DatetimeIndex(out[datetime_col]),
        )

        coordinate_series = (
            coordinate_series
            .interpolate(method="time", limit_area="inside")
            .ffill()
            .bfill()
        )

        out[f"{column}_for_solar"] = (
            coordinate_series.to_numpy()
        )

    if out[
        [f"{lat_col}_for_solar", f"{lon_col}_for_solar"]
    ].isna().any().any():
        raise ValueError(
            f"{cruise}: insufficient coordinates "
            "to calculate solar elevation."
        )

    times = pd.DatetimeIndex(out[datetime_col])

    out["solar_elev"] = np.asarray(
        solar_elevation(
            times,
            out[f"{lat_col}_for_solar"].to_numpy(),
            out[f"{lon_col}_for_solar"].to_numpy(),
        ),
        dtype=float,
    )

    out["clearsky"] = np.asarray(
        clearsky_par(
            out["solar_elev"].to_numpy(),
            clearsky_coefficient,
        ),
        dtype=float,
    )

    out["clearsky"] = out["clearsky"].clip(lower=0)

    if not np.isfinite(
        out[["solar_elev", "clearsky"]].to_numpy()
    ).all():
        raise ValueError(
            f"{cruise}: solar geometry/model returned "
            "nonfinite values."
        )

    # No clock correction is performed.
    out["time_shift_h"] = 0.0

    # ----------------------------------------------------------
    # Select measurements: norm_par -> par.
    # ----------------------------------------------------------
    def numeric_column(column):
        if column not in out.columns:
            return pd.Series(
                np.nan, index=out.index, dtype=float
            )

        return pd.to_numeric(
            out[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    norm_par = numeric_column(norm_par_col)
    raw_par = numeric_column(par_col)

    out["par_selected"] = norm_par.combine_first(raw_par)

    out["par_input_source"] = np.select(
        [
            norm_par.notna(),
            norm_par.isna() & raw_par.notna(),
        ],
        ["norm_par", "par"],
        default="missing",
    )

    # ----------------------------------------------------------
    # Cruise-wide outlier criterion.
    #
    # Use only sufficiently illuminated observations for ratios.
    # Remove high outliers only, not low cloud-related values.
    # ----------------------------------------------------------
    out["par_ratio"] = np.nan

    ratio_eligible = (
        out["par_selected"].notna()
        & out["clearsky"].gt(min_clearsky_for_ratio)
    )

    out.loc[ratio_eligible, "par_ratio"] = (
        out.loc[ratio_eligible, "par_selected"]
        / out.loc[ratio_eligible, "clearsky"]
    )

    ratios = out.loc[ratio_eligible, "par_ratio"]

    out["par_outlier"] = False
    ratio_upper_limit = np.nan

    if remove_outliers and len(ratios) >= min_outlier_points:
        ratio_median = float(ratios.median())

        mad = float(
            (ratios - ratio_median).abs().median()
        )

        robust_scale = 1.4826 * mad

        if np.isfinite(robust_scale) and robust_scale > 0:
            ratio_upper_limit = (
                ratio_median
                + outlier_n_mads * robust_scale
            )

            out["par_outlier"] = (
                ratio_eligible
                & out["par_ratio"].gt(ratio_upper_limit)
            )

    # If there are no qualifying outliers, nothing is removed.
    out["par_ratio_upper_limit"] = ratio_upper_limit

    out["par_clean"] = out["par_selected"].mask(
        out["par_outlier"]
    )

    # ----------------------------------------------------------
    # Clear-sky model for missing/rejected measurements.
    # ----------------------------------------------------------
    clean_ratio_mask = (
        out["par_clean"].notna()
        & out["par_clean"].ge(0)
        & out["clearsky"].gt(min_clearsky_for_ratio)
    )

    model_scale = clearsky_index_fallback

    if scale_model_to_observations:
        clean_ratios = (
            out.loc[clean_ratio_mask, "par_clean"]
            / out.loc[clean_ratio_mask, "clearsky"]
        )

        if len(clean_ratios) >= 3:
            candidate = float(clean_ratios.median())

            if np.isfinite(candidate) and candidate > 0:
                model_scale = candidate
    else:
        # Use the unscaled clear-sky model.
        model_scale = 1.0

    out["model_scale"] = model_scale

    out["par_model"] = out["clearsky"] * model_scale

    out["par_imputed"] = out["par_clean"].isna()

    out["par_source"] = out["par_input_source"].copy()
    out.loc[out["par_imputed"], "par_source"] = (
        "clearsky_model"
    )

    # Model filling occurs before smoothing; no night zeroing yet.
    out["par_filled"] = out["par_clean"].combine_first(
        out["par_model"]
    )

    # ----------------------------------------------------------
    # Savitzky-Golay across the ENTIRE cruise.
    # ----------------------------------------------------------
    values = out["par_filled"].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{cruise}: nonfinite values remain before smoothing."
        )

    actual_window = min(window_length, len(values))

    if actual_window % 2 == 0:
        actual_window -= 1

    if actual_window > polyorder:
        smoothed = savgol_filter(
            values,
            window_length=actual_window,
            polyorder=polyorder,
            mode="interp",
        )

        out["sg_applied"] = True
    else:
        smoothed = values.copy()
        out["sg_applied"] = False

    out["sg_window_used"] = actual_window

    # Preserve the unconstrained filter output for diagnostics.
    out["par_smooth_before_zero"] = smoothed

    # ----------------------------------------------------------
    # Apply physical constraints AFTER smoothing.
    # ----------------------------------------------------------
    out["solar_night"] = out["solar_elev"].le(
        night_elevation_deg
    )

    # Diagnostic flag, not a replacement for checking clock errors.
    out["high_par_during_solar_night"] = (
        out["solar_night"]
        & out["par_selected"].gt(100)
    )

    out[output_col] = np.clip(smoothed, 0.0, None)

    out.loc[out["solar_night"], output_col] = 0.0

    out["par_forced_zero"] = out["solar_night"]

    return out


# def smooth_daily_par(
#     df,
#     cruise_col="cruise",
#     cruise_day_col="cruise_day",
#     datetime_col="time",
#     par_col="norm_par",
#     time_of_day_col="time_of_day",
#     lat_col="lat",
#     lon_col="lon",
#     min_par=5.0,
#     window_length=5,          # Savitzky-Golay window in samples (5 = 2.5 h)
#     polyorder=2,              # 2 preserves the dawn-to-noon slope
#     output_col="norm_par_smooth",
#     outlier_window=13,        # Hampel window in samples (13 = 6.5 h)
#     outlier_n_mads=6.0,       # conservative: clouds are signal, not outliers
#     max_interp_gap="2h",      # linear interpolation only up to this gap
#     dark_correct=True,        # subtract cruise nighttime median (sensor offset)
#     align_timestamps=True,    # shift mislabeled clocks onto the solar clock
#     recompute_time_of_day=False,  # relabel day/night from solar elevation
#     fill_grid=True,           # reindex to the full 30-min grid
#     clearsky_coefficient=2200.0,
#     clearsky_index_fallback=0.55,
#     offset_threshold_h=2.0,
# ):
#     """
#     Clean, smooth, and impute PAR for one cruise (or cruise_day group).

#     Columns added to the output:
#       solar_elev        : solar elevation (degrees)
#       par_clean          : after dark correction, outlier removal
#       par_filled         : after gap imputation and night/low zeroing
#       norm_par_smooth    : final smoothed values (night = 0)
#       par_source         : observed | interpolated | clearsky_model |
#                           imputed_grid | forced_zero
#       par_outlier        : True where an outlier was removed
#       par_imputed        : True where a value was imputed
#       time_shift_h       : clock correction applied to `time` (0 if none)
#     """
#     out = df.copy()
#     out[datetime_col] = pd.to_datetime(out[datetime_col])
#     out = out.sort_values([cruise_day_col, datetime_col]).copy()

#     par = pd.to_numeric(out[par_col], errors="coerce")

#     # -- solar geometry ------------------------------------------------------
#     def _geom(frame):
#         lat = frame[lat_col].astype(float).interpolate().ffill().bfill() \
#             if lat_col in frame else None
#         lon = frame[lon_col].astype(float).interpolate().ffill().bfill() \
#             if lon_col in frame else None
#         return solar_elevation(frame[datetime_col], lat, lon)

#     if lat_col in out and lon_col in out and len(out):
#         out["solar_elev"] = _geom(out)
#     else:
#         out["solar_elev"] = np.nan
#     out["clearsky"] = clearsky_par(out["solar_elev"], clearsky_coefficient)

#     # -- clock alignment ------------------------------------------------------
#     out["time_shift_h"] = 0.0
#     if align_timestamps and lat_col in out:
#         offset = _detect_clock_offset(out, par_col, datetime_col,
#                                       cruise_day_col, offset_threshold_h)
#         if np.isfinite(offset) and abs(offset) > offset_threshold_h:
#             # round to the nearest half hour; shift the clock back onto
#             # the solar clock
#             shift = -round(offset * 2) / 2.0
#             out[datetime_col] = out[datetime_col] - pd.to_timedelta(shift, unit="h")
#             out["time_shift_h"] = shift
#             out["solar_elev"] = _geom(out)
#             out["clearsky"] = clearsky_par(out["solar_elev"],
#                                            clearsky_coefficient)
#             # the original day/night labels were computed on the wrong
#             # clock; they must be re-derived from solar elevation
#             recompute_time_of_day = True

#     # cruises with no usable PAR are driven entirely by the solar model:
#     # re-derive the labels so they are consistent with it
#     par_check = pd.to_numeric(out[par_col], errors="coerce")
#     if par_check.notna().sum() < 6 or (par_check.max() if par_check.notna().any() else 0) < 100:
#         recompute_time_of_day = True

#     # -- recompute time_of_day if requested ------------------------------------
#     if recompute_time_of_day:
#         elev = out["solar_elev"].to_numpy(dtype=float)
#         t_next = pd.DatetimeIndex(out[datetime_col]) + pd.Timedelta("1h")
#         elev_next = solar_elevation(t_next, out[lat_col].astype(float),
#                                    out[lon_col].astype(float))
#         out[time_of_day_col] = time_of_day_from_elevation(elev, elev_next)

#     time_label = out[time_of_day_col].astype(str).str.lower().str.strip()
#     is_night = time_label.eq("night")

#     # -- dark-offset correction -------------------------------------------------
#     if dark_correct and is_night.any():
#         dark = float(np.nanmedian(par[is_night]))
#         if 0 < dark < 10:  # plausible sensor dark offset, not a label problem
#             par = (par - dark).clip(lower=0)
#             out[par_col] = par

#     # -- outlier removal (within cruise_day) ------------------------------------
#     out["par_outlier"] = False
#     out["par_clean"] = par
#     cs = out["clearsky"].to_numpy(dtype=float)
#     par_arr = out["par_clean"].to_numpy(dtype=float).copy()
#     for cd, pos in out.groupby(cruise_day_col, sort=False).indices.items():
#         y = par_arr[pos]
#         # Hampel on the clear-sky ratio (quasi-stationary), not raw PAR,
#         # so the dawn-to-noon trend is never itself flagged.
#         ratio = np.where(cs[pos] > 50, y / cs[pos], np.nan)
#         bad = (_hampel_mask(ratio, outlier_window, outlier_n_mads)
#                if len(y) >= outlier_window else np.zeros(len(y), dtype=bool))
#         # NOTE: isolated 30-min spikes that stay within the clear-sky
#         # envelope are usually broken-cloud enhancement, not sensor
#         # glitches, and are left in (Savitzky-Golay absorbs them).
#         # physically impossible (daylight only): above the clear-sky envelope
#         bad |= (cs[pos] > 50) & (y > 1.5 * cs[pos])
#         bad = np.where(np.isnan(y), False, bad)  # missing data is not an outlier
#         out.iloc[pos, out.columns.get_loc("par_outlier")] = bad
#         y[bad] = np.nan
#         out.iloc[pos, out.columns.get_loc("par_clean")] = y
#         par_arr[pos] = y

#     # -- gaps: reindex to the 30-min grid ---------------------------------------
#     out["par_source"] = np.where(out["par_clean"].isna(), "missing", "observed")
#     out["par_imputed"] = False
#     out["grid_filled"] = False

#     if fill_grid and len(out) >= 2:
#         full_idx = pd.date_range(out[datetime_col].min(),
#                                  out[datetime_col].max(), freq="30min")
#         reindexed = out.set_index(datetime_col).reindex(full_idx)
#         new_rows = reindexed.index.difference(pd.DatetimeIndex(out[datetime_col]))
#         if len(new_rows):
#             for c in [cruise_col, cruise_day_col, time_of_day_col, "time_shift_h"]:
#                 reindexed[c] = reindexed[c].ffill().bfill()
#             for c in [lat_col, lon_col]:
#                 reindexed[c] = reindexed[c].interpolate().ffill().bfill()
#             reindexed["solar_elev"] = solar_elevation(
#                 reindexed.index, reindexed[lat_col], reindexed[lon_col])
#             reindexed["clearsky"] = clearsky_par(
#                 reindexed["solar_elev"], clearsky_coefficient)
#             for c in ["par_source", "par_outlier", "par_clean"]:
#                 reindexed[c] = reindexed[c]
#             reindexed["par_source"] = reindexed["par_source"].fillna("imputed_grid")
#             reindexed["par_imputed"] = reindexed["par_imputed"].fillna(True)
#             reindexed["par_outlier"] = reindexed["par_outlier"].fillna(False)
#             reindexed["grid_filled"] = reindexed["grid_filled"].fillna(True)
#             out = reindexed.reset_index(names=datetime_col)
#             if recompute_time_of_day:
#                 elev = out["solar_elev"].to_numpy(dtype=float)
#                 t_next = pd.DatetimeIndex(out[datetime_col]) + pd.Timedelta("1h")
#                 elev_next = solar_elevation(t_next, out[lat_col].astype(float),
#                                             out[lon_col].astype(float))
#                 out[time_of_day_col] = time_of_day_from_elevation(elev, elev_next)
#             time_label = out[time_of_day_col].astype(str).str.lower()
#             is_night = time_label.eq("night")

#     # -- imputation: interpolate short gaps, clear-sky model long ones ----------
#     ts = pd.Series(out["par_clean"].to_numpy(dtype=float),
#                    index=pd.DatetimeIndex(out[datetime_col]))
#     interp_limit = int(pd.Timedelta(max_interp_gap) / pd.Timedelta("30min"))
#     interp = ts.interpolate(method="time", limit_area="inside",
#                             limit=interp_limit)

#     # clear-sky index for scaling the model: day-level -> fallback
#     cs_now = out["clearsky"].to_numpy(dtype=float)
#     daylight_obs = ts.notna().to_numpy() & (cs_now > 100)
#     if daylight_obs.sum() >= 3:
#         csi = float(np.nanmedian(
#             ts.to_numpy()[daylight_obs] / cs_now[daylight_obs]))
#     else:
#         csi = clearsky_index_fallback
#     csi = float(np.clip(csi, 0.05, 1.0)) if np.isfinite(csi) else clearsky_index_fallback

#     imputed_values = cs_now * csi
#     final = np.where(ts.notna(), ts.to_numpy(),
#                      np.where(interp.notna(), interp.to_numpy(), imputed_values))

#     # relabel imputed rows with what filled them
#     src = out["par_source"].to_numpy().astype(object)
#     src[ts.isna().to_numpy() & interp.notna().to_numpy()] = "interpolated"
#     src[ts.isna().to_numpy() & interp.isna().to_numpy()] = "clearsky_model"
#     out["par_source"] = src
#     out["par_imputed"] = out["par_imputed"] | ts.isna().to_numpy()

#     # -- night = 0, low values = 0 ------------------------------------------------
#     is_night_arr = is_night.to_numpy()
#     below = final < min_par
#     final = np.where(is_night_arr | below, 0.0, final)
#     src = out["par_source"].to_numpy().astype(object)
#     src[is_night_arr | below] = np.where(
#         np.isin(src[is_night_arr | below], ["observed", "missing"]),
#         "forced_zero", src[is_night_arr | below])
#     out["par_source"] = src
#     out["par_filled"] = final

#     # -- Savitzky-Golay over each cruise_day --------------------------------------
#     out[output_col] = 0.0
#     for cd, pos in out.groupby(cruise_day_col, sort=False).indices.items():
#         y = final[pos]
#         this_window = min(window_length, len(y))
#         if this_window % 2 == 0:
#             this_window -= 1
#         if this_window > polyorder:
#             smoothed = savgol_filter(y, window_length=this_window,
#                                      polyorder=polyorder, mode="interp")
#         else:
#             smoothed = y
#         out.iloc[pos, out.columns.get_loc(output_col)] = np.clip(
#             smoothed, 0, None)

#     # re-enforce constraints after smoothing
#     out.iloc[np.where(is_night_arr)[0], out.columns.get_loc(output_col)] = 0.0
#     out.iloc[np.where(below)[0], out.columns.get_loc(output_col)] = 0.0

#     return out
