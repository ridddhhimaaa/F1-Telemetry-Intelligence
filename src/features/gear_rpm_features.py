"""
F1 Telemetry Intelligence - Transmission and Engine Feature Engineering
=======================================================================

Computes drivetrain, transmission shift events, and power unit dynamics:
- Shift event detection: Upshifts (Delta nGear > 0), Downshifts (Delta nGear < 0).
- Running cumulative gear shifts, upshift count, and downshift count per lap.
- Engine rev rate: d(RPM)/dt (RPM/s) using actual timestamp differences.
- Mechanical drivetrain ratio: RPM / Speed (revs per km/h).
- Engine RPM utilization: RPM / 13,000 (percentage of maximum power band).
- Shift optimization indicators and gear transition flags.

Scientific and Numerical Standards:
- Prevents division-by-zero at standstill by clamping speed >= 5.0 km/h.
- Strictly isolated by (Driver, LapNumber) without inter-lap shift carryover.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.features.gear_rpm_features")
MAX_RPM_LIMITER = 13000.0  # Nominal maximum F1 1.6L Turbo Hybrid ICE RPM limiter


def detect_shift_events(
    gear_series: pd.Series,
    time_sec: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Detects gear changes, upshifts, downshifts, and running cumulative shift counts.

    Returns:
        (IsGearShift, IsUpshift, IsDownshift, Cumulative_Shifts, Cumulative_Upshifts, Cumulative_Downshifts)
    """
    g = gear_series.to_numpy(dtype=int)
    n = len(g)

    is_shift = np.zeros(n, dtype=int)
    is_up = np.zeros(n, dtype=int)
    is_down = np.zeros(n, dtype=int)
    cum_shifts = np.zeros(n, dtype=int)
    cum_up = np.zeros(n, dtype=int)
    cum_down = np.zeros(n, dtype=int)

    shifts_count = 0
    upshifts_count = 0
    downshifts_count = 0

    for i in range(1, n):
        dg = g[i] - g[i - 1]
        if dg != 0:
            is_shift[i] = 1
            shifts_count += 1
            if dg > 0:
                is_up[i] = 1
                upshifts_count += 1
            else:
                is_down[i] = 1
                downshifts_count += 1

        cum_shifts[i] = shifts_count
        cum_up[i] = upshifts_count
        cum_down[i] = downshifts_count

    return (
        pd.Series(is_shift, index=gear_series.index, name="IsGearShift"),
        pd.Series(is_up, index=gear_series.index, name="IsUpshift"),
        pd.Series(is_down, index=gear_series.index, name="IsDownshift"),
        pd.Series(cum_shifts, index=gear_series.index, name="Gear_Change_Count"),
        pd.Series(cum_up, index=gear_series.index, name="Upshift_Count"),
        pd.Series(cum_down, index=gear_series.index, name="Downshift_Count")
    )


def compute_rpm_derivatives(
    rpm_series: pd.Series,
    time_sec: pd.Series,
    min_dt: float = 0.005
) -> pd.Series:
    """
    Computes engine acceleration / RPM rate of change d(RPM)/dt in RPM/s.
    """
    rpm = rpm_series.to_numpy(dtype=np.float64)
    t = time_sec.to_numpy(dtype=np.float64)
    n = len(rpm)

    if n < 2:
        return pd.Series(np.zeros(n), index=rpm_series.index, name="RPM_Change_Rate_per_s")

    dt = np.maximum(np.diff(t), min_dt)
    drpm = np.diff(rpm)

    deriv = np.empty(n, dtype=np.float64)
    deriv[0] = drpm[0] / dt[0]
    deriv[1:] = drpm / dt

    # Clip to physical engine rev limits (+/- 15,000 RPM/s)
    deriv = np.clip(deriv, -15000.0, 15000.0)

    return pd.Series(np.round(deriv, 1), index=rpm_series.index, name="RPM_Change_Rate_per_s")


def compute_gear_rpm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates full gear and RPM feature generation grouped strictly by (Driver, LapNumber).

    Input columns required:
        - Driver, LapNumber, nGear, RPM, Speed, Time (or SessionTime)

    Generated features:
        - Ratio_RPM_per_kmh: Drivetrain ratio index (RPM / Speed).
        - RPM_Band_Pct: Engine RPM utilization (RPM / 13,000 * 100).
        - RPM_Change_Rate_per_s: Engine rotational acceleration in RPM/s.
        - IsGearShift: Binary indicator (1 if gear changed on this tick).
        - IsUpshift: Binary indicator (1 on upshift).
        - IsDownshift: Binary indicator (1 on downshift).
        - Gear_Change_Count: Cumulative total gear shifts in lap.
        - Upshift_Count: Cumulative upshift count in lap.
        - Downshift_Count: Cumulative downshift count in lap.
    """
    logger.info("Computing gear and RPM features across %d rows...", len(df))

    df_work = df.copy()
    if pd.api.types.is_timedelta64_dtype(df_work["Time"]):
        df_work["_time_sec"] = df_work["Time"].dt.total_seconds()
    elif "SessionTime" in df_work.columns and pd.api.types.is_timedelta64_dtype(df_work["SessionTime"]):
        df_work["_time_sec"] = df_work["SessionTime"].dt.total_seconds()
    else:
        df_work["_time_sec"] = pd.to_numeric(df_work["Time"])

    feature_dfs = []

    for (driver, lap), grp in df_work.groupby(["Driver", "LapNumber"], sort=False):
        grp_res = pd.DataFrame(index=grp.index)

        # 1. Drivetrain ratio proxy (clamped speed >= 5 km/h to prevent division by zero)
        spd_clamped = np.maximum(grp["Speed"].to_numpy(dtype=np.float64), 5.0)
        grp_res["Ratio_RPM_per_kmh"] = np.round(grp["RPM"].to_numpy(dtype=np.float64) / spd_clamped, 2)

        # 2. RPM powerband utilization
        grp_res["RPM_Band_Pct"] = np.round(
            (grp["RPM"].to_numpy(dtype=np.float64) / MAX_RPM_LIMITER) * 100.0, 1
        )

        # 3. RPM derivative
        grp_res["RPM_Change_Rate_per_s"] = compute_rpm_derivatives(grp["RPM"], grp["_time_sec"])

        # 4. Shift detection and cumulative counts
        is_shift, is_up, is_down, cum_s, cum_up, cum_down = detect_shift_events(
            grp["nGear"], grp["_time_sec"]
        )
        grp_res["IsGearShift"] = is_shift
        grp_res["IsUpshift"] = is_up
        grp_res["IsDownshift"] = is_down
        grp_res["Gear_Change_Count"] = cum_s
        grp_res["Upshift_Count"] = cum_up
        grp_res["Downshift_Count"] = cum_down

        feature_dfs.append(grp_res)

    result_df = pd.concat(feature_dfs).loc[df.index]
    logger.info("Gear and RPM features completed successfully.")
    return result_df
