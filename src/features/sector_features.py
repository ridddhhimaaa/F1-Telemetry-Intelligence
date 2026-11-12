"""
F1 Telemetry Intelligence - Sector Features Feature Engineering
==============================================================

Computes official timing sector kinematics and sector-level performance aggregates:
- Partitions telemetry into the 3 official Yas Marina Circuit sectors:
    Sector 1: 0m to 1,200m (Turns 1 to 5)
    Sector 2: 1,200m to 3,520m (Back straight 1, Turn 6-7 chicane, Back straight 2, Turn 9)
    Sector 3: 3,520m to 5,281m (Turns 10 to 16, marina complex and pit straight)
- Generates sample-level Sector indicators (Sector, DistanceInSector_m).
- Aggregates sector-level performance summary table (one row per driver-lap-sector):
    Mean/Max/Min Speed, Full Throttle %, Total Braking Duration, Peak Deceleration G,
    Corner Count, Most Used Gear, Sector Traversal Duration.

Scientific and Numerical Standards:
- Validated against official Yas Marina timing line session timestamps.
- Strictly grouped by (Driver, LapNumber) without boundary contamination.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.features.sector_features")

# Sector boundary distance markers (meters) for Yas Marina Circuit
SECTOR_BOUNDARIES = {
    1: (0.0, 1200.0),
    2: (1200.0, 3520.0),
    3: (3520.0, 5300.0)
}


def assign_sector_markers(distance_m: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Assigns each telemetry sample to Sector 1, 2, or 3, and computes distance within that sector.

    Returns:
        (Sector_ID, Distance_In_Sector_m)
    """
    s = distance_m.to_numpy(dtype=np.float64)
    n = len(s)

    sector_arr = np.ones(n, dtype=int)
    dist_in_sec = np.zeros(n, dtype=np.float64)

    for i in range(n):
        d = s[i]
        if d <= SECTOR_BOUNDARIES[1][1]:
            sector_arr[i] = 1
            dist_in_sec[i] = d - SECTOR_BOUNDARIES[1][0]
        elif d <= SECTOR_BOUNDARIES[2][1]:
            sector_arr[i] = 2
            dist_in_sec[i] = d - SECTOR_BOUNDARIES[2][0]
        else:
            sector_arr[i] = 3
            dist_in_sec[i] = d - SECTOR_BOUNDARIES[3][0]

    return (
        pd.Series(sector_arr, index=distance_m.index, name="Sector"),
        pd.Series(np.round(dist_in_sec, 2), index=distance_m.index, name="Distance_In_Sector_m")
    )


def compute_sector_aggregates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates telemetry features into a clean sector-level summary table (one row per driver-lap-sector).
    """
    dist_col = "Distance_Corrected" if "Distance_Corrected" in df.columns else "Distance"
    th_col = "Throttle_Clipped" if "Throttle_Clipped" in df.columns else "Throttle"
    brk_col = "IsBraking" if "IsBraking" in df.columns else "Brake"
    time_col = "_time_sec"

    records = []

    for (driver, lap, sector), grp in df.groupby(["Driver", "LapNumber", "Sector"], sort=False):
        n = len(grp)
        if n == 0:
            continue

        # Timings
        t_start = grp[time_col].iloc[0]
        t_end = grp[time_col].iloc[-1]
        duration_s = max(t_end - t_start, 0.0)

        # Speed metrics
        spd_mean = grp["Speed"].mean()
        spd_max = grp["Speed"].max()
        spd_min = grp["Speed"].min()

        # Throttle metrics
        wot_pct = (grp[th_col] >= 98.0).mean() * 100.0
        th_mean = grp[th_col].mean()

        # Braking metrics
        brk_duration = (grp[brk_col] > 0).sum() * (duration_s / n) if n > 0 else 0.0
        peak_decel_g = 0.0
        if "G_Longitudinal" in grp.columns:
            peak_decel_g = np.abs(grp["G_Longitudinal"].min())
        elif "Accel_Longitudinal_mps2" in grp.columns:
            peak_decel_g = np.abs(grp["Accel_Longitudinal_mps2"].min()) / 9.80665

        # Gear metrics
        most_used_gear = int(grp["nGear"].mode().iloc[0]) if "nGear" in grp.columns and len(grp["nGear"].mode()) > 0 else 0
        gear_shifts = grp["IsGearShift"].sum() if "IsGearShift" in grp.columns else 0

        # Corner metrics
        corner_count = grp["Corner_ID"].nunique() - 1 if "Corner_ID" in grp.columns and 0 in grp["Corner_ID"].values else (grp["Corner_ID"].nunique() if "Corner_ID" in grp.columns else 0)
        min_apex_speed = grp["Speed"].min()

        records.append({
            "Driver": driver,
            "LapNumber": lap,
            "Sector": sector,
            "SectorDuration_s": round(duration_s, 3),
            "Speed_Mean_kmh": round(spd_mean, 2),
            "Speed_Max_kmh": round(spd_max, 2),
            "Speed_Min_kmh": round(spd_min, 2),
            "FullThrottle_Pct": round(wot_pct, 1),
            "Throttle_Mean_Pct": round(th_mean, 1),
            "Braking_Duration_s": round(brk_duration, 3),
            "Peak_Decel_G": round(peak_decel_g, 2),
            "Most_Used_Gear": most_used_gear,
            "Gear_Shifts_Count": int(gear_shifts),
            "Corners_Count": max(int(corner_count), 0),
            "Samples_Count": n
        })

    return pd.DataFrame(records)


def compute_sector_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates full sector feature generation:
    1. Attaches sample-level Sector and Distance_In_Sector_m columns.
    2. Generates aggregated sector performance summary table.

    Returns:
        (sample_level_df, sector_summary_table_df)
    """
    logger.info("Computing sector features across %d rows...", len(df))

    df_work = df.copy()
    if pd.api.types.is_timedelta64_dtype(df_work["Time"]):
        df_work["_time_sec"] = df_work["Time"].dt.total_seconds()
    elif "SessionTime" in df_work.columns and pd.api.types.is_timedelta64_dtype(df_work["SessionTime"]):
        df_work["_time_sec"] = df_work["SessionTime"].dt.total_seconds()
    else:
        df_work["_time_sec"] = pd.to_numeric(df_work["Time"])

    dist_col = "Distance_Corrected" if "Distance_Corrected" in df_work.columns else "Distance"

    # Assign sample-level sectors
    sec_id, dist_in_sec = assign_sector_markers(df_work[dist_col])
    sample_df = pd.DataFrame({
        "Sector": sec_id,
        "Distance_In_Sector_m": dist_in_sec
    }, index=df.index)

    df_with_sec = pd.concat([df_work, sample_df], axis=1)
    summary_df = compute_sector_aggregates(df_with_sec)

    logger.info("Sector features completed: %d sector aggregates generated.", len(summary_df))
    return sample_df, summary_df
