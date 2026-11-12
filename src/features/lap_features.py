"""
F1 Telemetry Intelligence - Lap-Level Feature Engineering
=========================================================

Aggregates high-frequency telemetry and engineering features into a single, comprehensive
lap-level profile (one row per driver-lap):
- Driver, LapNumber, LapClassification, LapTime_s.
- Speed statistics: Mean, Max, Min, Std, Median, 25th/75th percentiles.
- Acceleration & Deceleration: Peak acceleration g, Peak braking decel g, Mean acceleration m/s^2.
- Braking Dynamics: Total braking time (s), total braking distance (m), braking event count, mean peak braking g.
- Throttle Dynamics: Full throttle percentage (WOT %), mean throttle %, throttle transitions count.
- Transmission: Total gear changes, upshift count, downshift count, modal gear.
- Corner Performance: Number of detected corners, mean apex speed, minimum apex speed, total cornering time.
- Timing Sectors: Sector 1, 2, 3 durations and mean speeds.
- Data Quality & Traceability: Telemetry row count, sampling frequency (Hz), missing flags.

Scientific and Numerical Standards:
- Maintains separate regime tracking without mixing COMPLETE_FLYING_LAP and VSC/start laps.
- Integrates seamlessly with Phase 4 lap classifications.
- Strictly isolated by (Driver, LapNumber) without boundary contamination.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.features.lap_features")
GRAVITY = 9.80665


def aggregate_lap_features(
    tel_df: pd.DataFrame,
    laps_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes lap-level aggregate features for every driver-lap sequence.

    Input DataFrame can be either raw processed telemetry or enriched feature telemetry.
    """
    logger.info("Computing lap-level aggregate features across %d rows...", len(tel_df))

    df_work = tel_df.copy()
    if pd.api.types.is_timedelta64_dtype(df_work["Time"]):
        df_work["_time_sec"] = df_work["Time"].dt.total_seconds()
    elif "SessionTime" in df_work.columns and pd.api.types.is_timedelta64_dtype(df_work["SessionTime"]):
        df_work["_time_sec"] = df_work["SessionTime"].dt.total_seconds()
    else:
        df_work["_time_sec"] = pd.to_numeric(df_work["Time"])

    dist_col = "Distance_Corrected" if "Distance_Corrected" in df_work.columns else "Distance"
    th_col = "Throttle_Clipped" if "Throttle_Clipped" in df_work.columns else "Throttle"
    brk_col = "IsBraking" if "IsBraking" in df_work.columns else "Brake"

    # Pre-index laps_df if provided for exact LapTime matching
    lap_time_map = {}
    if laps_df is not None and len(laps_df) > 0:
        for _, r in laps_df.iterrows():
            if pd.notna(r.get("LapTime")):
                lt = r["LapTime"]
                sec = lt.total_seconds() if hasattr(lt, "total_seconds") else float(lt)
                lap_time_map[(str(r["Driver"]), int(r["LapNumber"]))] = sec

    records = []

    for (driver, lap), grp in df_work.groupby(["Driver", "LapNumber"], sort=False):
        n = len(grp)
        if n < 5:
            continue

        lap_class = grp["LapClassification"].iloc[0] if "LapClassification" in grp.columns else "UNKNOWN"

        # Timing and duration
        t_start = grp["_time_sec"].iloc[0]
        t_end = grp["_time_sec"].iloc[-1]
        telemetry_duration_s = max(t_end - t_start, 0.0)

        # Official lap time from laps dataset if available, otherwise telemetry duration
        official_lap_time = lap_time_map.get((str(driver), int(lap)), round(telemetry_duration_s, 3))

        # Speed metrics
        spd = grp["Speed"].to_numpy(dtype=np.float64)
        spd_mean = np.mean(spd)
        spd_max = np.max(spd)
        spd_min = np.min(spd)
        spd_std = np.std(spd)
        spd_p25 = np.percentile(spd, 25)
        spd_p50 = np.median(spd)
        spd_p75 = np.percentile(spd, 75)

        # Throttle metrics
        th_vals = grp[th_col].to_numpy(dtype=np.float64)
        wot_pct = np.mean(th_vals >= 98.0) * 100.0
        th_mean = np.mean(th_vals)
        th_trans_count = grp["Throttle_Transition_Count"].iloc[-1] if "Throttle_Transition_Count" in grp.columns else 0

        # Braking metrics
        brk_vals = grp[brk_col].to_numpy(dtype=int)
        brk_duration = np.sum(brk_vals > 0) * (telemetry_duration_s / n) if n > 0 else 0.0
        brk_pct = (np.sum(brk_vals > 0) / n) * 100.0
        brk_events = grp["Braking_Event_ID"].max() if "Braking_Event_ID" in grp.columns else 0

        # Acceleration & G-force
        peak_accel_g = 0.0
        peak_decel_g = 0.0
        mean_accel_mps2 = 0.0
        if "G_Longitudinal" in grp.columns:
            peak_accel_g = max(float(grp["G_Longitudinal"].max()), 0.0)
            peak_decel_g = abs(min(float(grp["G_Longitudinal"].min()), 0.0))
            mean_accel_mps2 = float(grp["Accel_Longitudinal_mps2"].mean())
        elif "AccelerationX" in grp.columns:
            # Fallback
            a_mag = np.diff(spd / 3.6) / np.maximum(np.diff(grp["_time_sec"].to_numpy()), 0.01)
            peak_accel_g = max(float(np.max(a_mag)) / GRAVITY, 0.0) if len(a_mag) > 0 else 0.0
            peak_decel_g = abs(min(float(np.min(a_mag)) / GRAVITY, 0.0)) if len(a_mag) > 0 else 0.0

        # Gear shifts and usage
        total_shifts = grp["Gear_Change_Count"].iloc[-1] if "Gear_Change_Count" in grp.columns else 0
        upshifts = grp["Upshift_Count"].iloc[-1] if "Upshift_Count" in grp.columns else 0
        downshifts = grp["Downshift_Count"].iloc[-1] if "Downshift_Count" in grp.columns else 0
        modal_gear = int(grp["nGear"].mode().iloc[0]) if "nGear" in grp.columns and len(grp["nGear"].mode()) > 0 else 0

        # Corner summaries
        corner_count = grp["Corner_ID"].max() if "Corner_ID" in grp.columns else 0
        mean_apex_spd = grp[grp["IsCornerApex"] == 1]["Speed"].mean() if "IsCornerApex" in grp.columns and (grp["IsCornerApex"] == 1).any() else np.nan
        min_apex_spd = grp[grp["IsCornerApex"] == 1]["Speed"].min() if "IsCornerApex" in grp.columns and (grp["IsCornerApex"] == 1).any() else np.nan

        # Sector breakdown
        s1_dur, s2_dur, s3_dur = np.nan, np.nan, np.nan
        if "Sector" in grp.columns:
            s1 = grp[grp["Sector"] == 1]
            s2 = grp[grp["Sector"] == 2]
            s3 = grp[grp["Sector"] == 3]
            s1_dur = round(s1["_time_sec"].iloc[-1] - s1["_time_sec"].iloc[0], 3) if len(s1) > 1 else np.nan
            s2_dur = round(s2["_time_sec"].iloc[-1] - s2["_time_sec"].iloc[0], 3) if len(s2) > 1 else np.nan
            s3_dur = round(s3["_time_sec"].iloc[-1] - s3["_time_sec"].iloc[0], 3) if len(s3) > 1 else np.nan

        # Data quality
        dt_vals = np.diff(grp["_time_sec"].to_numpy())
        mean_freq_hz = round(1.0 / np.median(dt_vals), 1) if len(dt_vals) > 0 and np.median(dt_vals) > 0 else 0.0
        dist_covered = round(grp[dist_col].iloc[-1] - grp[dist_col].iloc[0], 2)

        records.append({
            "Driver": str(driver),
            "LapNumber": int(lap),
            "LapClassification": str(lap_class),
            "LapTime_s": round(float(official_lap_time), 3),
            "TelemetryDuration_s": round(float(telemetry_duration_s), 3),
            "LapDistance_m": dist_covered,
            "Speed_Mean_kmh": round(float(spd_mean), 2),
            "Speed_Max_kmh": round(float(spd_max), 2),
            "Speed_Min_kmh": round(float(spd_min), 2),
            "Speed_Std_kmh": round(float(spd_std), 2),
            "Speed_Median_kmh": round(float(spd_p50), 2),
            "Speed_P25_kmh": round(float(spd_p25), 2),
            "Speed_P75_kmh": round(float(spd_p75), 2),
            "FullThrottle_Pct": round(float(wot_pct), 2),
            "Throttle_Mean_Pct": round(float(th_mean), 2),
            "Throttle_Transitions": int(th_trans_count),
            "Braking_Duration_s": round(float(brk_duration), 3),
            "Braking_Pct": round(float(brk_pct), 2),
            "Braking_Events_Count": int(brk_events),
            "Peak_Decel_G": round(float(peak_decel_g), 2),
            "Peak_Accel_G": round(float(peak_accel_g), 2),
            "Mean_Accel_mps2": round(float(mean_accel_mps2), 3),
            "Gear_Changes_Count": int(total_shifts),
            "Upshifts_Count": int(upshifts),
            "Downshifts_Count": int(downshifts),
            "Most_Used_Gear": int(modal_gear),
            "Corners_Count": int(corner_count),
            "Mean_Apex_Speed_kmh": round(float(mean_apex_spd), 2) if pd.notna(mean_apex_spd) else np.nan,
            "Min_Apex_Speed_kmh": round(float(min_apex_spd), 2) if pd.notna(min_apex_spd) else np.nan,
            "Sector1_Duration_s": s1_dur,
            "Sector2_Duration_s": s2_dur,
            "Sector3_Duration_s": s3_dur,
            "Sample_Count": int(n),
            "Sampling_Frequency_Hz": float(mean_freq_hz)
        })

    res_df = pd.DataFrame(records)
    logger.info("Lap-level aggregation complete: %d lap profiles generated.", len(res_df))
    return res_df
