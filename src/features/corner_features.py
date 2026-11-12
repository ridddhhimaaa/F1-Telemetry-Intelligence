"""
F1 Telemetry Intelligence - Corner Performance Feature Engineering
==================================================================

Computes cornering kinematic performance features across detected corners:
- Corner entry speed (v_entry), apex minimum speed (v_apex), corner exit speed (v_exit).
- Speed loss through braking/entry phase: Delta v_loss = v_entry - v_apex (km/h).
- Acceleration / exit drive: Delta v_exit = v_exit - v_apex (km/h).
- Corner traversal duration (seconds) and pre-apex vs post-apex phases.
- Corner-level performance summary table (one row per corner event per lap).

Scientific and Numerical Standards:
- Seamlessly consumes Corner_ID and IsCornerApex from track_geometry module.
- Preserves exact physical cornering dynamics without spatial interpolation artifacts.
- Strictly grouped by (Driver, LapNumber) without inter-lap leakage.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.features.corner_features")


def compute_corner_kinematics(
    corner_id: pd.Series,
    is_apex: pd.Series,
    speed_kmh: pd.Series,
    distance_m: pd.Series,
    time_sec: pd.Series,
    radius_m: Optional[pd.Series] = None,
    turn_dir: Optional[pd.Series] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes sample-level corner metrics and aggregates a corner summary table for the lap.

    Returns:
        (sample_level_features_df, corner_summary_table_df)
    """
    cid = corner_id.to_numpy(dtype=int)
    apx = is_apex.to_numpy(dtype=int)
    spd = speed_kmh.to_numpy(dtype=np.float64)
    dist = distance_m.to_numpy(dtype=np.float64)
    t = time_sec.to_numpy(dtype=np.float64)
    r = radius_m.to_numpy(dtype=np.float64) if radius_m is not None else np.full(len(cid), 1000.0)
    tdir = turn_dir.to_numpy(dtype=object) if turn_dir is not None else np.full(len(cid), "CORNER")

    n = len(cid)

    # Sample-level feature arrays
    v_entry_arr = np.zeros(n, dtype=np.float64)
    v_apex_arr = np.zeros(n, dtype=np.float64)
    v_exit_arr = np.zeros(n, dtype=np.float64)
    spd_loss_arr = np.zeros(n, dtype=np.float64)
    exit_acc_arr = np.zeros(n, dtype=np.float64)
    corner_dur_arr = np.zeros(n, dtype=np.float64)
    is_pre_apex = np.zeros(n, dtype=int)
    is_post_apex = np.zeros(n, dtype=int)

    summary_records = []
    unique_corners = sorted([c for c in set(cid) if c > 0])

    for c in unique_corners:
        mask = np.where(cid == c)[0]
        if len(mask) == 0:
            continue

        c_start = mask[0]
        c_end = mask[-1]
        c_dur = t[c_end] - t[c_start]

        # Entry speed is speed at entry of corner
        v_entry = spd[c_start]
        # Exit speed is speed at exit of corner
        v_exit = spd[c_end]

        # Apex location: where is_apex == 1, or minimum speed if not flagged
        apex_sub = np.where(apx[c_start:c_end + 1] == 1)[0]
        if len(apex_sub) > 0:
            apex_idx = c_start + apex_sub[0]
        else:
            apex_idx = c_start + np.argmin(spd[c_start:c_end + 1])

        v_apex = spd[apex_idx]
        spd_loss = max(v_entry - v_apex, 0.0)
        exit_accel = max(v_exit - v_apex, 0.0)
        min_rad = np.min(r[c_start:c_end + 1])
        t_direction = tdir[c_start]

        # Assign sample-level values
        v_entry_arr[c_start:c_end + 1] = np.round(v_entry, 2)
        v_apex_arr[c_start:c_end + 1] = np.round(v_apex, 2)
        v_exit_arr[c_start:c_end + 1] = np.round(v_exit, 2)
        spd_loss_arr[c_start:c_end + 1] = np.round(spd_loss, 2)
        exit_acc_arr[c_start:c_end + 1] = np.round(exit_accel, 2)
        corner_dur_arr[c_start:c_end + 1] = np.round(c_dur, 3)

        # Pre-apex and post-apex indicators
        is_pre_apex[c_start:apex_idx] = 1
        is_post_apex[apex_idx + 1:c_end + 1] = 1

        summary_records.append({
            "Corner_ID": c,
            "TurnDirection": t_direction,
            "StartDistance_m": round(dist[c_start], 1),
            "ApexDistance_m": round(dist[apex_idx], 1),
            "EndDistance_m": round(dist[c_end], 1),
            "CornerLength_m": round(dist[c_end] - dist[c_start], 1),
            "EntrySpeed_kmh": round(v_entry, 2),
            "ApexSpeed_kmh": round(v_apex, 2),
            "ExitSpeed_kmh": round(v_exit, 2),
            "SpeedLoss_kmh": round(spd_loss, 2),
            "ExitAccel_kmh": round(exit_accel, 2),
            "CornerDuration_s": round(c_dur, 3),
            "MinRadius_m": round(min_rad, 1)
        })

    sample_df = pd.DataFrame({
        "Corner_Entry_Speed_kmh": v_entry_arr,
        "Corner_Apex_Speed_kmh": v_apex_arr,
        "Corner_Exit_Speed_kmh": v_exit_arr,
        "Corner_Speed_Loss_kmh": spd_loss_arr,
        "Corner_Exit_Accel_kmh": exit_acc_arr,
        "Corner_Traversal_Duration_s": corner_dur_arr,
        "IsPreApexZone": is_pre_apex,
        "IsPostApexZone": is_post_apex
    }, index=corner_id.index)

    summary_df = pd.DataFrame(summary_records)
    return sample_df, summary_df


def compute_corner_features(
    df: pd.DataFrame,
    corner_id_col: str = "Corner_ID",
    apex_col: str = "IsCornerApex",
    radius_col: str = "Radius_of_Curvature_m",
    turn_dir_col: str = "TurnDirection"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates full corner performance feature generation grouped strictly by (Driver, LapNumber).

    Input columns required:
        - Driver, LapNumber, Speed, Distance_Corrected (or Distance), Time (or SessionTime), Corner_ID, IsCornerApex

    Returns:
        (sample_level_corner_features_df, corner_summary_table_df)
    """
    logger.info("Computing corner performance features across %d rows...", len(df))

    df_work = df.copy()
    if pd.api.types.is_timedelta64_dtype(df_work["Time"]):
        df_work["_time_sec"] = df_work["Time"].dt.total_seconds()
    elif "SessionTime" in df_work.columns and pd.api.types.is_timedelta64_dtype(df_work["SessionTime"]):
        df_work["_time_sec"] = df_work["SessionTime"].dt.total_seconds()
    else:
        df_work["_time_sec"] = pd.to_numeric(df_work["Time"])

    dist_col = "Distance_Corrected" if "Distance_Corrected" in df_work.columns else "Distance"

    sample_dfs = []
    summary_dfs = []

    for (driver, lap), grp in df_work.groupby(["Driver", "LapNumber"], sort=False):
        c_id = grp[corner_id_col] if corner_id_col in grp.columns else pd.Series(0, index=grp.index)
        c_apx = grp[apex_col] if apex_col in grp.columns else pd.Series(0, index=grp.index)
        c_rad = grp[radius_col] if radius_col in grp.columns else None
        c_dir = grp[turn_dir_col] if turn_dir_col in grp.columns else None

        s_df, sum_df = compute_corner_kinematics(
            c_id, c_apx, grp["Speed"], grp[dist_col], grp["_time_sec"], c_rad, c_dir
        )
        sample_dfs.append(s_df)

        if len(sum_df) > 0:
            sum_df.insert(0, "Driver", driver)
            sum_df.insert(1, "LapNumber", lap)
            summary_dfs.append(sum_df)

    res_samples = pd.concat(sample_dfs).loc[df.index]
    res_summary = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()

    logger.info("Corner performance features completed: %d corner events recorded.", len(res_summary))
    return res_samples, res_summary
