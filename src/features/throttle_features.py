"""
F1 Telemetry Intelligence - Throttle Dynamics Feature Engineering
=================================================================

Computes throttle pedal application, release rates, and regime features:
- Throttle derivative d(Throttle)/dt (%/s) using actual timestamp differences.
- Throttle application rate (positive dThrottle/dt) vs release rate (negative dThrottle/dt).
- Full-throttle (Wide Open Throttle, WOT >= 98%), Partial-throttle (5-98%), and Off-throttle durations.
- Cumulative WOT percentage.
- Throttle transition detection and counts.
- Preserves raw vs clipped throttle distinctions.

Scientific and Numerical Standards:
- Accounts for irregular dt from timestamp differences; clamps dt >= 0.005s.
- Strictly avoids division-by-zero.
- Grouped by (Driver, LapNumber) to prevent boundary leakage.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.features.throttle_features")
WOT_THRESHOLD = 98.0  # Wide-Open Throttle threshold (%)
OFF_THROTTLE_THRESHOLD = 5.0  # Off-throttle threshold (%)


def compute_throttle_derivatives(
    throttle_pct: pd.Series,
    time_sec: pd.Series,
    min_dt: float = 0.005
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes throttle derivative (%/s), application rate, and release rate.

    Returns:
        (dThrottle_dt, Throttle_Application_Rate, Throttle_Release_Rate)
    """
    th = throttle_pct.to_numpy(dtype=np.float64)
    t = time_sec.to_numpy(dtype=np.float64)
    n = len(th)

    if n < 2:
        zeros = pd.Series(np.zeros(n), index=throttle_pct.index)
        return zeros, zeros, zeros

    dt = np.diff(t)
    dt_clamped = np.maximum(dt, min_dt)
    dth = np.diff(th)

    # First derivative
    deriv = np.empty(n, dtype=np.float64)
    deriv[0] = dth[0] / dt_clamped[0]
    deriv[1:] = dth / dt_clamped

    # Clip to physical human foot actuation limits (~500 %/s)
    deriv = np.clip(deriv, -600.0, 600.0)

    # Application rate is positive derivative, release rate is magnitude of negative derivative
    app_rate = np.where(deriv > 0.0, deriv, 0.0)
    rel_rate = np.where(deriv < 0.0, -deriv, 0.0)

    return (
        pd.Series(np.round(deriv, 2), index=throttle_pct.index, name="Throttle_dPct_per_s"),
        pd.Series(np.round(app_rate, 2), index=throttle_pct.index, name="Throttle_Application_Rate"),
        pd.Series(np.round(rel_rate, 2), index=throttle_pct.index, name="Throttle_Release_Rate")
    )


def compute_throttle_regime_durations(
    throttle_pct: pd.Series,
    time_sec: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes duration of WOT, partial throttle, and off-throttle regimes,
    along with cumulative WOT percentage and transition counts.

    Returns:
        (WOT_Duration_s, PartialThrottle_Duration_s, OffThrottle_Duration_s, WOT_Cumulative_Pct, Throttle_Transition_Count)
    """
    th = throttle_pct.to_numpy(dtype=np.float64)
    t = time_sec.to_numpy(dtype=np.float64)
    n = len(th)

    wot_dur = np.zeros(n, dtype=np.float64)
    part_dur = np.zeros(n, dtype=np.float64)
    off_dur = np.zeros(n, dtype=np.float64)
    wot_cum_pct = np.zeros(n, dtype=np.float64)
    trans_count = np.zeros(n, dtype=int)

    wot_start = None
    part_start = None
    off_start = None
    wot_total_time = 0.0
    transition_counter = 0
    prev_state = "OFF" if th[0] <= OFF_THROTTLE_THRESHOLD else ("WOT" if th[0] >= WOT_THRESHOLD else "PARTIAL")

    for i in range(n):
        val = th[i]
        curr_state = "WOT" if val >= WOT_THRESHOLD else ("OFF" if val <= OFF_THROTTLE_THRESHOLD else "PARTIAL")

        # Detect transition into application
        if prev_state == "OFF" and curr_state in ("PARTIAL", "WOT"):
            transition_counter += 1
        elif prev_state == "PARTIAL" and curr_state == "WOT":
            transition_counter += 1
        trans_count[i] = transition_counter
        prev_state = curr_state

        # WOT Duration
        if val >= WOT_THRESHOLD:
            if wot_start is None:
                wot_start = i
            wot_dur[i] = t[i] - t[wot_start]
            if i > 0:
                wot_total_time += max(t[i] - t[i - 1], 0.0)
        else:
            wot_start = None

        # Partial Duration
        if OFF_THROTTLE_THRESHOLD < val < WOT_THRESHOLD:
            if part_start is None:
                part_start = i
            part_dur[i] = t[i] - t[part_start]
        else:
            part_start = None

        # Off Throttle Duration
        if val <= OFF_THROTTLE_THRESHOLD:
            if off_start is None:
                off_start = i
            off_dur[i] = t[i] - t[off_start]
        else:
            off_start = None

        # Cumulative WOT % of elapsed lap time
        elapsed = t[i] - t[0]
        wot_cum_pct[i] = (wot_total_time / elapsed * 100.0) if elapsed > 0.1 else 0.0

    return (
        pd.Series(np.round(wot_dur, 3), index=throttle_pct.index, name="FullThrottle_Duration_s"),
        pd.Series(np.round(part_dur, 3), index=throttle_pct.index, name="PartialThrottle_Duration_s"),
        pd.Series(np.round(off_dur, 3), index=throttle_pct.index, name="OffThrottle_Duration_s"),
        pd.Series(np.round(wot_cum_pct, 2), index=throttle_pct.index, name="FullThrottle_Cumulative_Pct"),
        pd.Series(trans_count, index=throttle_pct.index, name="Throttle_Transition_Count")
    )


def compute_throttle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates full throttle dynamics feature generation grouped strictly by (Driver, LapNumber).

    Input columns required:
        - Driver, LapNumber, Throttle (and optionally Throttle_Clipped), Time (or SessionTime)

    Generated features:
        - Throttle_Pct: Active throttle percentage used (Throttle_Clipped if present, else Throttle).
        - Throttle_dPct_per_s: Throttle rate of change (%/s).
        - Throttle_Application_Rate: Positive application rate (%/s).
        - Throttle_Release_Rate: Absolute negative release rate (%/s).
        - IsFullThrottle: Binary flag (1 if Throttle >= 98%).
        - IsPartialThrottle: Binary flag (1 if 5% < Throttle < 98%).
        - IsOffThrottle: Binary flag (1 if Throttle <= 5%).
        - FullThrottle_Duration_s: Contiguous elapsed duration at WOT.
        - PartialThrottle_Duration_s: Contiguous elapsed duration at partial throttle.
        - OffThrottle_Duration_s: Contiguous elapsed duration off throttle.
        - FullThrottle_Cumulative_Pct: Running cumulative WOT percentage for the lap.
        - Throttle_Transition_Count: Running count of throttle application transitions.
    """
    logger.info("Computing throttle dynamics features across %d rows...", len(df))

    df_work = df.copy()
    if pd.api.types.is_timedelta64_dtype(df_work["Time"]):
        df_work["_time_sec"] = df_work["Time"].dt.total_seconds()
    elif "SessionTime" in df_work.columns and pd.api.types.is_timedelta64_dtype(df_work["SessionTime"]):
        df_work["_time_sec"] = df_work["SessionTime"].dt.total_seconds()
    else:
        df_work["_time_sec"] = pd.to_numeric(df_work["Time"])

    th_col = "Throttle_Clipped" if "Throttle_Clipped" in df_work.columns else "Throttle"
    feature_dfs = []

    for (driver, lap), grp in df_work.groupby(["Driver", "LapNumber"], sort=False):
        grp_res = pd.DataFrame(index=grp.index)
        grp_res["Throttle_Pct"] = grp[th_col]

        # Derivatives
        dth, app_rate, rel_rate = compute_throttle_derivatives(grp[th_col], grp["_time_sec"])
        grp_res["Throttle_dPct_per_s"] = dth
        grp_res["Throttle_Application_Rate"] = app_rate
        grp_res["Throttle_Release_Rate"] = rel_rate

        # Binary regime flags
        grp_res["IsFullThrottle"] = (grp[th_col] >= WOT_THRESHOLD).astype(int)
        grp_res["IsPartialThrottle"] = ((grp[th_col] > OFF_THROTTLE_THRESHOLD) & (grp[th_col] < WOT_THRESHOLD)).astype(int)
        grp_res["IsOffThrottle"] = (grp[th_col] <= OFF_THROTTLE_THRESHOLD).astype(int)

        # Durations & transitions
        wot_dur, part_dur, off_dur, cum_pct, trans_cnt = compute_throttle_regime_durations(
            grp[th_col], grp["_time_sec"]
        )
        grp_res["FullThrottle_Duration_s"] = wot_dur
        grp_res["PartialThrottle_Duration_s"] = part_dur
        grp_res["OffThrottle_Duration_s"] = off_dur
        grp_res["FullThrottle_Cumulative_Pct"] = cum_pct
        grp_res["Throttle_Transition_Count"] = trans_cnt

        feature_dfs.append(grp_res)

    result_df = pd.concat(feature_dfs).loc[df.index]
    logger.info("Throttle dynamics features completed successfully.")
    return result_df
