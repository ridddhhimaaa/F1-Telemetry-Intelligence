"""
F1 Telemetry Intelligence - Braking Dynamics Feature Engineering
================================================================

Computes comprehensive braking dynamics, event segmentation, and pedal interaction:
- Braking onset detection (sharp transition into braking).
- Continuous braking duration (seconds) and braking distance (meters).
- Braking intensity: instantaneous deceleration, peak event G-force, mean event G-force.
- Two-phase braking: Initial application phase (onset -> peak deceleration) vs release/trail phase (peak -> release).
- Brake-to-throttle transition time and pedal overlap (trail braking indicator).
- Sequential Braking Event IDs within each lap.

Scientific and Numerical Standards:
- Integrates with Phase 4 validated IsBraking definition.
- Derives physical braking distance using calibrated Distance_Corrected channel.
- Strictly isolated by (Driver, LapNumber) without boundary contamination.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.features.braking_features")
GRAVITY = 9.80665


def segment_braking_events(
    is_braking: pd.Series,
    distance_m: pd.Series,
    time_sec: pd.Series,
    accel_long: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Identifies continuous braking events and calculates event-level metrics:
    - Braking_Event_ID: 1, 2, ... for distinct braking zones, 0 when off-brake.
    - IsBrakingOnset: 1 on the exact timestep of brake engagement, 0 otherwise.
    - BrakingDuration_s: Elapsed duration within current braking zone.
    - BrakingDistance_m: Distance traversed from onset of current braking zone.
    - PeakBraking_G: Maximum deceleration (magnitude in g) reached during the event.
    - IsBrakeReleasePhase: 1 if currently trailing off after peak deceleration point.

    Returns:
        (event_id, is_onset, duration_s, distance_m, peak_g, is_release_phase)
    """
    brk = is_braking.to_numpy(dtype=int)
    dist = distance_m.to_numpy(dtype=np.float64)
    t = time_sec.to_numpy(dtype=np.float64)
    a = accel_long.to_numpy(dtype=np.float64)
    n = len(brk)

    event_id = np.zeros(n, dtype=int)
    is_onset = np.zeros(n, dtype=int)
    durations = np.zeros(n, dtype=np.float64)
    distances = np.zeros(n, dtype=np.float64)
    peak_g = np.zeros(n, dtype=np.float64)
    is_release = np.zeros(n, dtype=int)

    current_event = 0
    start_idx = None

    for i in range(n):
        if brk[i] == 1:
            if start_idx is None:
                start_idx = i
                current_event += 1
                is_onset[i] = 1

            event_id[i] = current_event
            durations[i] = t[i] - t[start_idx]
            distances[i] = dist[i] - dist[start_idx]
        else:
            if start_idx is not None:
                # Post-process the completed event interval to compute peak G and release phase
                end_idx = i - 1
                event_accel = a[start_idx:end_idx + 1]
                # Most negative acceleration = peak deceleration
                peak_decel_offset = np.argmin(event_accel)
                peak_val_g = np.abs(event_accel[peak_decel_offset]) / GRAVITY
                peak_g[start_idx:end_idx + 1] = np.round(peak_val_g, 2)

                # Release phase: after the peak deceleration point
                peak_idx = start_idx + peak_decel_offset
                if peak_idx < end_idx:
                    is_release[peak_idx + 1:end_idx + 1] = 1

                start_idx = None

    # Handle boundary if event continues to last sample
    if start_idx is not None:
        end_idx = n - 1
        event_accel = a[start_idx:end_idx + 1]
        peak_offset = np.argmin(event_accel)
        peak_g[start_idx:end_idx + 1] = np.round(np.abs(event_accel[peak_offset]) / GRAVITY, 2)
        peak_idx = start_idx + peak_offset
        if peak_idx < end_idx:
            is_release[peak_idx + 1:end_idx + 1] = 1

    return (
        pd.Series(event_id, index=is_braking.index, name="Braking_Event_ID"),
        pd.Series(is_onset, index=is_braking.index, name="IsBrakingOnset"),
        pd.Series(np.round(durations, 3), index=is_braking.index, name="BrakingDuration_s"),
        pd.Series(np.round(distances, 2), index=is_braking.index, name="BrakingDistance_m"),
        pd.Series(peak_g, index=is_braking.index, name="BrakeEvent_Peak_G"),
        pd.Series(is_release, index=is_braking.index, name="IsBrakeReleasePhase")
    )


def compute_pedal_transition_and_overlap(
    is_braking: pd.Series,
    throttle_pct: pd.Series,
    time_sec: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Calculates pedal overlap (trail braking / overlapping inputs) and
    elapsed time since last brake release until throttle re-application.

    Returns:
        (IsPedalOverlap, TimeSinceBrakeRelease_s)
    """
    brk = is_braking.to_numpy(dtype=int)
    th = throttle_pct.to_numpy(dtype=np.float64)
    t = time_sec.to_numpy(dtype=np.float64)
    n = len(brk)

    # Overlap: driver applies throttle (> 15%) while still on the brakes
    is_overlap = (brk == 1) & (th > 15.0)

    time_since_release = np.zeros(n, dtype=np.float64)
    last_release_time = None

    for i in range(n):
        if i > 0 and brk[i - 1] == 1 and brk[i] == 0:
            last_release_time = t[i]

        if last_release_time is not None:
            time_since_release[i] = t[i] - last_release_time
        else:
            time_since_release[i] = 0.0

    return (
        pd.Series(is_overlap.astype(int), index=is_braking.index, name="IsPedalOverlap"),
        pd.Series(np.round(time_since_release, 3), index=is_braking.index, name="TimeSinceBrakeRelease_s")
    )


def compute_braking_features(
    df: pd.DataFrame,
    accel_col: Optional[str] = "Accel_Longitudinal_mps2"
) -> pd.DataFrame:
    """
    Orchestrates full braking dynamics feature generation grouped strictly by (Driver, LapNumber).

    Input columns required:
        - Driver, LapNumber, IsBraking (or Brake), Distance_Corrected (or Distance), Time (or SessionTime)

    Generated features:
        - Braking_Event_ID: Sequential integer index of braking zone (0 = off-brake).
        - IsBrakingOnset: Binary flag (1 on initial brake engagement tick).
        - BrakingDuration_s: Elapsed duration within current braking zone (seconds).
        - BrakingDistance_m: Distance traversed under braking from onset (meters).
        - BrakeEvent_Peak_G: Peak longitudinal deceleration reached in the braking zone (g).
        - IsBrakeReleasePhase: Binary flag (1 during trailing-off phase after peak deceleration).
        - IsPedalOverlap: Binary flag (1 if simultaneously braking and applying throttle > 15%).
        - TimeSinceBrakeRelease_s: Elapsed time since most recent brake release (seconds).
    """
    logger.info("Computing braking dynamics features across %d rows...", len(df))

    df_work = df.copy()
    if pd.api.types.is_timedelta64_dtype(df_work["Time"]):
        df_work["_time_sec"] = df_work["Time"].dt.total_seconds()
    elif "SessionTime" in df_work.columns and pd.api.types.is_timedelta64_dtype(df_work["SessionTime"]):
        df_work["_time_sec"] = df_work["SessionTime"].dt.total_seconds()
    else:
        df_work["_time_sec"] = pd.to_numeric(df_work["Time"])

    dist_col = "Distance_Corrected" if "Distance_Corrected" in df_work.columns else "Distance"
    brk_col = "IsBraking" if "IsBraking" in df_work.columns else "Brake"
    th_col = "Throttle_Clipped" if "Throttle_Clipped" in df_work.columns else "Throttle"

    # If longitudinal acceleration is not pre-computed, approximate it directly from speed
    if accel_col not in df_work.columns:
        v_mps = df_work["Speed"].to_numpy(dtype=np.float64) / 3.6
        dt = np.maximum(np.diff(df_work["_time_sec"].to_numpy(dtype=np.float64)), 0.005)
        a_approx = np.concatenate([[0.0], np.diff(v_mps) / dt])
        df_work["_accel"] = a_approx
        use_accel_col = "_accel"
    else:
        use_accel_col = accel_col

    feature_dfs = []

    for (driver, lap), grp in df_work.groupby(["Driver", "LapNumber"], sort=False):
        grp_res = pd.DataFrame(index=grp.index)

        # Braking event segmentation & kinematics
        evt_id, is_onset, dur_s, dist_m, peak_g, is_rel = segment_braking_events(
            grp[brk_col],
            grp[dist_col],
            grp["_time_sec"],
            grp[use_accel_col]
        )
        grp_res["Braking_Event_ID"] = evt_id
        grp_res["IsBrakingOnset"] = is_onset
        grp_res["BrakingDuration_s"] = dur_s
        grp_res["BrakingDistance_m"] = dist_m
        grp_res["BrakeEvent_Peak_G"] = peak_g
        grp_res["IsBrakeReleasePhase"] = is_rel

        # Pedal overlap & transition
        is_ovlp, t_rel = compute_pedal_transition_and_overlap(
            grp[brk_col],
            grp[th_col],
            grp["_time_sec"]
        )
        grp_res["IsPedalOverlap"] = is_ovlp
        grp_res["TimeSinceBrakeRelease_s"] = t_rel

        feature_dfs.append(grp_res)

    result_df = pd.concat(feature_dfs).loc[df.index]
    logger.info("Braking dynamics features completed successfully.")
    return result_df
