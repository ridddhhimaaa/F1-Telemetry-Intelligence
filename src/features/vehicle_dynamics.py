"""
F1 Telemetry Intelligence - Vehicle Dynamics Feature Engineering
================================================================

Computes body-frame longitudinal and lateral vehicle dynamics from telemetry:
- Longitudinal acceleration (m/s^2 and g-force) using actual timestamp differences.
- Speed change rate (km/h/s).
- Acceleration, braking, and coasting dynamical regimes.
- Consecutive acceleration and deceleration phase durations.
- Combined resultant acceleration (friction circle utilization).
- Rolling speed variation.

Scientific and Numerical Standards:
- Accounts for irregular sampling intervals (dt from timestamp differences).
- Clamps dt >= 0.005s to prevent division-by-zero or numerical blowup.
- Applies conservative Savitzky-Golay or rolling filters without destroying genuine peak braking.
- Strictly groups by (Driver, LapNumber) to prevent boundary leakage across laps.
- Does NOT claim derived accelerations are raw IMU measurements.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger("f1_telemetry.features.vehicle_dynamics")
GRAVITY = 9.80665  # Standard acceleration due to gravity, m/s^2


def compute_longitudinal_acceleration(
    speed_kmh: pd.Series,
    time_sec: pd.Series,
    smooth: bool = True,
    window_length: int = 5,
    polyorder: int = 2,
    min_dt: float = 0.005,
    max_accel_limit: float = 35.0  # ~3.5g traction limit
) -> tuple[pd.Series, pd.Series]:
    """
    Computes longitudinal acceleration (m/s^2) and G-force (g) from speed and time.

    Formula:
        v_mps = speed_kmh / 3.6
        dt = max(t_i - t_{i-1}, min_dt)
        a_long = dv_mps / dt

    Returns:
        (accel_long_mps2, g_long)
    """
    v_mps = (speed_kmh / 3.6).to_numpy(dtype=np.float64)
    t = time_sec.to_numpy(dtype=np.float64)
    n = len(v_mps)

    if n < 2:
        return (
            pd.Series(np.zeros(n, dtype=np.float64), index=speed_kmh.index),
            pd.Series(np.zeros(n, dtype=np.float64), index=speed_kmh.index)
        )

    # Compute actual irregular timestamp differences
    dt = np.diff(t)
    dt_clamped = np.maximum(dt, min_dt)
    dv = np.diff(v_mps)

    # Forward derivative for initial point, backward for others
    a_raw = np.empty(n, dtype=np.float64)
    a_raw[0] = dv[0] / dt_clamped[0]
    a_raw[1:] = dv / dt_clamped

    # Handle numerical extremes / outliers safely
    a_raw = np.clip(a_raw, -70.0, 40.0)  # F1 cars can reach -5.5g (-54 m/s^2) in Turn 6 braking

    if smooth and n >= window_length:
        # Savitzky-Golay filter preserves peak deceleration points while eliminating quantization noise
        w = window_length if window_length % 2 == 1 else window_length + 1
        w = min(w, n if n % 2 == 1 else n - 1)
        if w > polyorder:
            a_smoothed = savgol_filter(a_raw, window_length=w, polyorder=polyorder)
        else:
            a_smoothed = pd.Series(a_raw).rolling(3, min_periods=1, center=True).mean().to_numpy()
    else:
        a_smoothed = a_raw

    g_long = a_smoothed / GRAVITY

    return (
        pd.Series(np.round(a_smoothed, 4), index=speed_kmh.index, name="Accel_Longitudinal_mps2"),
        pd.Series(np.round(g_long, 4), index=speed_kmh.index, name="G_Longitudinal")
    )


def compute_lateral_acceleration(
    speed_kmh: pd.Series,
    curvature_m_inv: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Computes lateral (centripetal) acceleration and lateral G-force.

    Formula:
        a_lat = v_mps^2 * kappa
        g_lat = a_lat / GRAVITY
    """
    v_mps = (speed_kmh / 3.6).to_numpy(dtype=np.float64)
    kappa = curvature_m_inv.to_numpy(dtype=np.float64)

    a_lat = (v_mps ** 2) * kappa
    # Clamp to physically observable F1 limits (+/- 6.5g = +/- 63.7 m/s^2 in Copse/Turn 9)
    a_lat = np.clip(a_lat, -65.0, 65.0)
    g_lat = a_lat / GRAVITY

    return (
        pd.Series(np.round(a_lat, 4), index=speed_kmh.index, name="Accel_Lateral_mps2"),
        pd.Series(np.round(g_lat, 4), index=speed_kmh.index, name="G_Lateral")
    )


def classify_dynamics_regimes(
    accel_long: pd.Series,
    speed_kmh: pd.Series,
    throttle: pd.Series,
    is_braking: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Classifies vehicle operational regimes and returns boolean flags:
    - IsAccelerating: a_long > 0.5 m/s^2 and Throttle > 20%
    - IsDecelerating: a_long < -0.5 m/s^2
    - IsCoasting: -0.5 <= a_long <= 0.5 m/s^2, Throttle < 15%, IsBraking == 0
    - DynamicsRegime: Categorical string regime
    """
    is_accel = (accel_long > 0.5) & (throttle > 15.0)
    is_decel = (accel_long < -0.5) | (is_braking > 0)
    is_coast = (~is_accel) & (~is_decel) & (throttle < 15.0) & (is_braking == 0)

    regimes = []
    a_vals = accel_long.to_numpy()
    v_vals = speed_kmh.to_numpy()
    brk_vals = is_braking.to_numpy()
    thr_vals = throttle.to_numpy()

    for a, v, b, th in zip(a_vals, v_vals, brk_vals, thr_vals):
        if b > 0 and a < -20.0:
            regimes.append("HEAVY_BRAKING")
        elif b > 0 or a < -2.0:
            regimes.append("MEDIUM_BRAKING")
        elif th > 85.0 and v >= 180.0:
            regimes.append("AERO_LIMITED_ACCEL")
        elif th > 50.0 and v < 180.0:
            regimes.append("TRACTION_LIMITED_ACCEL")
        elif th > 15.0:
            regimes.append("PARTIAL_THROTTLE_ACCEL")
        elif th < 15.0 and b == 0 and abs(a) <= 1.0:
            regimes.append("COASTING")
        else:
            regimes.append("TRANSITION")

    return (
        pd.Series(is_accel.astype(int), index=speed_kmh.index, name="IsAccelerating"),
        pd.Series(is_decel.astype(int), index=speed_kmh.index, name="IsDecelerating"),
        pd.Series(is_coast.astype(int), index=speed_kmh.index, name="IsCoasting"),
        pd.Series(regimes, index=speed_kmh.index, name="DynamicsRegime")
    )


def compute_phase_durations(
    is_event: pd.Series,
    time_sec: pd.Series
) -> pd.Series:
    """
    Calculates the cumulative elapsed duration (in seconds) of ongoing continuous phases.
    Resets to 0.0 when the condition is False.
    """
    event_arr = is_event.to_numpy(dtype=int)
    t_arr = time_sec.to_numpy(dtype=np.float64)
    n = len(event_arr)
    durations = np.zeros(n, dtype=np.float64)

    start_idx = None
    for i in range(n):
        if event_arr[i] == 1:
            if start_idx is None:
                start_idx = i
            durations[i] = t_arr[i] - t_arr[start_idx]
        else:
            start_idx = None

    return pd.Series(np.round(durations, 3), index=is_event.index)


def compute_vehicle_dynamics_features(
    df: pd.DataFrame,
    curvature_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Orchestrates full vehicle dynamics feature generation grouped strictly by (Driver, LapNumber).

    Input columns required:
        - Driver, LapNumber, Speed, Time (or SessionTime), Throttle, IsBraking (or Brake)

    Generated features:
        - Accel_Longitudinal_mps2: Longitudinal acceleration in m/s^2.
        - G_Longitudinal: Longitudinal G-force (g).
        - SpeedChangeRate_kmh_per_s: Rate of speed change (km/h/s).
        - Accel_Lateral_mps2: Lateral acceleration in m/s^2 (if curvature supplied).
        - G_Lateral: Lateral G-force (if curvature supplied).
        - G_Combined: Resultant total horizontal acceleration sqrt(g_long^2 + g_lat^2).
        - IsAccelerating: Binary flag (1 if accelerating, 0 otherwise).
        - IsDecelerating: Binary flag (1 if decelerating, 0 otherwise).
        - IsCoasting: Binary flag (1 if coasting, 0 otherwise).
        - DynamicsRegime: Operational regime string.
        - AccelPhaseDuration_s: Elapsed duration of current acceleration phase.
        - DecelPhaseDuration_s: Elapsed duration of current braking/deceleration phase.
        - Speed_Rolling_Std_1s: Rolling standard deviation of speed (~1.0s window).
    """
    logger.info("Computing vehicle dynamics features across %d rows...", len(df))

    # Ensure time in seconds
    df_work = df.copy()
    if pd.api.types.is_timedelta64_dtype(df_work["Time"]):
        df_work["_time_sec"] = df_work["Time"].dt.total_seconds()
    elif "SessionTime" in df_work.columns and pd.api.types.is_timedelta64_dtype(df_work["SessionTime"]):
        df_work["_time_sec"] = df_work["SessionTime"].dt.total_seconds()
    else:
        df_work["_time_sec"] = pd.to_numeric(df_work["Time"])

    throttle_col = "Throttle_Clipped" if "Throttle_Clipped" in df_work.columns else "Throttle"
    brake_col = "IsBraking" if "IsBraking" in df_work.columns else "Brake"

    feature_dfs = []

    # Process group by group to strictly prevent inter-lap leakage
    for (driver, lap), grp in df_work.groupby(["Driver", "LapNumber"], sort=False):
        grp_res = pd.DataFrame(index=grp.index)

        # 1. Longitudinal acceleration
        a_long, g_long = compute_longitudinal_acceleration(
            grp["Speed"],
            grp["_time_sec"]
        )
        grp_res["Accel_Longitudinal_mps2"] = a_long
        grp_res["G_Longitudinal"] = g_long
        grp_res["SpeedChangeRate_kmh_per_s"] = np.round(a_long * 3.6, 3)

        # 2. Lateral acceleration (if curvature present)
        if curvature_col and curvature_col in grp.columns:
            a_lat, g_lat = compute_lateral_acceleration(grp["Speed"], grp[curvature_col])
            grp_res["Accel_Lateral_mps2"] = a_lat
            grp_res["G_Lateral"] = g_lat
            grp_res["G_Combined"] = np.round(np.sqrt(g_long ** 2 + g_lat ** 2), 4)
        else:
            grp_res["Accel_Lateral_mps2"] = np.nan
            grp_res["G_Lateral"] = np.nan
            grp_res["G_Combined"] = np.abs(g_long)

        # 3. Dynamic regimes & flags
        is_accel, is_decel, is_coast, regime = classify_dynamics_regimes(
            a_long,
            grp["Speed"],
            grp[throttle_col],
            grp[brake_col]
        )
        grp_res["IsAccelerating"] = is_accel
        grp_res["IsDecelerating"] = is_decel
        grp_res["IsCoasting"] = is_coast
        grp_res["DynamicsRegime"] = regime

        # 4. Phase durations
        grp_res["AccelPhaseDuration_s"] = compute_phase_durations(is_accel, grp["_time_sec"])
        grp_res["DecelPhaseDuration_s"] = compute_phase_durations(is_decel, grp["_time_sec"])

        # 5. Rolling speed variation (~10 samples at ~10 Hz = 1.0 second)
        grp_res["Speed_Rolling_Std_1s"] = np.round(
            grp["Speed"].rolling(window=10, min_periods=2).std().fillna(0.0), 3
        )

        feature_dfs.append(grp_res)

    result_df = pd.concat(feature_dfs).loc[df.index]
    logger.info("Vehicle dynamics features completed successfully.")
    return result_df
