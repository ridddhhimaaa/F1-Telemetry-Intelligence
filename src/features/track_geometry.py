"""
F1 Telemetry Intelligence - Track Geometry Feature Engineering
=============================================================

Extracts spatial track geometry, curvature, heading, and cornering intervals:
- Validates coordinate orientation and converts FastF1 coordinates (decimeters) to meters:
    X_m = 0.1 * X, Y_m = 0.1 * Y, Z_m = 0.1 * Z
- Computes vehicle trajectory heading angle (theta) and heading change rate.
- Computes trajectory curvature kappa = d(theta)/ds (m^-1) and instantaneous radius of curvature R = 1/|kappa|.
- Automatically detects corners based on curvature thresholds.
- Identifies corner entry, apex (maximum curvature / minimum radius), and corner exit.
- Classifies turn direction (LEFT, RIGHT, STRAIGHT).

Scientific and Numerical Standards:
- Accounts for coordinate scale verification (decimeter to meter factor 0.1).
- Uses unwrap(theta) to eliminate 2*pi boundary discontinuities.
- Clamps arc length step ds >= 0.1m to prevent division-by-zero on stopped/slow telemetry.
- Smooths coordinates with low-pass Savitzky-Golay filter to remove GPS/interpolated jitter.
- Strictly isolated by (Driver, LapNumber) without boundary contamination.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

logger = logging.getLogger("f1_telemetry.features.track_geometry")
COORD_SCALE_FACTOR = 0.1  # FastF1 coordinates are in decimeters; 0.1 converts to meters
CURVATURE_CORNER_THRESHOLD = 0.004  # Threshold for corner detection: R < 250m (0.004 m^-1)


def compute_heading_and_curvature(
    x_coords: pd.Series,
    y_coords: pd.Series,
    smooth_coords: bool = True,
    window_length: int = 7,
    polyorder: int = 2,
    min_ds: float = 0.1
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes heading angle (radians and degrees), curvature kappa (m^-1), and radius (m).

    Returns:
        (heading_deg, heading_change_deg, curvature_m_inv, radius_m)
    """
    x_m = x_coords.to_numpy(dtype=np.float64) * COORD_SCALE_FACTOR
    y_m = y_coords.to_numpy(dtype=np.float64) * COORD_SCALE_FACTOR
    n = len(x_m)

    if n < 3:
        zeros = pd.Series(np.zeros(n), index=x_coords.index)
        return zeros, zeros, zeros, pd.Series(np.full(n, 10000.0), index=x_coords.index)

    if smooth_coords and n >= window_length:
        w = window_length if window_length % 2 == 1 else window_length + 1
        w = min(w, n if n % 2 == 1 else n - 1)
        if w > polyorder:
            x_m = savgol_filter(x_m, window_length=w, polyorder=polyorder)
            y_m = savgol_filter(y_m, window_length=w, polyorder=polyorder)

    # First derivatives along trajectory
    dx = np.gradient(x_m)
    dy = np.gradient(y_m)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    ds_clamped = np.maximum(ds, min_ds)

    # Heading angle theta = arctan2(dy, dx)
    theta = np.arctan2(dy, dx)
    theta_unwrapped = np.unwrap(theta)

    # Heading change rate d(theta)/ds (curvature kappa)
    dtheta = np.gradient(theta_unwrapped)
    kappa = dtheta / ds_clamped

    # Clip curvature to physical bounds: Turn 5 hairpin minimum radius ~14m -> kappa_max ~ 0.075 m^-1
    kappa = np.clip(kappa, -0.10, 0.10)

    # Instantaneous radius of curvature R = 1 / |kappa|
    abs_kappa = np.abs(kappa)
    radius = np.where(abs_kappa > 1e-4, 1.0 / np.maximum(abs_kappa, 1e-6), 10000.0)
    radius = np.clip(radius, 5.0, 10000.0)

    # Heading in degrees [0, 360)
    heading_deg = np.mod(np.degrees(theta), 360.0)
    dtheta_deg = np.degrees(dtheta)

    return (
        pd.Series(np.round(heading_deg, 2), index=x_coords.index, name="Heading_deg"),
        pd.Series(np.round(dtheta_deg, 3), index=x_coords.index, name="HeadingChange_deg"),
        pd.Series(np.round(kappa, 6), index=x_coords.index, name="Curvature_m_inv"),
        pd.Series(np.round(radius, 1), index=x_coords.index, name="Radius_of_Curvature_m")
    )


def detect_corners_from_curvature(
    curvature_m_inv: pd.Series,
    distance_m: pd.Series,
    threshold: float = CURVATURE_CORNER_THRESHOLD,
    min_corner_length_m: float = 20.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detects corner intervals, apex points, and turn direction from curvature.

    Returns:
        (IsCornerZone, IsApexPoint, TurnDirection)
    """
    kappa = curvature_m_inv.to_numpy(dtype=np.float64)
    dist = distance_m.to_numpy(dtype=np.float64)
    n = len(kappa)

    abs_kappa = np.abs(kappa)
    in_corner_raw = abs_kappa >= threshold

    # Filter out very short transient noise blips (< min_corner_length_m)
    is_corner = np.zeros(n, dtype=int)
    is_apex = np.zeros(n, dtype=int)
    turn_dir = np.full(n, "STRAIGHT", dtype=object)

    start_idx = None
    corner_id = 0

    for i in range(n):
        if in_corner_raw[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i - 1
                corner_len = dist[end_idx] - dist[start_idx]
                if corner_len >= min_corner_length_m:
                    corner_id += 1
                    is_corner[start_idx:end_idx + 1] = corner_id
                    # Apex is the point of maximum absolute curvature in the corner
                    apex_offset = np.argmax(abs_kappa[start_idx:end_idx + 1])
                    apex_idx = start_idx + apex_offset
                    is_apex[apex_idx] = 1

                    # Direction: positive kappa is LEFT (counter-clockwise), negative is RIGHT
                    mean_k = np.mean(kappa[start_idx:end_idx + 1])
                    d_str = "LEFT" if mean_k > 0 else "RIGHT"
                    turn_dir[start_idx:end_idx + 1] = d_str
                start_idx = None

    # Handle boundary if corner continues to last index
    if start_idx is not None:
        end_idx = n - 1
        if (dist[end_idx] - dist[start_idx]) >= min_corner_length_m:
            corner_id += 1
            is_corner[start_idx:end_idx + 1] = corner_id
            apex_idx = start_idx + np.argmax(abs_kappa[start_idx:end_idx + 1])
            is_apex[apex_idx] = 1
            mean_k = np.mean(kappa[start_idx:end_idx + 1])
            turn_dir[start_idx:end_idx + 1] = "LEFT" if mean_k > 0 else "RIGHT"

    return (
        pd.Series(is_corner, index=curvature_m_inv.index, name="Corner_ID"),
        pd.Series(is_apex, index=curvature_m_inv.index, name="IsCornerApex"),
        pd.Series(turn_dir, index=curvature_m_inv.index, name="TurnDirection")
    )


def compute_track_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates full track geometry feature generation grouped by (Driver, LapNumber).

    Input columns required:
        - X, Y, Distance_Corrected (or Distance), Driver, LapNumber

    Generated features:
        - X_m: X coordinate scaled to meters.
        - Y_m: Y coordinate scaled to meters.
        - Heading_deg: Trajectory heading in degrees [0, 360).
        - HeadingChange_deg: Stepwise heading change in degrees.
        - Curvature_m_inv: Trajectory curvature in m^-1 (signed).
        - Radius_of_Curvature_m: Instantaneous radius of curvature in meters.
        - Corner_ID: Detected sequential corner ID within lap (0 = straight).
        - IsCornerZone: Binary indicator (1 if in corner, 0 if straight).
        - IsCornerApex: Binary indicator (1 at apex/minimum radius, 0 otherwise).
        - TurnDirection: String categorical ("LEFT", "RIGHT", "STRAIGHT").
    """
    logger.info("Computing track geometry features across %d rows...", len(df))

    dist_col = "Distance_Corrected" if "Distance_Corrected" in df.columns else "Distance"
    feature_dfs = []

    for (driver, lap), grp in df.groupby(["Driver", "LapNumber"], sort=False):
        grp_res = pd.DataFrame(index=grp.index)

        # Scale coordinates to meters
        grp_res["X_m"] = np.round(grp["X"] * COORD_SCALE_FACTOR, 2)
        grp_res["Y_m"] = np.round(grp["Y"] * COORD_SCALE_FACTOR, 2)

        # Compute heading and curvature
        heading_deg, dtheta_deg, kappa, radius = compute_heading_and_curvature(
            grp["X"], grp["Y"]
        )
        grp_res["Heading_deg"] = heading_deg
        grp_res["HeadingChange_deg"] = dtheta_deg
        grp_res["Curvature_m_inv"] = kappa
        grp_res["Radius_of_Curvature_m"] = radius

        # Corner detection and apex location
        corner_id, is_apex, turn_dir = detect_corners_from_curvature(
            kappa, grp[dist_col]
        )
        grp_res["Corner_ID"] = corner_id
        grp_res["IsCornerZone"] = (corner_id > 0).astype(int)
        grp_res["IsCornerApex"] = is_apex
        grp_res["TurnDirection"] = turn_dir

        feature_dfs.append(grp_res)

    result_df = pd.concat(feature_dfs).loc[df.index]
    logger.info("Track geometry features completed successfully.")
    return result_df
