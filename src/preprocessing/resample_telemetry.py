"""
F1 Telemetry Intelligence - Spatial Resampling Module (Phase 4, Stage 7)
========================================================================

Resamples asynchronous time-domain telemetry onto a uniform 1.0-metre track
distance grid across complete racing laps.

Purpose:
  - Enables direct driver-to-driver telemetry delta overlays (e.g. Turn 5 apex speed,
    Turn 6 braking point distance) at identical physical track coordinates.

Methodology:
  - Restricted to valid complete laps (LapClassification == 'COMPLETE_FLYING_LAP' or complete laps).
  - Spatial Grid: Uniform 1.0-metre intervals from 0.0m to 5,280.0m (Yas Marina circuit length).
  - Linear/Monotonic interpolation for continuous channels (Speed, Throttle, RPM, coordinates).
  - Nearest/Step interpolation for discrete channels (Gear, Braking, DRS).
  - Stored as a separate derived analytical dataset: 'resampled_telemetry.parquet'.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger("f1_telemetry.resample")

TRACK_LENGTH_M = 5281.0
GRID_STEP_M = 1.0


class SpatialResampler:
    """Interpolates telemetry onto uniform circuit spatial intervals."""

    def __init__(
        self,
        step_meters: float = GRID_STEP_M,
        track_length: float = TRACK_LENGTH_M
    ) -> None:
        self.step_meters = step_meters
        self.track_length = track_length

    def resample(
        self,
        telemetry_df: pd.DataFrame,
        only_complete_laps: bool = True
    ) -> pd.DataFrame:
        """
        Resample eligible driver laps onto a uniform distance grid.
        Returns:
            resampled_df (pd.DataFrame with 1-meter spatial intervals)
        """
        logger.info("Executing spatial distance resampling (Stage 7)...")

        if telemetry_df.empty:
            logger.warning("Empty telemetry provided for resampling.")
            return pd.DataFrame()

        dist_col = "Distance_Corrected" if "Distance_Corrected" in telemetry_df.columns else "Distance"
        if dist_col not in telemetry_df.columns:
            logger.error("Distance column '%s' absent; cannot resample.", dist_col)
            return pd.DataFrame()

        # Filter eligible laps: strictly complete racing laps
        eligible = telemetry_df.copy()
        if only_complete_laps and "IsCompleteLap" in eligible.columns:
            eligible = eligible[eligible["IsCompleteLap"] == True]

        if eligible.empty:
            logger.warning("No complete laps eligible for spatial resampling.")
            return pd.DataFrame()

        continuous_cols = [
            c for c in [
                "Speed", "Throttle_Clipped", "Throttle", "RPM",
                "X", "Y", "Z", "AccelerationX", "AccelerationY", "AccelerationZ"
            ] if c in eligible.columns
        ]

        discrete_cols = [
            c for c in ["nGear", "IsBraking", "Brake_Int", "DRS"] if c in eligible.columns
        ]

        resampled_frames: List[pd.DataFrame] = []

        for (drv, lap_no), grp in eligible.groupby(["Driver", "LapNumber"], sort=True):
            # Sort and remove duplicate distance coordinates
            grp_clean = grp.sort_values(dist_col).drop_duplicates(subset=[dist_col], keep="first")
            d_vals = grp_clean[dist_col].values

            if len(d_vals) < 10:
                continue

            d_min = float(d_vals[0])
            d_max = float(d_vals[-1])

            # Ensure valid circuit coverage (at least 4800m)
            if (d_max - d_min) < 4800.0:
                continue

            # Standard uniform distance grid strictly bounded within observed data
            grid_start = np.ceil(max(0.0, d_min))
            grid_end = np.floor(min(self.track_length, d_max))
            uniform_dist = np.arange(grid_start, grid_end + self.step_meters, self.step_meters)

            if len(uniform_dist) == 0:
                continue

            resampled_lap = pd.DataFrame({
                "Driver": drv,
                "LapNumber": lap_no,
                "Distance_Meter": uniform_dist,
            })

            # Interpolate continuous kinematics
            for col in continuous_cols:
                f_interp = interp1d(
                    d_vals,
                    grp_clean[col].values.astype(float),
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                resampled_lap[col] = f_interp(uniform_dist).round(2)

            # Interpolate discrete channels (nearest neighbor)
            for col in discrete_cols:
                f_disc = interp1d(
                    d_vals,
                    grp_clean[col].values.astype(float),
                    kind="nearest",
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                resampled_lap[col] = f_disc(uniform_dist).astype(int)

            if "LapClassification" in grp:
                resampled_lap["LapClassification"] = grp["LapClassification"].iloc[0]

            resampled_frames.append(resampled_lap)

        if not resampled_frames:
            logger.warning("No laps met the 4800m spatial coverage criterion for resampling.")
            return pd.DataFrame()

        resampled_df = pd.concat(resampled_frames, ignore_index=True)

        logger.info(
            "Stage 7 completed: %d uniform 1.0m spatial slices generated across %d driver-laps",
            len(resampled_df),
            resampled_df[["Driver", "LapNumber"]].drop_duplicates().shape[0]
        )

        return resampled_df


def resample_telemetry_spatial(
    telemetry_df: pd.DataFrame,
    step_meters: float = GRID_STEP_M
) -> pd.DataFrame:
    """Functional interface for spatial telemetry resampling."""
    resampler = SpatialResampler(step_meters=step_meters)
    return resampler.resample(telemetry_df)
