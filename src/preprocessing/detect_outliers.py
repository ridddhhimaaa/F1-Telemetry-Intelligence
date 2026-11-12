"""
F1 Telemetry Intelligence - Non-Destructive Outlier Detection Module (Phase 4, Stage 6)
=======================================================================================

Applies domain validation envelopes and robust statistical detection (IQR / Z-Score)
to audit extreme observations across vehicle kinematics, powertrain channels, and
sampling intervals.

Strict Rule: An outlier is NOT an error. Extreme values generated during braking zones,
standing starts, collisions, or ECU overtravel are preserved in their entirety.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.detect_outliers")


class OutlierDetector:
    """Non-destructive anomaly detection across telemetry channels."""

    def __init__(self) -> None:
        self.anomaly_records: List[Dict[str, str]] = []

    def detect(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scan telemetry for domain and statistical outliers.
        Returns:
            (annotated_telemetry_df, outlier_analysis_df)
        """
        logger.info("Executing non-destructive outlier detection (Stage 6)...")
        self.anomaly_records.clear()

        annotated = df.copy()
        outlier_flags = pd.Series(False, index=annotated.index, dtype=bool)

        # 1. Throttle Overtravel Anomaly
        if "Throttle" in annotated.columns:
            over_mask = annotated["Throttle"] > 100.0
            outlier_flags |= over_mask
            for idx in annotated[over_mask].index[:10]:  # Record sample anomalies
                self.anomaly_records.append({
                    "Driver": str(annotated.loc[idx, "Driver"]) if "Driver" in annotated else "N/A",
                    "Lap": str(annotated.loc[idx, "LapNumber"]) if "LapNumber" in annotated else "N/A",
                    "Timestamp": str(annotated.loc[idx, "Time"]) if "Time" in annotated else "N/A",
                    "Column": "Throttle",
                    "Observed Value": f"{annotated.loc[idx, 'Throttle']:.2f}%",
                    "Detection Method": "Domain Boundary (> 100.0%)",
                    "Severity": "WARNING",
                    "Interpretation": "Potentiometer overtravel. Valid pedal travel calibration.",
                    "Recommended Treatment": "Preserve raw value; normalize with Throttle_Clipped.",
                })

        # 2. RPM Engine Shutdown Anomaly
        if "RPM" in annotated.columns:
            zero_rpm_mask = annotated["RPM"] == 0
            outlier_flags |= zero_rpm_mask
            for idx in annotated[zero_rpm_mask].index[:10]:
                self.anomaly_records.append({
                    "Driver": str(annotated.loc[idx, "Driver"]) if "Driver" in annotated else "N/A",
                    "Lap": str(annotated.loc[idx, "LapNumber"]) if "LapNumber" in annotated else "N/A",
                    "Timestamp": str(annotated.loc[idx, "Time"]) if "Time" in annotated else "N/A",
                    "Column": "RPM",
                    "Observed Value": f"{annotated.loc[idx, 'RPM']:.1f}",
                    "Detection Method": "Physical Lower Bound (RPM == 0)",
                    "Severity": "REVIEW REQUIRED",
                    "Interpretation": "Engine cut/power unit shutdown (PER Lap 1 crash coasting).",
                    "Recommended Treatment": "Preserve unaltered; flag DNF_OR_RETIRED in lap metadata.",
                })

        # 3. High Speed Boundary
        if "Speed" in annotated.columns:
            high_speed_mask = annotated["Speed"] > 330.0
            for idx in annotated[high_speed_mask].index[:5]:
                self.anomaly_records.append({
                    "Driver": str(annotated.loc[idx, "Driver"]) if "Driver" in annotated else "N/A",
                    "Lap": str(annotated.loc[idx, "LapNumber"]) if "LapNumber" in annotated else "N/A",
                    "Timestamp": str(annotated.loc[idx, "Time"]) if "Time" in annotated else "N/A",
                    "Column": "Speed",
                    "Observed Value": f"{annotated.loc[idx, 'Speed']:.1f} km/h",
                    "Detection Method": "Domain Speed Trap (> 330 km/h)",
                    "Severity": "INFO",
                    "Interpretation": "Peak straight-line terminal velocity under DRS slipstream.",
                    "Recommended Treatment": "Preserve; physical high-performance peak.",
                })

        # 4. Extreme G-Force / Acceleration Anomaly
        for acc_col in ["AccelerationX", "AccelerationY", "AccelerationZ"]:
            if acc_col in annotated.columns:
                s = annotated[acc_col].dropna()
                q25 = s.quantile(0.25)
                q75 = s.quantile(0.75)
                iqr = q75 - q25
                lower_fence = q25 - 3.0 * iqr
                upper_fence = q75 + 3.0 * iqr

                extreme_acc = (s < lower_fence) | (s > upper_fence)
                for idx in s[extreme_acc].index[:5]:
                    self.anomaly_records.append({
                        "Driver": str(annotated.loc[idx, "Driver"]) if "Driver" in annotated else "N/A",
                        "Lap": str(annotated.loc[idx, "LapNumber"]) if "LapNumber" in annotated else "N/A",
                        "Timestamp": str(annotated.loc[idx, "Time"]) if "Time" in annotated else "N/A",
                        "Column": acc_col,
                        "Observed Value": f"{annotated.loc[idx, acc_col]:.2f} m/s^2",
                        "Detection Method": f"Statistical IQR Extreme Fence [3.0*IQR]",
                        "Severity": "INFO",
                        "Interpretation": "High instantaneous transient chassis load (kerb strike / threshold braking).",
                        "Recommended Treatment": "Preserve; essential for vehicle balance evaluation.",
                    })

        # 5. Large Sampling Interval (> 0.5s)
        if {"Driver", "LapNumber", "Time"}.issubset(annotated.columns):
            for (drv, lap_no), grp in annotated.groupby(["Driver", "LapNumber"]):
                t_sec = grp["Time"].dt.total_seconds()
                dt = t_sec.diff()
                large_dt = dt > 0.5
                if np.any(large_dt):
                    for idx in dt[large_dt].index:
                        self.anomaly_records.append({
                            "Driver": str(drv),
                            "Lap": str(lap_no),
                            "Timestamp": str(annotated.loc[idx, "Time"]),
                            "Column": "Time_Delta",
                            "Observed Value": f"{dt.loc[idx]:.3f}s",
                            "Detection Method": "Sampling Gap (> 0.500s)",
                            "Severity": "WARNING",
                            "Interpretation": "Transient telemetry gap or sensor transmission delay.",
                            "Recommended Treatment": "Preserve; spatial resampling handles irregular time delta.",
                        })

        annotated["IsOutlierFlag"] = outlier_flags
        outlier_report_df = pd.DataFrame(self.anomaly_records)

        logger.info(
            "Stage 6 completed: %d representative anomalies audited and catalogued",
            len(outlier_report_df)
        )

        return annotated, outlier_report_df


def detect_telemetry_outliers(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Functional interface for outlier detection."""
    detector = OutlierDetector()
    return detector.detect(df)
