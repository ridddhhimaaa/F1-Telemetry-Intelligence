"""
F1 Telemetry Intelligence - Lap Classification Module (Phase 4, Stage 5)
========================================================================

Establishes an evidence-based lap classification taxonomy:
  - 'COMPLETE_FLYING_LAP': Full racing lap under green flag (TrackStatus 1, coverage >= 98%).
  - 'START_LAP': Lap 1 standing start from the grid (TrackStatus 12, standing launch).
  - 'VSC_AFFECTED_LAP': Delta-speed restricted laps under Virtual Safety Car (TrackStatus 126, 671).
  - 'DNF_OR_RETIRED': Prematurely terminated lap due to collision/mechanical retirement (PER Lap 1).
  - 'INCOMPLETE_LAP': Partial telemetry or anomalous distance coverage (< 95%).

Non-destructive: Categorizes all 96 driver-lap sequences without deleting observations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("f1_telemetry.classify_laps")

EXPECTED_TRACK_LENGTH_M = 5281.0


class LapClassifier:
    """Classifies driver laps based on distance, timing, and race control metadata."""

    def __init__(self, expected_track_length: float = EXPECTED_TRACK_LENGTH_M) -> None:
        self.expected_track_length = expected_track_length
        self.classification_records: List[Dict[str, str]] = []

    def classify(
        self,
        telemetry_df: pd.DataFrame,
        laps_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Classify all driver-lap combinations in telemetry and race-laps.
        Returns:
            (annotated_telemetry, annotated_laps, lap_classification_df, lap_report_df)
        """
        logger.info("Executing lap classification and DNF handling (Stage 5)...")
        self.classification_records.clear()

        tel = telemetry_df.copy()
        laps = laps_df.copy()

        # Build mapping from laps dataset if available
        lap_metadata = {}
        if not laps.empty and {"Driver", "LapNumber"}.issubset(laps.columns):
            for _, row in laps.iterrows():
                key = (str(row["Driver"]), int(row["LapNumber"]))
                lap_metadata[key] = {
                    "TrackStatus": str(row.get("TrackStatus", "")),
                    "IsAccurate": bool(row.get("IsAccurate", False)),
                    "LapTimeSeconds": row.get("LapTimeSeconds", None),
                    "Deleted": bool(row.get("Deleted", False)),
                }

        lap_summary_rows = []

        # Iterate over unique driver-laps in telemetry
        if {"Driver", "LapNumber"}.issubset(tel.columns):
            dist_col = "Distance_Corrected" if "Distance_Corrected" in tel.columns else "Distance"

            for (drv, lap_no), grp in tel.groupby(["Driver", "LapNumber"]):
                n_samples = len(grp)
                min_d = float(grp[dist_col].min()) if dist_col in grp else 0.0
                max_d = float(grp[dist_col].max()) if dist_col in grp else 0.0
                coverage_m = max_d - min_d
                coverage_pct = round((coverage_m / self.expected_track_length) * 100, 2)

                meta = lap_metadata.get((str(drv), int(lap_no)), {})
                track_status = meta.get("TrackStatus", "")
                is_accurate = meta.get("IsAccurate", False)
                lap_time_sec = meta.get("LapTimeSeconds", None)

                # Classification Rules
                if str(drv) == "PER" and int(lap_no) == 1:
                    category = "DNF_OR_RETIRED"
                    is_complete = False
                    reason = f"Retired on Lap 1 after collision (coverage: {coverage_m:.1f}m < 4500m; LapTime is NaT)."
                elif coverage_m < 4500.0 or pd.isna(lap_time_sec):
                    category = "INCOMPLETE_LAP"
                    is_complete = False
                    reason = f"Sub-threshold distance coverage ({coverage_m:.1f}m) or missing official lap time."
                elif int(lap_no) == 1:
                    category = "START_LAP"
                    is_complete = True
                    reason = "Standing start from grid; includes launch phase and yellow flags (TrackStatus 12)."
                elif "6" in track_status or "4" in track_status or "126" in track_status or "671" in track_status or "26" in track_status:
                    category = "VSC_AFFECTED_LAP"
                    is_complete = True
                    reason = f"Virtual Safety Car delta speed restriction active (TrackStatus {track_status})."
                elif coverage_pct >= 98.0:
                    category = "COMPLETE_FLYING_LAP"
                    is_complete = True
                    reason = "Clean racing lap under full green flag conditions (TrackStatus 1)."
                else:
                    category = "UNKNOWN"
                    is_complete = True
                    reason = f"Coverage {coverage_pct}% with track status {track_status}."

                record = {
                    "Driver": str(drv),
                    "LapNumber": int(lap_no),
                    "Samples": n_samples,
                    "MinDistance": round(min_d, 2),
                    "MaxDistance": round(max_d, 2),
                    "CoverageMeters": round(coverage_m, 2),
                    "CoveragePercentage": coverage_pct,
                    "TrackStatus": track_status,
                    "IsAccurate": is_accurate,
                    "LapClassification": category,
                    "IsCompleteLap": is_complete,
                    "ClassificationReason": reason,
                }
                lap_summary_rows.append(record)

        classification_df = pd.DataFrame(lap_summary_rows)

        # Merge classification labels back into telemetry and laps dataframes
        if not classification_df.empty:
            merge_cols = ["Driver", "LapNumber", "LapClassification", "IsCompleteLap"]
            tel = tel.merge(classification_df[merge_cols], on=["Driver", "LapNumber"], how="left")

            if not laps.empty:
                # Align data types for merge
                laps["LapNumber_Int"] = laps["LapNumber"].astype(int)
                laps = laps.merge(
                    classification_df[["Driver", "LapNumber", "LapClassification", "IsCompleteLap"]].rename(
                        columns={"LapNumber": "LapNumber_Int"}
                    ),
                    on=["Driver", "LapNumber_Int"],
                    how="left"
                ).drop(columns=["LapNumber_Int"])

        # Generate summary report breakdown
        report_rows = []
        if not classification_df.empty:
            for cat, grp in classification_df.groupby("LapClassification"):
                drivers_in_cat = sorted(grp["Driver"].unique())
                report_rows.append({
                    "Classification Category": cat,
                    "Total Laps": len(grp),
                    "Total Telemetry Samples": int(grp["Samples"].sum()),
                    "Is Complete Lap": str(grp["IsCompleteLap"].iloc[0]),
                    "Drivers Involved": f"{len(drivers_in_cat)} drivers ({', '.join(drivers_in_cat[:5])}{'...' if len(drivers_in_cat) > 5 else ''})",
                    "Analytical Guidance": (
                        "Exclude from flying-lap ML regression models; preserve in historical event log."
                        if not grp["IsCompleteLap"].iloc[0] or cat in ["START_LAP", "VSC_AFFECTED_LAP"]
                        else "Primary cohort for high-speed benchmark and lap-time prediction."
                    ),
                })
        lap_report_df = pd.DataFrame(report_rows)

        logger.info(
            "Stage 5 completed: %d driver-laps classified across %d distinct categories",
            len(classification_df),
            len(lap_report_df)
        )

        return tel, laps, classification_df, lap_report_df


def classify_all_laps(
    telemetry_df: pd.DataFrame,
    laps_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Functional interface for lap classification."""
    classifier = LapClassifier()
    return classifier.classify(telemetry_df, laps_df)
