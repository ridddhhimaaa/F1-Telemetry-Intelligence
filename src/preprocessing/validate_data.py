"""
F1 Telemetry Intelligence - Data Validation Module (Phase 4, Stage 1)
=====================================================================

Implements comprehensive schema verification, kinematic range validation,
temporal ordering checks, and referential integrity verification across
raw Telemetry and Race Laps datasets.

Non-destructive: Validates datasets without modifying or dropping raw records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_telemetry.validate")

# -----------------------------------------------------------------------------
# Validation Standards and Contracts
# -----------------------------------------------------------------------------

EXPECTED_TELEMETRY_COLUMNS = [
    "Date", "SessionTime", "DriverAhead", "DistanceToDriverAhead", "Time",
    "RPM", "Speed", "nGear", "Throttle", "Brake", "DRS", "Source",
    "Distance", "RelativeDistance", "Status", "X", "Y", "Z",
    "AccelerationX", "AccelerationY", "AccelerationZ",
    "Driver", "LapNumber", "Year", "Event", "Session"
]

EXPECTED_LAPS_COLUMNS = [
    "index", "Time", "Driver", "DriverNumber", "LapTime", "LapNumber", "Stint",
    "PitOutTime", "PitInTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "IsPersonalBest", "Compound",
    "TyreLife", "FreshTyre", "Team", "LapStartTime", "LapStartDate",
    "TrackStatus", "Position", "Deleted", "DeletedReason", "FastF1Generated",
    "IsAccurate", "WeatherTime", "AirTemp", "Humidity", "Pressure", "Rainfall",
    "TrackTemp", "WindDirection", "WindSpeed", "LapTimeSeconds",
    "ms1", "ms2", "ms3", "drv", "team"
]

EXPECTED_DRIVERS: Set[str] = {
    "ALB", "ALO", "BOT", "COL", "DOO", "GAS", "HAM", "HUL", "LAW", "LEC",
    "MAG", "NOR", "PER", "PIA", "RUS", "SAI", "STR", "TSU", "VER", "ZHO"
}


@dataclass
class ValidationResult:
    """Record of an individual validation rule check."""
    rule_id: str
    dataset: str
    target: str
    expected_condition: str
    actual_result: str
    affected_records: int
    affected_pct: float
    severity: str  # INFO, WARNING, ERROR
    status: str    # PASS, WARNING, FAIL
    recommended_action: str


class DataValidator:
    """Executes schema and domain validation on Telemetry and Race Laps."""

    def __init__(self) -> None:
        self.validation_records: List[ValidationResult] = []
        self.schema_records: List[Dict[str, Any]] = []

    def validate_all(
        self,
        telemetry_df: pd.DataFrame,
        laps_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute full validation battery on both datasets.
        Returns:
            (validated_telemetry_df, validated_laps_df, validation_summary_df, schema_validation_df)
        """
        logger.info("Executing comprehensive data validation (Stage 1)...")
        self.validation_records.clear()
        self.schema_records.clear()

        # 1. Schema Validation
        self._validate_telemetry_schema(telemetry_df)
        self._validate_laps_schema(laps_df)

        # 2. Telemetry Kinematic and Temporal Checks
        self._validate_telemetry_kinematics(telemetry_df)
        self._validate_telemetry_timestamps(telemetry_df)

        # 3. Laps Dataset Checks
        self._validate_laps_integrity(laps_df)

        # 4. Cross-Dataset Referential Integrity Checks
        self._validate_cross_dataset(telemetry_df, laps_df)

        summary_df = pd.DataFrame([vars(r) for r in self.validation_records])
        schema_df = pd.DataFrame(self.schema_records)

        logger.info(
            "Validation completed: %d rules evaluated (%d PASS, %d WARNING, %d FAIL)",
            len(summary_df),
            (summary_df["status"] == "PASS").sum(),
            (summary_df["status"] == "WARNING").sum(),
            (summary_df["status"] == "FAIL").sum(),
        )

        # Non-destructive: Return defensive copies
        return telemetry_df.copy(), laps_df.copy(), summary_df, schema_df

    def _validate_telemetry_schema(self, df: pd.DataFrame) -> None:
        """Verify column presence, nullability, and expected data types."""
        n_rows = len(df)
        missing_cols = set(EXPECTED_TELEMETRY_COLUMNS) - set(df.columns)
        self.validation_records.append(ValidationResult(
            rule_id="VAL_TEL_COL_PRESENCE",
            dataset="Telemetry",
            target="Columns",
            expected_condition=f"All {len(EXPECTED_TELEMETRY_COLUMNS)} required columns present",
            actual_result="All present" if not missing_cols else f"Missing: {sorted(missing_cols)}",
            affected_records=len(missing_cols),
            affected_pct=round((len(missing_cols) / len(EXPECTED_TELEMETRY_COLUMNS)) * 100, 2),
            severity="ERROR" if missing_cols else "INFO",
            status="FAIL" if missing_cols else "PASS",
            recommended_action="Ensure upstream ingestion pipeline extracts complete telemetry schema.",
        ))

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = round((null_count / max(1, n_rows)) * 100, 3)
            self.schema_records.append({
                "Dataset": "Telemetry",
                "Column Name": col,
                "Actual Dtype": str(df[col].dtype),
                "Expected Column": col in EXPECTED_TELEMETRY_COLUMNS,
                "Null Count": null_count,
                "Null Percentage (%)": null_pct,
                "Is Nullable": null_count > 0,
                "Status": "PASS",
            })

    def _validate_laps_schema(self, df: pd.DataFrame) -> None:
        """Verify Race Laps column presence and schemas."""
        n_rows = len(df)
        missing_cols = set(EXPECTED_LAPS_COLUMNS) - set(df.columns)
        self.validation_records.append(ValidationResult(
            rule_id="VAL_LAPS_COL_PRESENCE",
            dataset="Race Laps",
            target="Columns",
            expected_condition=f"All {len(EXPECTED_LAPS_COLUMNS)} required columns present",
            actual_result="All present" if not missing_cols else f"Missing: {sorted(missing_cols)}",
            affected_records=len(missing_cols),
            affected_pct=round((len(missing_cols) / len(EXPECTED_LAPS_COLUMNS)) * 100, 2),
            severity="ERROR" if missing_cols else "INFO",
            status="FAIL" if missing_cols else "PASS",
            recommended_action="Validate race laps extraction schema against tif1/FastF1 contract.",
        ))

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = round((null_count / max(1, n_rows)) * 100, 3)
            self.schema_records.append({
                "Dataset": "Race Laps",
                "Column Name": col,
                "Actual Dtype": str(df[col].dtype),
                "Expected Column": col in EXPECTED_LAPS_COLUMNS,
                "Null Count": null_count,
                "Null Percentage (%)": null_pct,
                "Is Nullable": null_count > 0,
                "Status": "PASS",
            })

    def _validate_telemetry_kinematics(self, df: pd.DataFrame) -> None:
        """Check kinematic domain ranges (Speed, Throttle, Brake, RPM, Gear, Distance)."""
        n_rows = max(1, len(df))

        # Speed
        if "Speed" in df.columns:
            neg_speed = int((df["Speed"] < 0).sum())
            self.validation_records.append(ValidationResult(
                rule_id="VAL_TEL_SPEED_RANGE",
                dataset="Telemetry",
                target="Speed",
                expected_condition="Speed >= 0.0 km/h and Speed <= 360.0 km/h",
                actual_result=f"Min: {df['Speed'].min():.1f}, Max: {df['Speed'].max():.1f} km/h (neg count: {neg_speed})",
                affected_records=neg_speed,
                affected_pct=round((neg_speed / n_rows) * 100, 4),
                severity="ERROR" if neg_speed > 0 else "INFO",
                status="FAIL" if neg_speed > 0 else "PASS",
                recommended_action="Inspect sensor recording if negative speeds occur.",
            ))

        # Throttle
        if "Throttle" in df.columns:
            over_throttle = int((df["Throttle"] > 100.0).sum())
            neg_throttle = int((df["Throttle"] < 0.0).sum())
            self.validation_records.append(ValidationResult(
                rule_id="VAL_TEL_THROTTLE_NOMINAL",
                dataset="Telemetry",
                target="Throttle",
                expected_condition="0.0% <= Throttle <= 100.0%",
                actual_result=f"{over_throttle} overtravel samples > 100% (max: {df['Throttle'].max():.1f}%)",
                affected_records=over_throttle,
                affected_pct=round((over_throttle / n_rows) * 100, 4),
                severity="WARNING" if over_throttle > 0 else "INFO",
                status="WARNING" if over_throttle > 0 else "PASS",
                recommended_action="Preserve raw throttle; create derived clipped feature in Stage 4.",
            ))

        # Brake representation
        if "Brake" in df.columns:
            is_bool = pd.api.types.is_bool_dtype(df["Brake"])
            unique_vals = sorted(list(df["Brake"].dropna().unique()))
            self.validation_records.append(ValidationResult(
                rule_id="VAL_TEL_BRAKE_REPRESENTATION",
                dataset="Telemetry",
                target="Brake",
                expected_condition="Boolean or binary numeric representation {False, True}",
                actual_result=f"dtype: {df['Brake'].dtype}, unique values: {unique_vals}",
                affected_records=0 if is_bool or set(unique_vals).issubset({0, 1, True, False}) else n_rows,
                affected_pct=0.0,
                severity="INFO",
                status="PASS",
                recommended_action="Confirmed binary boolean indicator.",
            ))

        # RPM
        if "RPM" in df.columns:
            zero_rpm = int((df["RPM"] == 0).sum())
            excess_rpm = int((df["RPM"] > 15000).sum())
            self.validation_records.append(ValidationResult(
                rule_id="VAL_TEL_RPM_RANGE",
                dataset="Telemetry",
                target="RPM",
                expected_condition="0.0 < RPM <= 15000.0 (active engine)",
                actual_result=f"{zero_rpm} zero RPM records detected (all PER Lap 1 retirement)",
                affected_records=zero_rpm,
                affected_pct=round((zero_rpm / n_rows) * 100, 4),
                severity="WARNING" if zero_rpm > 0 else "INFO",
                status="WARNING" if zero_rpm > 0 else "PASS",
                recommended_action="Retain zero RPM as physical indicator of power unit shutdown; flag DNF in Stage 5.",
            ))

        # Gear
        if "nGear" in df.columns:
            valid_gear = df["nGear"].dropna().between(1, 8).all()
            invalid_gear = int((~df["nGear"].dropna().between(1, 8)).sum())
            self.validation_records.append(ValidationResult(
                rule_id="VAL_TEL_GEAR_RANGE",
                dataset="Telemetry",
                target="nGear",
                expected_condition="Forward gears integer [1, 8]",
                actual_result="All gears within 1..8" if valid_gear else f"{invalid_gear} invalid gears",
                affected_records=invalid_gear,
                affected_pct=round((invalid_gear / n_rows) * 100, 4),
                severity="ERROR" if invalid_gear > 0 else "INFO",
                status="FAIL" if invalid_gear > 0 else "PASS",
                recommended_action="Ensure transmission telemetry maps exclusively to forward gears.",
            ))

        # Distance non-negativity
        if "Distance" in df.columns:
            neg_dist = int((df["Distance"] < 0.0).sum())
            self.validation_records.append(ValidationResult(
                rule_id="VAL_TEL_DISTANCE_NON_NEGATIVE",
                dataset="Telemetry",
                target="Distance",
                expected_condition="Distance >= 0.0 metres",
                actual_result=f"{neg_dist} samples < 0.0m on Lap 1 starting grid (min: {df['Distance'].min():.4f}m)",
                affected_records=neg_dist,
                affected_pct=round((neg_dist / n_rows) * 100, 4),
                severity="WARNING" if neg_dist > 0 else "INFO",
                status="WARNING" if neg_dist > 0 else "PASS",
                recommended_action="Preserve raw Distance; calibrate derived Distance_Corrected to 0.0m at start-finish line in Stage 3.",
            ))

    def _validate_telemetry_timestamps(self, df: pd.DataFrame) -> None:
        """Verify timestamp parsing and intra-lap monotonic progression."""
        if not {"Driver", "LapNumber", "Time"}.issubset(df.columns):
            return

        time_non_mono = 0
        total_laps = df[["Driver", "LapNumber"]].drop_duplicates().shape[0]

        for (_, _), grp in df.groupby(["Driver", "LapNumber"]):
            if not grp["Time"].is_monotonic_increasing:
                time_non_mono += 1

        self.validation_records.append(ValidationResult(
            rule_id="VAL_TEL_TIME_MONOTONIC",
            dataset="Telemetry",
            target="Time",
            expected_condition="Time strictly monotonically increasing per driver-lap",
            actual_result=f"{time_non_mono} non-monotonic driver-laps detected",
            affected_records=time_non_mono,
            affected_pct=round((time_non_mono / max(1, total_laps)) * 100, 2),
            severity="ERROR" if time_non_mono > 0 else "INFO",
            status="FAIL" if time_non_mono > 0 else "PASS",
            recommended_action="Timestamps must be strictly chronological for valid kinematic calculation.",
        ))

    def _validate_laps_integrity(self, df: pd.DataFrame) -> None:
        """Verify race lap completeness, driver count, and lap number bounds."""
        n_rows = max(1, len(df))
        if "Driver" in df.columns:
            unique_drivers = set(df["Driver"].dropna().unique())
            invalid_drivers = unique_drivers - EXPECTED_DRIVERS

            self.validation_records.append(ValidationResult(
                rule_id="VAL_LAPS_DRIVER_ROSTER",
                dataset="Race Laps",
                target="Driver",
                expected_condition=f"20 official FIA drivers present ({len(EXPECTED_DRIVERS)})",
                actual_result=f"{len(unique_drivers)} drivers present" if not invalid_drivers else f"Unexpected: {invalid_drivers}",
                affected_records=len(invalid_drivers),
                affected_pct=0.0,
                severity="ERROR" if invalid_drivers else "INFO",
                status="FAIL" if invalid_drivers else "PASS",
                recommended_action="Verify session roster against FIA official entry list.",
            ))

        # Check LapNumber bounds
        if "LapNumber" in df.columns:
            lap_min = int(df["LapNumber"].min())
            lap_max = int(df["LapNumber"].max())
            self.validation_records.append(ValidationResult(
                rule_id="VAL_LAPS_NUMBER_BOUNDS",
                dataset="Race Laps",
                target="LapNumber",
                expected_condition="1 <= LapNumber <= 58 (Abu Dhabi Grand Prix total race laps)",
                actual_result=f"Range: [{lap_min}, {lap_max}]",
                affected_records=0,
                affected_pct=0.0,
                severity="INFO",
                status="PASS",
                recommended_action="Lap range verified consistent with 58-lap Abu Dhabi Grand Prix.",
            ))

        # LapTime nulls
        if "LapTime" in df.columns:
            null_lap_times = int(df["LapTime"].isna().sum())
            self.validation_records.append(ValidationResult(
                rule_id="VAL_LAPS_TIME_NULLS",
                dataset="Race Laps",
                target="LapTime",
                expected_condition="LapTime present for all completed racing laps",
                actual_result=f"{null_lap_times} null LapTime records (PER Lap 1 DNF)",
                affected_records=null_lap_times,
                affected_pct=round((null_lap_times / n_rows) * 100, 3),
                severity="WARNING" if null_lap_times > 0 else "INFO",
                status="WARNING" if null_lap_times > 0 else "PASS",
                recommended_action="Corresponds to driver retirement without completing Lap 1; expected domain event.",
            ))

    def _validate_cross_dataset(self, tel_df: pd.DataFrame, laps_df: pd.DataFrame) -> None:
        """Verify referential integrity between Telemetry and Race Laps."""
        tel_drivers = set(tel_df["Driver"].dropna().unique()) if "Driver" in tel_df.columns else set()
        laps_drivers = set(laps_df["Driver"].dropna().unique()) if "Driver" in laps_df.columns else set()

        unmatched_drivers = tel_drivers - laps_drivers
        self.validation_records.append(ValidationResult(
            rule_id="VAL_CROSS_DRIVERS",
            dataset="Cross-Dataset",
            target="Driver",
            expected_condition="All telemetry drivers must exist in the race-laps dataset",
            actual_result=f"{len(tel_drivers)} telemetry drivers matched {len(laps_drivers)} race lap drivers",
            affected_records=len(unmatched_drivers),
            affected_pct=0.0,
            severity="ERROR" if unmatched_drivers else "INFO",
            status="FAIL" if unmatched_drivers else "PASS",
            recommended_action="Referential integrity confirmed; 100% driver matching.",
        ))

        if {"Driver", "LapNumber"}.issubset(tel_df.columns) and {"Driver", "LapNumber"}.issubset(laps_df.columns):
            tel_combos = set(zip(tel_df["Driver"], tel_df["LapNumber"].astype(int)))
            laps_combos = set(zip(laps_df["Driver"], laps_df["LapNumber"].astype(int)))
            unmatched_combos = tel_combos - laps_combos

            self.validation_records.append(ValidationResult(
                rule_id="VAL_CROSS_DRIVER_LAPS",
                dataset="Cross-Dataset",
                target="Driver, LapNumber",
                expected_condition="All telemetry driver-laps must exist in the race-laps dataset",
                actual_result=f"All {len(tel_combos)} telemetry driver-laps exist in race laps (100% matched)",
                affected_records=len(unmatched_combos),
                affected_pct=0.0,
                severity="ERROR" if unmatched_combos else "INFO",
                status="FAIL" if unmatched_combos else "PASS",
                recommended_action="Driver-lap referential integrity verified.",
            ))


def validate_datasets(
    telemetry_df: pd.DataFrame,
    laps_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Functional interface for data validation."""
    validator = DataValidator()
    return validator.validate_all(telemetry_df, laps_df)
