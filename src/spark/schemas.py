"""
F1 Telemetry Intelligence - Explicit Schema Definitions & Validation Utilities
Milestone 7.2: Schema Validation & Parquet Ingestion (Hardened)

This module provides deterministic PySpark StructType definitions derived directly
from canonical and compatibility source metadata, along with comprehensive schema
and null audit tools.
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import pyspark types safely if available in active environment
try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        LongType,
        DoubleType,
        BooleanType,
        TimestampType,
        TimestampNTZType,
        IntegerType,
    )
except ImportError:
    # Minimal fallback stubs to allow schema module inspection from non-Spark environments
    StructType = Any
    StructField = Any
    StringType = LongType = DoubleType = BooleanType = TimestampType = TimestampNTZType = IntegerType = Any


# 1. Primary Dataset: resampled_telemetry.parquet (495,849 rows x 18 cols)
RESAMPLED_TELEMETRY_SCHEMA = StructType([
    StructField("Driver", StringType(), True),
    StructField("LapNumber", LongType(), True),
    StructField("Distance_Meter", DoubleType(), True),
    StructField("Speed", DoubleType(), True),
    StructField("Throttle_Clipped", DoubleType(), True),
    StructField("Throttle", DoubleType(), True),
    StructField("RPM", DoubleType(), True),
    StructField("X", DoubleType(), True),
    StructField("Y", DoubleType(), True),
    StructField("Z", DoubleType(), True),
    StructField("AccelerationX", DoubleType(), True),
    StructField("AccelerationY", DoubleType(), True),
    StructField("AccelerationZ", DoubleType(), True),
    StructField("nGear", LongType(), True),
    StructField("IsBraking", LongType(), True),
    StructField("Brake_Int", LongType(), True),
    StructField("DRS", LongType(), True),
    StructField("LapClassification", StringType(), True),
])

# 2. Secondary Dataset: lap_features.parquet (96 rows x 34 cols)
LAP_FEATURES_SCHEMA = StructType([
    StructField("Driver", StringType(), True),
    StructField("LapNumber", LongType(), True),
    StructField("LapClassification", StringType(), True),
    StructField("LapTime_s", DoubleType(), True),
    StructField("TelemetryDuration_s", DoubleType(), True),
    StructField("LapDistance_m", DoubleType(), True),
    StructField("Speed_Mean_kmh", DoubleType(), True),
    StructField("Speed_Max_kmh", DoubleType(), True),
    StructField("Speed_Min_kmh", DoubleType(), True),
    StructField("Speed_Std_kmh", DoubleType(), True),
    StructField("Speed_Median_kmh", DoubleType(), True),
    StructField("Speed_P25_kmh", DoubleType(), True),
    StructField("Speed_P75_kmh", DoubleType(), True),
    StructField("FullThrottle_Pct", DoubleType(), True),
    StructField("Throttle_Mean_Pct", DoubleType(), True),
    StructField("Throttle_Transitions", LongType(), True),
    StructField("Braking_Duration_s", DoubleType(), True),
    StructField("Braking_Pct", DoubleType(), True),
    StructField("Braking_Events_Count", LongType(), True),
    StructField("Peak_Decel_G", DoubleType(), True),
    StructField("Peak_Accel_G", DoubleType(), True),
    StructField("Mean_Accel_mps2", DoubleType(), True),
    StructField("Gear_Changes_Count", LongType(), True),
    StructField("Upshifts_Count", LongType(), True),
    StructField("Downshifts_Count", LongType(), True),
    StructField("Most_Used_Gear", LongType(), True),
    StructField("Corners_Count", LongType(), True),
    StructField("Mean_Apex_Speed_kmh", DoubleType(), True),
    StructField("Min_Apex_Speed_kmh", DoubleType(), True),
    StructField("Sector1_Duration_s", DoubleType(), True),
    StructField("Sector2_Duration_s", DoubleType(), True),
    StructField("Sector3_Duration_s", DoubleType(), True),
    StructField("Sample_Count", LongType(), True),
    StructField("Sampling_Frequency_Hz", DoubleType(), True),
])

# 3. Secondary Dataset: sector_features.parquet (288 rows x 15 cols)
SECTOR_FEATURES_SCHEMA = StructType([
    StructField("Driver", StringType(), True),
    StructField("LapNumber", LongType(), True),
    StructField("Sector", LongType(), True),
    StructField("SectorDuration_s", DoubleType(), True),
    StructField("Speed_Mean_kmh", DoubleType(), True),
    StructField("Speed_Max_kmh", DoubleType(), True),
    StructField("Speed_Min_kmh", DoubleType(), True),
    StructField("FullThrottle_Pct", DoubleType(), True),
    StructField("Throttle_Mean_Pct", DoubleType(), True),
    StructField("Braking_Duration_s", DoubleType(), True),
    StructField("Peak_Decel_G", DoubleType(), True),
    StructField("Most_Used_Gear", LongType(), True),
    StructField("Gear_Shifts_Count", LongType(), True),
    StructField("Corners_Count", LongType(), True),
    StructField("Samples_Count", LongType(), True),
])

# 4. Secondary Dataset: corner_features.parquet (1,469 rows x 15 cols)
CORNER_FEATURES_SCHEMA = StructType([
    StructField("Driver", StringType(), True),
    StructField("LapNumber", LongType(), True),
    StructField("Corner_ID", LongType(), True),
    StructField("TurnDirection", StringType(), True),
    StructField("StartDistance_m", DoubleType(), True),
    StructField("ApexDistance_m", DoubleType(), True),
    StructField("EndDistance_m", DoubleType(), True),
    StructField("CornerLength_m", DoubleType(), True),
    StructField("EntrySpeed_kmh", DoubleType(), True),
    StructField("ApexSpeed_kmh", DoubleType(), True),
    StructField("ExitSpeed_kmh", DoubleType(), True),
    StructField("SpeedLoss_kmh", DoubleType(), True),
    StructField("ExitAccel_kmh", DoubleType(), True),
    StructField("CornerDuration_s", DoubleType(), True),
    StructField("MinRadius_m", DoubleType(), True),
])

# 5. Compatibility Dataset: telemetry_features_compat.parquet (72,476 rows x 99 cols)
TELEMETRY_FEATURES_COMPAT_SCHEMA = StructType([
    StructField("Date", TimestampNTZType(), True),
    StructField("SessionTime", DoubleType(), True),
    StructField("DriverAhead", StringType(), True),
    StructField("DistanceToDriverAhead", DoubleType(), True),
    StructField("Time", DoubleType(), True),
    StructField("RPM", DoubleType(), True),
    StructField("Speed", DoubleType(), True),
    StructField("nGear", LongType(), True),
    StructField("Throttle", DoubleType(), True),
    StructField("Brake", BooleanType(), True),
    StructField("DRS", LongType(), True),
    StructField("Source", StringType(), True),
    StructField("Distance", DoubleType(), True),
    StructField("RelativeDistance", DoubleType(), True),
    StructField("Status", StringType(), True),
    StructField("X", DoubleType(), True),
    StructField("Y", DoubleType(), True),
    StructField("Z", DoubleType(), True),
    StructField("AccelerationX", DoubleType(), True),
    StructField("AccelerationY", DoubleType(), True),
    StructField("AccelerationZ", DoubleType(), True),
    StructField("Driver", StringType(), True),
    StructField("LapNumber", LongType(), True),
    StructField("Year", LongType(), True),
    StructField("Event", StringType(), True),
    StructField("Session", StringType(), True),
    StructField("IsLeadingRace", BooleanType(), True),
    StructField("IsTimingLineBoundary", BooleanType(), True),
    StructField("Distance_Corrected", DoubleType(), True),
    StructField("Distance_Anomaly_Flag", BooleanType(), True),
    StructField("Throttle_Clipped", DoubleType(), True),
    StructField("Throttle_Overtravel_Flag", BooleanType(), True),
    StructField("IsBraking", BooleanType(), True),
    StructField("Brake_Int", LongType(), True),
    StructField("LapClassification", StringType(), True),
    StructField("IsCompleteLap", BooleanType(), True),
    StructField("IsOutlierFlag", BooleanType(), True),
    StructField("X_m", DoubleType(), True),
    StructField("Y_m", DoubleType(), True),
    StructField("Heading_deg", DoubleType(), True),
    StructField("HeadingChange_deg", DoubleType(), True),
    StructField("Curvature_m_inv", DoubleType(), True),
    StructField("Radius_of_Curvature_m", DoubleType(), True),
    StructField("Corner_ID", LongType(), True),
    StructField("IsCornerZone", LongType(), True),
    StructField("IsCornerApex", LongType(), True),
    StructField("TurnDirection", StringType(), True),
    StructField("Accel_Longitudinal_mps2", DoubleType(), True),
    StructField("G_Longitudinal", DoubleType(), True),
    StructField("SpeedChangeRate_kmh_per_s", DoubleType(), True),
    StructField("Accel_Lateral_mps2", DoubleType(), True),
    StructField("G_Lateral", DoubleType(), True),
    StructField("G_Combined", DoubleType(), True),
    StructField("IsAccelerating", LongType(), True),
    StructField("IsDecelerating", LongType(), True),
    StructField("IsCoasting", LongType(), True),
    StructField("DynamicsRegime", StringType(), True),
    StructField("AccelPhaseDuration_s", DoubleType(), True),
    StructField("DecelPhaseDuration_s", DoubleType(), True),
    StructField("Speed_Rolling_Std_1s", DoubleType(), True),
    StructField("Throttle_Pct", DoubleType(), True),
    StructField("Throttle_dPct_per_s", DoubleType(), True),
    StructField("Throttle_Application_Rate", DoubleType(), True),
    StructField("Throttle_Release_Rate", DoubleType(), True),
    StructField("IsFullThrottle", LongType(), True),
    StructField("IsPartialThrottle", LongType(), True),
    StructField("IsOffThrottle", LongType(), True),
    StructField("FullThrottle_Duration_s", DoubleType(), True),
    StructField("PartialThrottle_Duration_s", DoubleType(), True),
    StructField("OffThrottle_Duration_s", DoubleType(), True),
    StructField("FullThrottle_Cumulative_Pct", DoubleType(), True),
    StructField("Throttle_Transition_Count", LongType(), True),
    StructField("Braking_Event_ID", LongType(), True),
    StructField("IsBrakingOnset", LongType(), True),
    StructField("BrakingDuration_s", DoubleType(), True),
    StructField("BrakingDistance_m", DoubleType(), True),
    StructField("BrakeEvent_Peak_G", DoubleType(), True),
    StructField("IsBrakeReleasePhase", LongType(), True),
    StructField("IsPedalOverlap", LongType(), True),
    StructField("TimeSinceBrakeRelease_s", DoubleType(), True),
    StructField("Corner_Entry_Speed_kmh", DoubleType(), True),
    StructField("Corner_Apex_Speed_kmh", DoubleType(), True),
    StructField("Corner_Exit_Speed_kmh", DoubleType(), True),
    StructField("Corner_Speed_Loss_kmh", DoubleType(), True),
    StructField("Corner_Exit_Accel_kmh", DoubleType(), True),
    StructField("Corner_Traversal_Duration_s", DoubleType(), True),
    StructField("IsPreApexZone", LongType(), True),
    StructField("IsPostApexZone", LongType(), True),
    StructField("Ratio_RPM_per_kmh", DoubleType(), True),
    StructField("RPM_Band_Pct", DoubleType(), True),
    StructField("RPM_Change_Rate_per_s", DoubleType(), True),
    StructField("IsGearShift", LongType(), True),
    StructField("IsUpshift", LongType(), True),
    StructField("IsDownshift", LongType(), True),
    StructField("Gear_Change_Count", LongType(), True),
    StructField("Upshift_Count", LongType(), True),
    StructField("Downshift_Count", LongType(), True),
    StructField("Sector", LongType(), True),
    StructField("Distance_In_Sector_m", DoubleType(), True),
])

# 6. Compatibility Dataset: processed_race_laps_compat.parquet (1,035 rows x 48 cols)
# Note: Spark case-sensitive mode (spark.sql.caseSensitive = true) required to preserve both 'Team' and 'team'
PROCESSED_RACE_LAPS_COMPAT_SCHEMA = StructType([
    StructField("index", LongType(), True),
    StructField("Time", DoubleType(), True),
    StructField("Driver", StringType(), True),
    StructField("DriverNumber", StringType(), True),
    StructField("LapTime", DoubleType(), True),
    StructField("LapNumber", DoubleType(), True),
    StructField("Stint", DoubleType(), True),
    StructField("PitOutTime", DoubleType(), True),
    StructField("PitInTime", DoubleType(), True),
    StructField("Sector1Time", DoubleType(), True),
    StructField("Sector2Time", DoubleType(), True),
    StructField("Sector3Time", DoubleType(), True),
    StructField("Sector1SessionTime", DoubleType(), True),
    StructField("Sector2SessionTime", DoubleType(), True),
    StructField("Sector3SessionTime", DoubleType(), True),
    StructField("SpeedI1", DoubleType(), True),
    StructField("SpeedI2", DoubleType(), True),
    StructField("SpeedFL", DoubleType(), True),
    StructField("SpeedST", DoubleType(), True),
    StructField("IsPersonalBest", BooleanType(), True),
    StructField("Compound", StringType(), True),
    StructField("TyreLife", DoubleType(), True),
    StructField("FreshTyre", BooleanType(), True),
    StructField("Team", StringType(), True),
    StructField("LapStartTime", DoubleType(), True),
    StructField("LapStartDate", TimestampNTZType(), True),
    StructField("TrackStatus", StringType(), True),
    StructField("Position", DoubleType(), True),
    StructField("Deleted", BooleanType(), True),
    StructField("DeletedReason", StringType(), True),
    StructField("FastF1Generated", BooleanType(), True),
    StructField("IsAccurate", BooleanType(), True),
    StructField("WeatherTime", DoubleType(), True),
    StructField("AirTemp", DoubleType(), True),
    StructField("Humidity", DoubleType(), True),
    StructField("Pressure", DoubleType(), True),
    StructField("Rainfall", BooleanType(), True),
    StructField("TrackTemp", DoubleType(), True),
    StructField("WindDirection", LongType(), True),
    StructField("WindSpeed", DoubleType(), True),
    StructField("LapTimeSeconds", DoubleType(), True),
    StructField("ms1", StringType(), True),
    StructField("ms2", StringType(), True),
    StructField("ms3", StringType(), True),
    StructField("drv", StringType(), True),
    StructField("team", StringType(), True),
    StructField("LapClassification", StringType(), True),
    StructField("IsCompleteLap", BooleanType(), True),
])


def validate_dataframe_schema(
    df: Any,
    expected_schema: Any,
    dataset_name: str = "Dataset",
    expected_row_count: Optional[int] = None,
    key_columns: Optional[List[str]] = None,
    required_non_null_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Rigorously audit a Spark DataFrame against an expected StructType schema.

    Checks:
    - Actual vs. expected column count.
    - Missing required columns and unexpected additional columns.
    - Column data type matches.
    - Total row count verification.
    - Explicit non-null constraints on primary keys / required fields.
    - Per-column null value breakdown.

    Returns:
    --------
    dict: Machine-readable audit report.
    """
    import pyspark.sql.functions as F

    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "actual_row_count": 0,
        "actual_column_count": len(df.columns),
        "expected_column_count": len(expected_schema.fields),
        "missing_columns": [],
        "unexpected_columns": [],
        "type_mismatches": [],
        "key_null_violations": {},
        "null_counts": {},
    }

    actual_fields = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    expected_fields = {f.name: f.dataType.simpleString() for f in expected_schema.fields}

    # 1. Column name checks
    missing_cols = [c for c in expected_fields if c not in actual_fields]
    extra_cols = [c for c in actual_fields if c not in expected_fields]

    if missing_cols:
        report["missing_columns"] = missing_cols
        report["errors"].append(f"Missing required columns: {missing_cols}")
        report["is_valid"] = False

    if extra_cols:
        report["unexpected_columns"] = extra_cols
        report["warnings"].append(f"Found unexpected additional columns: {extra_cols}")

    # 2. Data type comparison for matching columns
    for col_name in expected_fields:
        if col_name in actual_fields:
            act_type = actual_fields[col_name]
            exp_type = expected_fields[col_name]
            # Consider integer/long, float/double, timestamp/timestamp_ntz flexible if non-destructive
            if act_type != exp_type:
                mismatch = f"{col_name}: actual={act_type}, expected={exp_type}"
                report["type_mismatches"].append(mismatch)
                # Allow benign timestamp variants or compatible numerics
                is_compatible = (
                    ("int" in act_type and "int" in exp_type) or
                    ("double" in act_type and "float" in exp_type) or
                    ("float" in act_type and "double" in exp_type) or
                    ("timestamp" in act_type and "timestamp" in exp_type)
                )
                if not is_compatible:
                    report["errors"].append(f"Type mismatch: {mismatch}")
                    report["is_valid"] = False

    # 3. Row count check
    actual_rows = df.count()
    report["actual_row_count"] = actual_rows
    if expected_row_count is not None and actual_rows != expected_row_count:
        report["errors"].append(
            f"Row count mismatch: actual {actual_rows:,} != expected {expected_row_count:,}"
        )
        report["is_valid"] = False

    # 4. Null value counts across columns
    null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
    null_row = df.agg(*null_exprs).collect()[0].asDict()
    report["null_counts"] = {k: v for k, v in null_row.items() if v > 0}

    # 5. Non-null validation for primary keys and required columns
    all_required_non_null = list(key_columns or [])
    if required_non_null_columns:
        for col_req in required_non_null_columns:
            if col_req not in all_required_non_null:
                all_required_non_null.append(col_req)

    for k in all_required_non_null:
        if k in null_row and null_row[k] > 0:
            report["key_null_violations"][k] = null_row[k]
            report["errors"].append(f"Required non-null column '{k}' contains {null_row[k]} null values!")
            report["is_valid"] = False

    return report
