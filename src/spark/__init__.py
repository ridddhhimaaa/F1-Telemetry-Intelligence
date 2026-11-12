"""
F1 Telemetry Intelligence - Apache Spark Distributed Processing Core
Milestone 7.1 & 7.2: Session Factory, Schema Validation & Parquet Ingestion (Hardened)
"""

from src.spark.session import (
    get_spark_session,
    stop_spark_session,
    get_spark_diagnostics,
    SPARK_PYTHON_EXE,
)
from src.spark.schemas import (
    RESAMPLED_TELEMETRY_SCHEMA,
    LAP_FEATURES_SCHEMA,
    SECTOR_FEATURES_SCHEMA,
    CORNER_FEATURES_SCHEMA,
    TELEMETRY_FEATURES_COMPAT_SCHEMA,
    PROCESSED_RACE_LAPS_COMPAT_SCHEMA,
    validate_dataframe_schema,
)
from src.spark.ingestion import (
    load_resampled_telemetry,
    load_lap_features,
    load_sector_features,
    load_corner_features,
    load_telemetry_features_compat,
    load_processed_race_laps_compat,
    create_compat_parquet,
    run_windows_write_read_smoke_test,
)

__all__ = [
    "get_spark_session",
    "stop_spark_session",
    "get_spark_diagnostics",
    "SPARK_PYTHON_EXE",
    "RESAMPLED_TELEMETRY_SCHEMA",
    "LAP_FEATURES_SCHEMA",
    "SECTOR_FEATURES_SCHEMA",
    "CORNER_FEATURES_SCHEMA",
    "TELEMETRY_FEATURES_COMPAT_SCHEMA",
    "PROCESSED_RACE_LAPS_COMPAT_SCHEMA",
    "validate_dataframe_schema",
    "load_resampled_telemetry",
    "load_lap_features",
    "load_sector_features",
    "load_corner_features",
    "load_telemetry_features_compat",
    "load_processed_race_laps_compat",
    "create_compat_parquet",
    "run_windows_write_read_smoke_test",
]
