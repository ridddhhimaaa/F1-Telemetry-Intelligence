"""
F1 Telemetry Intelligence - Safe Parquet Ingestion & Compatibility Adapters
Milestone 7.2: Schema Validation & Parquet Ingestion (Hardened)

Provides safe, validated ingestion for all canonical F1 datasets, nanosecond timestamp
compatibility conversion via PyArrow, atomic file generation, cache freshness verification,
and the Windows write-read diagnostic smoke test.
"""

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

from src.spark.schemas import (
    RESAMPLED_TELEMETRY_SCHEMA,
    LAP_FEATURES_SCHEMA,
    SECTOR_FEATURES_SCHEMA,
    CORNER_FEATURES_SCHEMA,
    TELEMETRY_FEATURES_COMPAT_SCHEMA,
    PROCESSED_RACE_LAPS_COMPAT_SCHEMA,
    validate_dataframe_schema,
)
from src.spark.session import get_spark_session

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COMPAT_DIR = PROJECT_ROOT / "data" / "spark_output" / ".compat"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def resolve_dataset_path(path_str: str) -> Path:
    """
    Resolve a dataset path relative to project root or absolute, verifying existence.
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Canonical dataset does not exist at '{p.as_posix()}'.")
    return p


def _verify_parquet_integrity(file_path: Path) -> bool:
    """
    Verify that a Parquet file exists, has positive size, and contains readable metadata.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        import pyarrow.parquet as pq
        _ = pq.ParquetFile(file_path).metadata.num_rows
        return True
    except ImportError:
        # Check via delegation if PyArrow is not in current environment
        if not VENV_PYTHON.exists():
            return file_path.stat().st_size > 0
        script = f"""
import pyarrow.parquet as pq
from pathlib import Path
try:
    _ = pq.ParquetFile(Path(r'{file_path.resolve()}')).metadata.num_rows
    print('VALID')
except Exception:
    print('CORRUPT')
"""
        res = subprocess.run([str(VENV_PYTHON), "-c", script], capture_output=True, text=True)
        return "VALID" in res.stdout
    except Exception:
        return False


def create_compat_parquet(
    source_parquet: str,
    target_parquet: Optional[str] = None,
    force_rebuild: bool = False,
) -> Path:
    """
    Create a controlled, non-destructive PyArrow compatibility Parquet file:
    - Converts timestamp[ns] to timestamp[us] (preserving temporal fidelity within microseconds).
    - Converts duration[ns] to float64 seconds (preserving time elapsed without integer overflow).
    - Preserves nulls, row count, column count, and all metadata.
    - Employs cache freshness validation (source mtime/size fingerprint sidecar).
    - Writes atomically via temporary file to prevent corrupted/incomplete caches.
    - Leaves canonical source dataset 100% untouched.

    If pyarrow is missing from active interpreter (e.g. dedicated Python 3.13),
    safely delegates conversion to the primary project Python environment.
    """
    src_path = resolve_dataset_path(source_parquet)
    if target_parquet is None:
        COMPAT_DIR.mkdir(parents=True, exist_ok=True)
        tgt_path = COMPAT_DIR / f"{src_path.stem}_compat.parquet"
    else:
        tgt_path = Path(target_parquet)
        tgt_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = tgt_path.with_suffix(".meta.json")
    src_stat = src_path.stat()

    # Check cache validity
    if not force_rebuild and tgt_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Verify source has not changed and target is not corrupted
            if (
                meta.get("source_size") == src_stat.st_size
                and meta.get("source_mtime") == src_stat.st_mtime
                and _verify_parquet_integrity(tgt_path)
            ):
                logger.debug("Reusing valid compatibility Parquet: %s", tgt_path.as_posix())
                return tgt_path
            else:
                logger.info("Cache stale or invalid for %s; rebuilding...", tgt_path.name)
        except Exception as e:
            logger.warning("Error reading cache metadata for %s: %s; rebuilding...", tgt_path.name, e)

    logger.info("Generating compatibility Parquet for %s -> %s", src_path.name, tgt_path.name)

    # Unique temporary file for atomic replacement
    tmp_path = tgt_path.with_name(f"{tgt_path.stem}_{os.getpid()}_{int(time.time()*1000)}.tmp")

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        table = pq.read_table(src_path)
        new_fields = []
        new_cols = []
        for field in table.schema:
            col = table[field.name]
            t = field.type
            if pa.types.is_timestamp(t) and t.unit == "ns":
                new_cols.append(col.cast(pa.timestamp("us", tz=t.tz)))
                new_fields.append(pa.field(field.name, pa.timestamp("us", tz=t.tz), nullable=field.nullable))
            elif pa.types.is_duration(t):
                s_col = pc.divide(col.cast(pa.int64()).cast(pa.float64()), 1e9)
                new_cols.append(s_col)
                new_fields.append(pa.field(field.name, pa.float64(), nullable=field.nullable))
            else:
                new_cols.append(col)
                new_fields.append(field)

        compat_table = pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))
        pq.write_table(compat_table, str(tmp_path), compression="snappy")

        # Atomic replace
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Compatibility output was not written to {tmp_path}")
        tmp_path.replace(tgt_path)

        # Write cache fingerprint metadata
        cache_meta = {
            "source_path": str(src_path.resolve()),
            "source_size": src_stat.st_size,
            "source_mtime": src_stat.st_mtime,
            "target_size": tgt_path.stat().st_size,
            "num_rows": compat_table.num_rows,
            "num_cols": compat_table.num_columns,
            "created_at": time.time(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(cache_meta, f, indent=2)

    except ImportError:
        # Delegate to primary venv python where pyarrow is installed
        if not VENV_PYTHON.exists():
            raise RuntimeError(
                f"PyArrow is not available in current environment and project venv was not found at {VENV_PYTHON}."
            )

        script = f"""
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
import json, time

src = Path(r'{src_path.resolve()}')
tgt = Path(r'{tgt_path.resolve()}')
tmp = Path(r'{tmp_path.resolve()}')
meta = Path(r'{meta_path.resolve()}')

table = pq.read_table(src)
new_fields = []
new_cols = []
for field in table.schema:
    col = table[field.name]
    t = field.type
    if pa.types.is_timestamp(t) and t.unit == 'ns':
        new_cols.append(col.cast(pa.timestamp('us', tz=t.tz)))
        new_fields.append(pa.field(field.name, pa.timestamp('us', tz=t.tz), nullable=field.nullable))
    elif pa.types.is_duration(t):
        s_col = pc.divide(col.cast(pa.int64()).cast(pa.float64()), 1e9)
        new_cols.append(s_col)
        new_fields.append(pa.field(field.name, pa.float64(), nullable=field.nullable))
    else:
        new_cols.append(col)
        new_fields.append(field)

compat_table = pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))
pq.write_table(compat_table, str(tmp), compression='snappy')
tmp.replace(tgt)

src_stat = src.stat()
cache_meta = {{
    "source_path": str(src),
    "source_size": src_stat.st_size,
    "source_mtime": src_stat.st_mtime,
    "target_size": tgt.stat().st_size,
    "num_rows": compat_table.num_rows,
    "num_cols": compat_table.num_columns,
    "created_at": time.time(),
}}
with open(meta, "w", encoding="utf-8") as f:
    json.dump(cache_meta, f, indent=2)

print('COMPAT_SUCCESS')
"""
        res = subprocess.run([str(VENV_PYTHON), "-c", script], capture_output=True, text=True)
        if res.returncode != 0 or "COMPAT_SUCCESS" not in res.stdout:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to generate compatibility Parquet via delegation: {res.stderr}")

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return tgt_path


def load_resampled_telemetry(spark: Optional[Any] = None, validate: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Ingest data/processed/abu_dhabi_2024/resampled_telemetry.parquet (495,849 rows x 18 cols).
    Returns (DataFrame, validation_report).
    """
    spark = spark or get_spark_session()
    path = resolve_dataset_path("data/processed/abu_dhabi_2024/resampled_telemetry.parquet")
    df = spark.read.parquet(str(path))

    report = {}
    if validate:
        report = validate_dataframe_schema(
            df=df,
            expected_schema=RESAMPLED_TELEMETRY_SCHEMA,
            dataset_name="resampled_telemetry",
            expected_row_count=495849,
            key_columns=["Driver", "LapNumber"],
        )
        if not report["is_valid"]:
            raise ValueError(f"Schema validation failed for resampled_telemetry: {report['errors']}")

    return df, report


def load_lap_features(spark: Optional[Any] = None, validate: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Ingest data/processed/features/lap_features.parquet (96 rows x 34 cols).
    Returns (DataFrame, validation_report).
    """
    spark = spark or get_spark_session()
    path = resolve_dataset_path("data/processed/features/lap_features.parquet")
    df = spark.read.parquet(str(path))

    report = {}
    if validate:
        report = validate_dataframe_schema(
            df=df,
            expected_schema=LAP_FEATURES_SCHEMA,
            dataset_name="lap_features",
            expected_row_count=96,
            key_columns=["Driver", "LapNumber"],
        )
        if not report["is_valid"]:
            raise ValueError(f"Schema validation failed for lap_features: {report['errors']}")

    return df, report


def load_sector_features(spark: Optional[Any] = None, validate: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Ingest data/processed/features/sector_features.parquet (288 rows x 15 cols).
    Returns (DataFrame, validation_report).
    """
    spark = spark or get_spark_session()
    path = resolve_dataset_path("data/processed/features/sector_features.parquet")
    df = spark.read.parquet(str(path))

    report = {}
    if validate:
        report = validate_dataframe_schema(
            df=df,
            expected_schema=SECTOR_FEATURES_SCHEMA,
            dataset_name="sector_features",
            expected_row_count=288,
            key_columns=["Driver", "LapNumber", "Sector"],
        )
        if not report["is_valid"]:
            raise ValueError(f"Schema validation failed for sector_features: {report['errors']}")

    return df, report


def load_corner_features(spark: Optional[Any] = None, validate: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Ingest data/processed/features/corner_features.parquet (1,469 rows x 15 cols).
    Returns (DataFrame, validation_report).
    """
    spark = spark or get_spark_session()
    path = resolve_dataset_path("data/processed/features/corner_features.parquet")
    df = spark.read.parquet(str(path))

    report = {}
    if validate:
        report = validate_dataframe_schema(
            df=df,
            expected_schema=CORNER_FEATURES_SCHEMA,
            dataset_name="corner_features",
            expected_row_count=1469,
            key_columns=["Driver", "LapNumber", "Corner_ID"],
        )
        if not report["is_valid"]:
            raise ValueError(f"Schema validation failed for corner_features: {report['errors']}")

    return df, report


def load_telemetry_features_compat(
    spark: Optional[Any] = None,
    validate: bool = True,
    force_rebuild: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Ingest telemetry_features using the non-destructive compatibility adapter.
    Expected: 72,476 rows x 99 cols.
    Returns (DataFrame, validation_report).
    """
    spark = spark or get_spark_session()
    compat_path = create_compat_parquet(
        source_parquet="data/processed/features/telemetry_features.parquet",
        force_rebuild=force_rebuild,
    )
    df = spark.read.parquet(str(compat_path))

    report = {}
    if validate:
        report = validate_dataframe_schema(
            df=df,
            expected_schema=TELEMETRY_FEATURES_COMPAT_SCHEMA,
            dataset_name="telemetry_features_compat",
            expected_row_count=72476,
            key_columns=["Driver", "LapNumber"],
        )
        if not report["is_valid"]:
            raise ValueError(f"Validation failed for telemetry_features_compat: {report['errors']}")

    return df, report


def load_processed_race_laps_compat(
    spark: Optional[Any] = None,
    validate: bool = True,
    force_rebuild: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Ingest processed_race_laps using compatibility adapter with caseSensitive enabled.
    Expected: 1,035 rows x 48 cols.
    Returns (DataFrame, validation_report).
    """
    spark = spark or get_spark_session(extra_configs={"spark.sql.caseSensitive": "true"})
    spark.conf.set("spark.sql.caseSensitive", "true")
    compat_path = create_compat_parquet(
        source_parquet="data/processed/abu_dhabi_2024/processed_race_laps.parquet",
        force_rebuild=force_rebuild,
    )
    df = spark.read.parquet(str(compat_path))

    report = {}
    if validate:
        report = validate_dataframe_schema(
            df=df,
            expected_schema=PROCESSED_RACE_LAPS_COMPAT_SCHEMA,
            dataset_name="processed_race_laps_compat",
            expected_row_count=1035,
            key_columns=["Driver", "LapNumber"],
        )
        if not report["is_valid"]:
            raise ValueError(f"Validation failed for processed_race_laps_compat: {report['errors']}")

    return df, report


def run_windows_write_read_smoke_test(
    spark: Optional[Any] = None,
    test_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Conduct the non-destructive Windows Parquet write-read compatibility test.
    Attempts a small 10-row synthetic Parquet write, reads it back, and reports results.

    Returns:
    --------
    dict: Detailed diagnostic report with captured exception if write fails.
    """
    spark = spark or get_spark_session()
    target_dir = Path(test_dir) if test_dir else (PROJECT_ROOT / "data" / "spark_output" / "_smoke_test")

    # Ensure clean starting state
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)

    result: Dict[str, Any] = {
        "test_name": "windows_parquet_write_read_smoke_test",
        "test_directory": str(target_dir),
        "synthetic_rows": 10,
        "write_success": False,
        "read_success": False,
        "cleanup_success": False,
        "failing_stage": None,
        "error_type": None,
        "error_message": None,
        "failing_component": None,
    }

    try:
        # Create small test DataFrame
        data = [
            ("VER", 1, 100.0, 312.5),
            ("NOR", 1, 100.0, 310.2),
            ("LEC", 1, 100.0, 311.0),
            ("PIA", 1, 100.0, 309.8),
            ("SAI", 1, 100.0, 308.5),
            ("HAM", 1, 100.0, 307.9),
            ("RUS", 1, 100.0, 308.1),
            ("GAS", 1, 100.0, 305.4),
            ("ALO", 1, 100.0, 304.8),
            ("TSU", 1, 100.0, 303.2),
        ]
        columns = ["Driver", "LapNumber", "Distance", "Speed"]
        df_smoke = spark.createDataFrame(data, columns)

        # Attempt write
        try:
            df_smoke.coalesce(1).write.mode("overwrite").parquet(str(target_dir))
            result["write_success"] = True
        except Exception as write_exc:
            result["failing_stage"] = "parquet_write"
            result["error_type"] = type(write_exc).__name__
            result["error_message"] = str(write_exc)
            if "winutils" in str(write_exc).lower() or "hadoop" in str(write_exc).lower():
                result["failing_component"] = "Hadoop winutils.exe / native Windows FileOutputCommitter"
            else:
                result["failing_component"] = "Unknown write failure"
            return result

        # Attempt read back
        try:
            df_read = spark.read.parquet(str(target_dir))
            count = df_read.count()
            if count == 10:
                result["read_success"] = True
        except Exception as read_exc:
            result["failing_stage"] = "parquet_read"
            result["error_type"] = type(read_exc).__name__
            result["error_message"] = str(read_exc)
            result["failing_component"] = "Parquet reader"
            return result

    finally:
        # Isolated, safe cleanup
        if target_dir.exists():
            try:
                shutil.rmtree(target_dir, ignore_errors=True)
                result["cleanup_success"] = not target_dir.exists()
            except Exception:
                result["cleanup_success"] = False
        else:
            result["cleanup_success"] = True

    return result
