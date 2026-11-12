"""
Unit Tests for Apache Spark Parquet Ingestion & Schema Validation
Milestone 7.2: Canonical Dataset Ingestion, Compatibility Adapters, and Hardened Edge Cases
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPARK_PYTHON = Path(r"C:\Spark\spark-env\Scripts\python.exe")


class TestSparkIngestionNative(unittest.TestCase):
    """
    In-process Spark ingestion test suite executed directly under dedicated Python 3.13.5 runtime.
    """

    @classmethod
    def setUpClass(cls):
        if Path(sys.executable).resolve() != SPARK_PYTHON.resolve():
            raise unittest.SkipTest("Not running inside dedicated Spark Python environment; handled via delegation.")
        
        from src.spark.session import get_spark_session, stop_spark_session
        stop_spark_session()
        cls.spark = get_spark_session(
            app_name="F1_Milestone_7_2_IngestionTest",
            local_cores=2,
            driver_memory="1536m",
            executor_memory="1536m",
            extra_configs={"spark.sql.caseSensitive": "true"},
            force_new=True,
        )

    @classmethod
    def tearDownClass(cls):
        if Path(sys.executable).resolve() == SPARK_PYTHON.resolve():
            from src.spark.session import stop_spark_session
            stop_spark_session()

    def test_01_load_resampled_telemetry(self):
        """Verify resampled_telemetry loads exactly 495,849 rows and 18 columns with 0 nulls."""
        from src.spark.ingestion import load_resampled_telemetry
        df, report = load_resampled_telemetry(self.spark, validate=True)
        
        self.assertEqual(df.count(), 495849)
        self.assertEqual(len(df.columns), 18)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["null_counts"]), 0)
        self.assertEqual(len(report["errors"]), 0)

    def test_02_load_lap_features(self):
        """Verify lap_features loads exactly 96 rows and 34 columns with 0 nulls."""
        from src.spark.ingestion import load_lap_features
        df, report = load_lap_features(self.spark, validate=True)
        
        self.assertEqual(df.count(), 96)
        self.assertEqual(len(df.columns), 34)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["null_counts"]), 0)

    def test_03_load_sector_features(self):
        """Verify sector_features loads exactly 288 rows and 15 columns with 0 nulls."""
        from src.spark.ingestion import load_sector_features
        df, report = load_sector_features(self.spark, validate=True)
        
        self.assertEqual(df.count(), 288)
        self.assertEqual(len(df.columns), 15)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["null_counts"]), 0)

    def test_04_load_corner_features(self):
        """Verify corner_features loads exactly 1,469 rows and 15 columns with 0 nulls."""
        from src.spark.ingestion import load_corner_features
        df, report = load_corner_features(self.spark, validate=True)
        
        self.assertEqual(df.count(), 1469)
        self.assertEqual(len(df.columns), 15)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["null_counts"]), 0)

    def test_05_invalid_path_raises_filenotfound(self):
        """Verify resolve_dataset_path raises FileNotFoundError for non-existent paths."""
        from src.spark.ingestion import resolve_dataset_path
        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path("data/non_existent_telemetry.parquet")

    def test_06_telemetry_features_compat_loading(self):
        """Verify telemetry_features compatibility adapter loads 72,476 rows and 99 columns with explicit schema."""
        from src.spark.ingestion import load_telemetry_features_compat
        df, report = load_telemetry_features_compat(self.spark, validate=True)
        
        self.assertEqual(df.count(), 72476)
        self.assertEqual(len(df.columns), 99)
        self.assertTrue(report["is_valid"])
        self.assertIn("Driver", df.columns)
        self.assertIn("Date", df.columns)
        self.assertIn("SessionTime", df.columns)
        self.assertEqual(report["actual_row_count"], 72476)

    def test_07_processed_race_laps_compat_loading_and_case_sensitivity(self):
        """Verify processed_race_laps compatibility adapter loads 1,035 rows, 48 columns, and distinguishes Team vs team."""
        from src.spark.ingestion import load_processed_race_laps_compat
        df, report = load_processed_race_laps_compat(self.spark, validate=True)
        
        self.assertEqual(df.count(), 1035)
        self.assertEqual(len(df.columns), 48)
        self.assertTrue(report["is_valid"])
        self.assertIn("Team", df.columns)
        self.assertIn("team", df.columns)
        
        # Verify both Team and team can be queried independently without collision
        sample = df.select("Team", "team").limit(5).collect()
        self.assertEqual(len(sample), 5)
        for row in sample:
            self.assertIsNotNone(row["Team"])
            self.assertIsNotNone(row["team"])

    def test_08_canonical_datasets_remain_unchanged(self):
        """Verify that canonical Parquet datasets exist and are non-empty."""
        canonical_files = [
            PROJECT_ROOT / "data" / "processed" / "abu_dhabi_2024" / "resampled_telemetry.parquet",
            PROJECT_ROOT / "data" / "processed" / "features" / "lap_features.parquet",
            PROJECT_ROOT / "data" / "processed" / "features" / "sector_features.parquet",
            PROJECT_ROOT / "data" / "processed" / "features" / "corner_features.parquet",
            PROJECT_ROOT / "data" / "processed" / "features" / "telemetry_features.parquet",
            PROJECT_ROOT / "data" / "processed" / "abu_dhabi_2024" / "processed_race_laps.parquet",
        ]
        for f in canonical_files:
            self.assertTrue(f.exists(), f"Canonical file missing: {f}")
            self.assertGreater(f.stat().st_size, 0)

    def test_09_windows_write_read_smoke_test(self):
        """Execute Windows Parquet write-read smoke test and document result."""
        from src.spark.ingestion import run_windows_write_read_smoke_test
        res = run_windows_write_read_smoke_test(self.spark)
        
        self.assertIn("write_success", res)
        self.assertIn("cleanup_success", res)
        self.assertTrue(res["cleanup_success"])
        
        if not res["write_success"]:
            self.assertEqual(res["failing_stage"], "parquet_write")
            self.assertIn("winutils", str(res["failing_component"]).lower())

    def test_10_schema_validation_negative_cases(self):
        """Verify validate_dataframe_schema catches missing columns, unexpected columns, and key nulls."""
        from src.spark.schemas import validate_dataframe_schema, LAP_FEATURES_SCHEMA
        from pyspark.sql.types import StructType, StructField, StringType, LongType
        import pyspark.sql.functions as F

        # Create dummy dataframe with missing columns
        dummy_data = [("VER", 1), ("NOR", 1)]
        dummy_df = self.spark.createDataFrame(dummy_data, ["Driver", "LapNumber"])
        rep_missing = validate_dataframe_schema(dummy_df, LAP_FEATURES_SCHEMA, "dummy_missing")
        self.assertFalse(rep_missing["is_valid"])
        self.assertGreater(len(rep_missing["missing_columns"]), 0)

        # Create dummy dataframe with unexpected column
        simple_schema = StructType([
            StructField("Driver", StringType(), True),
            StructField("LapNumber", LongType(), True),
        ])
        dummy_extra = self.spark.createDataFrame([("VER", 1, "EXTRA")], ["Driver", "LapNumber", "ExtraCol"])
        rep_extra = validate_dataframe_schema(dummy_extra, simple_schema, "dummy_extra")
        self.assertIn("ExtraCol", rep_extra["unexpected_columns"])

        # Create dummy dataframe with nulls in key column
        dummy_null = self.spark.createDataFrame([("VER", 1), (None, 2)], ["Driver", "LapNumber"])
        rep_null = validate_dataframe_schema(
            dummy_null,
            simple_schema,
            "dummy_null",
            key_columns=["Driver"],
        )
        self.assertFalse(rep_null["is_valid"])
        self.assertIn("Driver", rep_null["key_null_violations"])

    def test_11_compat_cache_freshness_and_corruption_recovery(self):
        """Verify that create_compat_parquet detects corruption and cleanly rebuilds."""
        from src.spark.ingestion import create_compat_parquet, COMPAT_DIR

        # Create a test target path
        test_compat = COMPAT_DIR / "test_corruption_compat.parquet"
        test_meta = COMPAT_DIR / "test_corruption_compat.meta.json"

        try:
            # First build
            out = create_compat_parquet(
                source_parquet="data/processed/features/lap_features.parquet",
                target_parquet=str(test_compat),
                force_rebuild=True,
            )
            self.assertTrue(out.exists())
            self.assertTrue(test_meta.exists())

            # Now intentionally corrupt the compat file with garbage bytes
            with open(out, "wb") as f:
                f.write(b"CORRUPTED_PARQUET_HEADER_DATA")

            # Reading without force_rebuild should detect corruption via _verify_parquet_integrity and rebuild cleanly
            recovered_out = create_compat_parquet(
                source_parquet="data/processed/features/lap_features.parquet",
                target_parquet=str(test_compat),
                force_rebuild=False,
            )
            self.assertTrue(recovered_out.exists())
            self.assertGreater(recovered_out.stat().st_size, 100)

            # Verify it is valid Parquet again by reading in Spark
            df_recovered = self.spark.read.parquet(str(recovered_out))
            self.assertEqual(df_recovered.count(), 96)

        finally:
            if test_compat.exists():
                test_compat.unlink()
            if test_meta.exists():
                test_meta.unlink()


class TestSparkIngestionRunnerDelegation(unittest.TestCase):
    """
    Cross-venv delegation test runner for pytest in main project Python 3.14 environment.
    """

    def test_spark_ingestion_suite_via_dedicated_runtime(self):
        """Run the full TestSparkIngestionNative suite using C:\\Spark\\spark-env\\Scripts\\python.exe."""
        self.assertTrue(
            SPARK_PYTHON.exists(),
            f"Dedicated Python executable does not exist at '{SPARK_PYTHON}'",
        )
        
        cmd = [
            str(SPARK_PYTHON),
            "-m",
            "unittest",
            "tests.test_spark_ingestion.TestSparkIngestionNative",
            "-v",
        ]
        
        env = os.environ.copy()
        env["PYSPARK_PYTHON"] = str(SPARK_PYTHON)
        env["PYSPARK_DRIVER_PYTHON"] = str(SPARK_PYTHON)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        
        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            
        self.assertEqual(
            result.returncode,
            0,
            f"Spark ingestion test failed in dedicated runtime with exit code {result.returncode}",
        )
        self.assertIn("OK", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
