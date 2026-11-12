"""
Unit and Integration Tests for F1 Telemetry Data Profiling Module (Phase 3)
===========================================================================

Validates data loading, stage 1-8 operations, error handling, edge cases,
file persistence, and raw data integrity guarantees.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.profile_telemetry import (
    ProfilingContext,
    load_dataset,
    locate_datasets,
    run_profiling,
    stage1_dataset_overview,
    stage2_missing_values,
    stage3_duplicate_analysis,
    stage4_statistical_analysis,
    stage5_driver_analysis,
    stage6_lap_analysis,
    stage7_validity_and_consistency,
    stage8_visual_profiling,
    generate_html_report,
)


class TestF1TelemetryProfiling(unittest.TestCase):
    """Test suite for Phase 3 telemetry profiling pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up paths and pre-calculated baseline hashes."""
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.raw_dir = cls.project_root / "data" / "raw"
        cls.metadata_dir = cls.project_root / "data" / "metadata"
        cls.reports_dir = cls.project_root / "reports" / "data_profiling_report"

        # Pre-computed reference hashes for raw datasets to verify immutability
        cls.expected_hashes = {
            "2024_abu_dhabi_race_laps.csv": "cd92d0a8534374606002b1b39dc51e66185b63fce79f4d67b03cd9ef05abf52c",
            "2024_abu_dhabi_race_laps.parquet": "8f56a5e246e981011090d6a969664f06e1d2915013239db21b94a0e1843be31e",
            "2024_abu_dhabi_race_telemetry.csv": "2104ff24209460a29e821336c73b47ab9310d14d82452fc4bf952af316d09296",
            "2024_abu_dhabi_race_telemetry.parquet": "6fa2d79b98a92741964b03e9cbcfc347928310384e052463b6eef20bf1cde2aa",
        }

    # -------------------------------------------------------------------------
    # 1. Data Loading and File Handling
    # -------------------------------------------------------------------------

    def test_locate_datasets(self):
        """Verify automatic detection of telemetry and laps datasets."""
        tel_path, laps_path, meta_path = locate_datasets(self.raw_dir, self.metadata_dir, preferred_format="parquet")
        self.assertTrue(tel_path.is_file(), f"Telemetry file not found: {tel_path}")
        self.assertTrue(laps_path.is_file(), f"Laps file not found: {laps_path}")
        self.assertTrue(meta_path is None or meta_path.is_file())

    def test_load_actual_parquet_datasets(self):
        """Verify loading actual Parquet files and check expected schemas."""
        tel_path = self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet"
        laps_path = self.raw_dir / "2024_abu_dhabi_race_laps.parquet"

        tel_df = load_dataset(tel_path)
        laps_df = load_dataset(laps_path)

        self.assertEqual(tel_df.shape, (72476, 26))
        self.assertEqual(laps_df.shape, (1035, 46))
        self.assertIn("Speed", tel_df.columns)
        self.assertIn("LapTime", laps_df.columns)

    def test_load_actual_csv_datasets(self):
        """Verify loading actual CSV files matches Parquet rows and columns."""
        tel_csv_path = self.raw_dir / "2024_abu_dhabi_race_telemetry.csv"
        laps_csv_path = self.raw_dir / "2024_abu_dhabi_race_laps.csv"

        tel_csv_df = load_dataset(tel_csv_path)
        laps_csv_df = load_dataset(laps_csv_path)

        self.assertEqual(tel_csv_df.shape, (72476, 26))
        self.assertEqual(laps_csv_df.shape, (1035, 46))

    def test_missing_file_handling(self):
        """Verify that attempting to load a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.raw_dir / "non_existent_file.parquet")

    # -------------------------------------------------------------------------
    # 2. Stage 1 - Stage 7 Core Logic Tests
    # -------------------------------------------------------------------------

    def test_stage1_dataset_overview(self):
        """Verify Stage 1 calculations for dataset overview and column inventory."""
        tel_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet")
        laps_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_laps.parquet")

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet",
            laps_path=self.raw_dir / "2024_abu_dhabi_race_laps.parquet",
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=tel_df,
            laps_df=laps_df,
        )

        summary_df, inv_df = stage1_dataset_overview(ctx)
        self.assertEqual(len(summary_df), 2)
        self.assertEqual(summary_df.loc[0, "Total Rows"], 72476)
        self.assertEqual(summary_df.loc[1, "Total Rows"], 1035)
        self.assertEqual(len(inv_df), 26 + 46)  # 72 total columns

    def test_stage2_missing_values(self):
        """Verify Stage 2 missing value analysis matches actual observations."""
        tel_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet")
        laps_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_laps.parquet")

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet",
            laps_path=self.raw_dir / "2024_abu_dhabi_race_laps.parquet",
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=tel_df,
            laps_df=laps_df,
        )

        mv_df, mv_driver, mv_lap = stage2_missing_values(ctx)

        # Check telemetry specific missing values
        tel_mv = mv_df[mv_df["Dataset"] == "Telemetry"].set_index("Column Name")
        self.assertEqual(tel_mv.loc["DriverAhead", "Missing Count"], 4019)
        self.assertEqual(tel_mv.loc["DistanceToDriverAhead", "Missing Count"], 2957)
        self.assertEqual(tel_mv.loc["Speed", "Missing Count"], 0)
        self.assertEqual(tel_mv.loc["Throttle", "Missing Count"], 0)
        self.assertEqual(tel_mv.loc["RPM", "Missing Count"], 0)

        # Check NOR carries all missing DistanceToDriverAhead
        nor_mv = mv_driver[mv_driver["Driver"] == "NOR"]
        self.assertEqual(nor_mv["DistanceToDriverAhead Missing"].iloc[0], 2957)

    def test_stage3_duplicate_analysis(self):
        """Verify Stage 3 duplicate checks (zero exact duplicates, 64 distance repeats)."""
        tel_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet")
        laps_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_laps.parquet")

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet",
            laps_path=self.raw_dir / "2024_abu_dhabi_race_laps.parquet",
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=tel_df,
            laps_df=laps_df,
        )

        dup_df, dup_details = stage3_duplicate_analysis(ctx)
        exact_check = dup_df[dup_df["Check Type"] == "Exact Row Duplicates"]
        self.assertTrue((exact_check["Duplicate Count"] == 0).all())

        dist_check = dup_df[dup_df["Check Type"] == "Repeated Distance within Driver & Lap"]
        self.assertEqual(dist_check["Duplicate Count"].iloc[0], 64)

    def test_stage4_statistical_analysis(self):
        """Verify Stage 4 numerical statistics and IQR calculations."""
        tel_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet")

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet",
            laps_path=self.raw_dir / "2024_abu_dhabi_race_laps.parquet",
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=tel_df,
            laps_df=pd.DataFrame(),
        )

        stats_df, sig_df = stage4_statistical_analysis(ctx)
        stats_map = stats_df.set_index("Column")

        self.assertAlmostEqual(stats_map.loc["Speed", "Mean"], 188.8164, places=2)
        self.assertAlmostEqual(stats_map.loc["Speed", "Max"], 337.0, places=1)
        self.assertAlmostEqual(stats_map.loc["Throttle", "Max"], 104.0, places=1)
        self.assertEqual(stats_map.loc["RPM", "Min"], 0.0)

    def test_stage5_driver_analysis(self):
        """Verify Stage 5 driver summaries (20 drivers, PER with 1 lap)."""
        tel_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet")

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet",
            laps_path=self.raw_dir / "2024_abu_dhabi_race_laps.parquet",
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=tel_df,
            laps_df=pd.DataFrame(),
        )

        drv_df = stage5_driver_analysis(ctx)
        self.assertEqual(len(drv_df), 20)
        self.assertEqual(drv_df[drv_df["Driver"] == "PER"]["Available Laps"].iloc[0], 1)
        self.assertEqual(drv_df[drv_df["Driver"] == "VER"]["Available Laps"].iloc[0], 5)

    def test_stage6_lap_analysis(self):
        """Verify Stage 6 lap summaries (96 unique driver-laps)."""
        tel_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet")

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet",
            laps_path=self.raw_dir / "2024_abu_dhabi_race_laps.parquet",
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=tel_df,
            laps_df=pd.DataFrame(),
        )

        lap_sum, lap_counts, lap_cov = stage6_lap_analysis(ctx)
        self.assertEqual(len(lap_sum), 96)
        self.assertEqual(len(lap_cov), 96)

        # Verify Perez Lap 1 incomplete status
        per_cov = lap_cov[(lap_cov["Driver"] == "PER") & (lap_cov["LapNumber"] == 1)]
        self.assertEqual(per_cov["Coverage Status"].iloc[0], "Incomplete (Early Retirement / DNF)")
        self.assertAlmostEqual(per_cov["Distance Coverage (m)"].iloc[0], 3932.8, delta=1.0)

    def test_stage7_validity_and_consistency(self):
        """Verify Stage 7 data quality checks and warning flags."""
        tel_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet")
        laps_df = load_dataset(self.raw_dir / "2024_abu_dhabi_race_laps.parquet")

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=self.raw_dir / "2024_abu_dhabi_race_telemetry.parquet",
            laps_path=self.raw_dir / "2024_abu_dhabi_race_laps.parquet",
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=tel_df,
            laps_df=laps_df,
        )

        checks_df, consistency_df, warnings_df = stage7_validity_and_consistency(ctx)

        # Cross-dataset checks pass
        drv_check = checks_df[checks_df["Check Name"] == "Cross-Dataset: Telemetry Drivers in Race Laps"]
        self.assertEqual(drv_check["Result"].iloc[0], "PASS")

        # Warnings include throttle overtravel and zero RPM
        warn_ids = warnings_df["Warning ID"].tolist()
        self.assertIn("WARN_THROTTLE_OVERTRAVEL", warn_ids)
        self.assertIn("WARN_RPM_ZERO_ENGINE_OFF", warn_ids)
        self.assertIn("WARN_DISTANCE_GRID_NEGATIVE", warn_ids)

    # -------------------------------------------------------------------------
    # 3. Edge Case Handling Tests
    # -------------------------------------------------------------------------

    def test_empty_dataframe_handling(self):
        """Verify that profiler handles empty DataFrames without unhandled crashes."""
        empty_tel = pd.DataFrame()
        empty_laps = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ctx = ProfilingContext(
                project_root=tmp_path,
                telemetry_path=tmp_path / "mock_tel.parquet",
                laps_path=tmp_path / "mock_laps.parquet",
                metadata_path=None,
                output_dir=tmp_path,
                charts_dir=tmp_path / "charts",
                telemetry_df=empty_tel,
                laps_df=empty_laps,
            )

            summary_df, inv_df = stage1_dataset_overview(ctx)
            self.assertEqual(summary_df.loc[0, "Total Rows"], 0)

            dup_df, _ = stage3_duplicate_analysis(ctx)
            self.assertEqual(dup_df.loc[0, "Duplicate Count"], 0)

            drv_df = stage5_driver_analysis(ctx)
            self.assertTrue(drv_df.empty)

    def test_missing_expected_columns(self):
        """Verify graceful degradation when expected columns are missing."""
        partial_df = pd.DataFrame({
            "Speed": [100.0, 200.0, 300.0],
            "Time": pd.to_timedelta([1, 2, 3], unit="s")
        })

        ctx = ProfilingContext(
            project_root=self.project_root,
            telemetry_path=Path("mock.parquet"),
            laps_path=Path("mock.parquet"),
            metadata_path=None,
            output_dir=self.reports_dir,
            charts_dir=self.reports_dir / "charts",
            telemetry_df=partial_df,
            laps_df=pd.DataFrame(),
        )

        stats_df, _ = stage4_statistical_analysis(ctx)
        self.assertIn("Speed", stats_df["Column"].values)
        self.assertNotIn("Throttle", stats_df["Column"].values)

    # -------------------------------------------------------------------------
    # 4. Deliverables and Artifact Verification
    # -------------------------------------------------------------------------

    def test_csv_deliverables_exist(self):
        """Verify all 16 specified CSV deliverables exist and are non-empty."""
        expected_csvs = [
            "dataset_summary.csv",
            "column_inventory.csv",
            "missing_values.csv",
            "missing_values_by_driver.csv",
            "missing_values_by_lap.csv",
            "duplicate_analysis.csv",
            "duplicate_details.csv",
            "numerical_statistics.csv",
            "telemetry_signal_summary.csv",
            "driver_summary.csv",
            "lap_summary.csv",
            "lap_sample_counts.csv",
            "lap_coverage_report.csv",
            "data_quality_checks.csv",
            "consistency_report.csv",
            "data_quality_warnings.csv",
        ]
        for csv_name in expected_csvs:
            p = self.reports_dir / csv_name
            self.assertTrue(p.is_file(), f"Expected CSV report missing: {csv_name}")
            self.assertGreater(p.stat().st_size, 0, f"CSV report is empty: {csv_name}")

    def test_chart_deliverables_exist(self):
        """Verify all 14 visual charts exist and are non-empty PNG images."""
        expected_charts = [
            "speed_distribution.png",
            "throttle_distribution.png",
            "brake_usage_distribution.png",
            "rpm_distribution.png",
            "avg_speed_by_driver.png",
            "sample_count_by_driver.png",
            "missing_values_by_column.png",
            "speed_vs_distance_selected_laps.png",
            "gear_distribution.png",
            "lap_avg_speed.png",
            "braking_percentage_by_driver.png",
            "avg_throttle_by_driver.png",
            "sample_counts_by_lap.png",
            "telemetry_correlation_heatmap.png",
        ]
        charts_dir = self.reports_dir / "charts"
        for chart_name in expected_charts:
            p = charts_dir / chart_name
            self.assertTrue(p.is_file(), f"Expected chart missing: {chart_name}")
            self.assertGreater(p.stat().st_size, 1000, f"Chart appears corrupted/empty: {chart_name}")

    def test_html_report_exists_and_contains_sections(self):
        """Verify that the HTML report exists and contains essential sections."""
        html_file = self.reports_dir / "profiling_report.html"
        self.assertTrue(html_file.is_file())
        self.assertGreater(html_file.stat().st_size, 5000)

        content = html_file.read_text(encoding="utf-8")
        self.assertIn("Formula 1 Telemetry Intelligence", content)
        self.assertIn("1. Dataset Overview &amp; Inventory", content)
        self.assertIn("2. Missing Value Investigation", content)
        self.assertIn("3. Duplicate Analysis", content)
        self.assertIn("4. Statistical &amp; Telemetry Signal Profiling", content)
        self.assertIn("5. Driver-Wise Telemetry Analysis", content)
        self.assertIn("6. Lap-Wise Analysis &amp; Spatial Distance Coverage", content)
        self.assertIn("7. Data Validity &amp; Cross-Dataset Consistency", content)
        self.assertIn("8. Visual Profiling Gallery", content)
        self.assertIn("9. Recommended Preprocessing Actions", content)

    # -------------------------------------------------------------------------
    # 5. Data Integrity Guarantee (Immutability Check)
    # -------------------------------------------------------------------------

    def test_raw_datasets_unmodified(self):
        """Verify SHA256 checksums of raw datasets to guarantee 100% data immutability."""
        for file_name, expected_hash in self.expected_hashes.items():
            path = self.raw_dir / file_name
            self.assertTrue(path.is_file(), f"Raw file missing: {file_name}")
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"DATA MUTATION DETECTED: {file_name} checksum altered from {expected_hash} to {current_hash}!"
            )


if __name__ == "__main__":
    unittest.main()
