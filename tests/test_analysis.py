"""
F1 Telemetry Intelligence - Analysis & EDA Unit Tests (Phase 5)
==============================================================

Verifies that all analytical functions, statistical summaries, filtering routines,
and data loaders execute correctly and preserve dataset immutability.
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.analysis.data_loader import (
    load_processed_telemetry,
    load_processed_laps,
    load_resampled_telemetry,
    load_all_datasets,
    filter_clean_flying_laps,
    filter_by_driver,
    filter_by_lap
)
from src.analysis.dataset_overview import (
    compute_dataset_dimensions,
    compute_lap_classification_summary,
    compute_sampling_frequency_stats
)
from src.analysis.data_quality import (
    audit_missing_values,
    audit_sensor_boundaries,
    audit_anomaly_flags
)
from src.analysis.driver_analysis import (
    compute_driver_telemetry_metrics,
    compute_driver_lap_time_stats
)
from src.analysis.speed_analysis import (
    compute_speed_percentiles_by_driver,
    analyze_drs_speed_impact
)
from src.analysis.braking_analysis import (
    detect_circuit_braking_zones,
    compare_turn6_braking_points
)
from src.analysis.throttle_analysis import (
    compute_throttle_regimes_by_driver
)
from src.analysis.gear_rpm_analysis import (
    compute_gear_usage_statistics,
    compute_gear_shifts_per_lap
)
from src.analysis.lap_comparison import (
    compute_driver_lap_delta
)
from src.analysis.spatial_analysis import (
    compute_sector_performance_summary
)
from src.analysis.correlation_analysis import (
    compute_telemetry_correlation_matrix
)
from src.analysis.ml_eda import (
    evaluate_candidate_targets,
    audit_feature_suitability,
    assess_data_leakage_risks,
    evaluate_dataset_splitting_strategies
)


class TestAnalysisModules(unittest.TestCase):
    """Test suite for Phase 5 reusable analysis modules."""

    @classmethod
    def setUpClass(cls):
        cls.tel_df, cls.laps_df, cls.res_df = load_all_datasets()
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.expected_hashes = {
            "2024_abu_dhabi_race_laps.csv": "cd92d0a8534374606002b1b39dc51e66185b63fce79f4d67b03cd9ef05abf52c",
            "2024_abu_dhabi_race_laps.parquet": "8f56a5e246e981011090d6a969664f06e1d2915013239db21b94a0e1843be31e",
            "2024_abu_dhabi_race_telemetry.csv": "2104ff24209460a29e821336c73b47ab9310d14d82452fc4bf952af316d09296",
            "2024_abu_dhabi_race_telemetry.parquet": "6fa2d79b98a92741964b03e9cbcfc347928310384e052463b6eef20bf1cde2aa"
        }

    def test_data_loader_shapes_and_types(self):
        """Verify loaded datasets match expected row and column counts."""
        self.assertEqual(len(self.tel_df), 72476)
        self.assertEqual(len(self.laps_df), 1035)
        self.assertEqual(len(self.res_df), 495849)

        # Check key columns
        self.assertIn("Distance_Corrected", self.tel_df.columns)
        self.assertIn("LapClassification", self.tel_df.columns)
        self.assertIn("Distance_Meter", self.res_df.columns)

    def test_filter_clean_flying_laps(self):
        """Verify filtering strictly returns COMPLETE_FLYING_LAP."""
        flying = filter_clean_flying_laps(self.tel_df)
        self.assertTrue((flying["LapClassification"] == "COMPLETE_FLYING_LAP").all())
        # Exactly 38 driver-lap sequences
        n_flying_laps = len(flying.groupby(["Driver", "LapNumber"]))
        self.assertEqual(n_flying_laps, 38)

    def test_dataset_overview_metrics(self):
        """Verify dimensions, lap classifications, and sampling frequencies."""
        dim_df = compute_dataset_dimensions(self.tel_df, self.laps_df, self.res_df)
        self.assertEqual(len(dim_df), 3)

        class_summary = compute_lap_classification_summary(self.tel_df)
        self.assertIn("COMPLETE_FLYING_LAP", class_summary["Classification"].values)
        self.assertEqual(class_summary["DriverLapSequences"].sum(), 96)

        samp = compute_sampling_frequency_stats(self.tel_df)
        self.assertAlmostEqual(samp["Median_DeltaT_s"], 0.125, delta=0.03)

    def test_data_quality_audits(self):
        """Verify missing values, sensor bounds, and anomaly flags."""
        nulls = audit_missing_values(self.tel_df, "Telemetry")
        self.assertIn("DistanceToDriverAhead", nulls["Column"].values)

        bounds = audit_sensor_boundaries(self.tel_df)
        self.assertIn("Speed", bounds["Channel"].values)
        self.assertIn("Throttle_Clipped", bounds["Channel"].values)

        flags = audit_anomaly_flags(self.tel_df)
        self.assertEqual(len(flags), 20)  # 20 drivers

    def test_driver_performance_metrics(self):
        """Verify driver metrics and pace rankings."""
        metrics = compute_driver_telemetry_metrics(self.tel_df, only_flying_laps=True)
        self.assertEqual(len(metrics), 19)  # 19 drivers with flying laps (PER retired on L1)
        self.assertIn("MeanSpeed_kmh", metrics.columns)
        self.assertIn("FullThrottle_pct", metrics.columns)

        pace = compute_driver_lap_time_stats(self.laps_df, only_accurate_green=True)
        self.assertGreater(len(pace), 0)
        self.assertIn("BestLapTime_s", pace.columns)

    def test_speed_and_drs_analysis(self):
        """Verify speed percentiles and DRS evaluation."""
        sp_pct = compute_speed_percentiles_by_driver(self.tel_df, only_flying=True)
        self.assertEqual(len(sp_pct), 19)
        self.assertGreater(sp_pct["Max_kmh"].max(), 320.0)

        drs = analyze_drs_speed_impact(self.tel_df)
        self.assertGreater(drs["DRS_Active_MeanSpeed"], drs["DRS_Inactive_MeanSpeed"])

    def test_braking_zones_detection(self):
        """Verify circuit braking zones detection and Turn 6 comparison."""
        bz = detect_circuit_braking_zones(self.res_df, lap_number=5)
        self.assertGreaterEqual(len(bz), 4)
        self.assertIn("EntrySpeed_kmh", bz.columns)

        t6 = compare_turn6_braking_points(self.res_df, lap_number=5)
        self.assertGreater(len(t6), 15)
        self.assertIn("BrakingPoint_m", t6.columns)

    def test_throttle_and_gear_rpm_analysis(self):
        """Verify throttle regimes and gear usage stats."""
        regimes = compute_throttle_regimes_by_driver(self.tel_df, only_flying=True)
        self.assertEqual(len(regimes), 19)

        gear_stats = compute_gear_usage_statistics(self.tel_df, only_flying=True)
        self.assertEqual(len(gear_stats), 8)  # Gears 1 to 8

        shifts = compute_gear_shifts_per_lap(self.tel_df)
        self.assertEqual(len(shifts), 96)

    def test_lap_comparison_delta(self):
        """Verify head-to-head spatial delta calculation."""
        delta = compute_driver_lap_delta(self.res_df, "NOR", "VER", lap_number=5)
        self.assertGreater(len(delta), 5000)
        self.assertIn("Delta_Speed_kmh", delta.columns)
        self.assertIn("Cumulative_Time_Delta_s", delta.columns)

    def test_spatial_and_correlation_analysis(self):
        """Verify sector summary and correlation matrix computation."""
        sec = compute_sector_performance_summary(self.res_df, lap_number=5)
        self.assertEqual(len(sec), 3)

        corr = compute_telemetry_correlation_matrix(self.tel_df, method="pearson", only_flying=True)
        self.assertIn("Speed", corr.columns)
        self.assertIn("Throttle_Clipped", corr.columns)
        # Speed and Throttle should have moderate-to-strong positive correlation (> 0.50)
        self.assertGreater(corr.loc["Speed", "Throttle_Clipped"], 0.50)

    def test_ml_eda_evaluations(self):
        """Verify ML target evaluation, feature suitability, leakage, and splits."""
        targets = evaluate_candidate_targets(self.tel_df, self.laps_df)
        self.assertGreaterEqual(len(targets), 4)

        features = audit_feature_suitability(self.tel_df)
        self.assertEqual(len(features), len(self.tel_df.columns))

        leakage = assess_data_leakage_risks(self.tel_df, self.laps_df)
        self.assertGreaterEqual(len(leakage), 4)

        splits = evaluate_dataset_splitting_strategies(self.tel_df)
        self.assertEqual(len(splits), 4)

    def test_raw_data_immutability(self):
        """Verify SHA-256 hashes of all raw files remain identical."""
        for fname, exp_hash in self.expected_hashes.items():
            fpath = self.raw_dir / fname
            actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, exp_hash, f"Raw data modified: {fname}")


if __name__ == "__main__":
    unittest.main()
