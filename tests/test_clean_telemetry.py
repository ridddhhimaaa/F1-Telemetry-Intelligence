"""
Unit Tests for Telemetry Distance & Signal Cleaning (Phase 4, Stages 3 & 4)
============================================================================
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.profile_telemetry import load_dataset
from src.preprocessing.clean_telemetry import clean_telemetry_signals


class TestCleanTelemetry(unittest.TestCase):
    """Test suite for distance calibration and throttle/brake signal cleaning."""

    @classmethod
    def setUpClass(cls):
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.tel_path = cls.raw_dir / "2024_abu_dhabi_race_telemetry.parquet"
        cls.tel_df = load_dataset(cls.tel_path)

    def test_raw_distance_immutability(self):
        """Verify original Distance column is strictly preserved."""
        cleaned_df, dist_rep, _ = clean_telemetry_signals(self.tel_df)
        self.assertTrue((cleaned_df["Distance"] == self.tel_df["Distance"]).all())

    def test_negative_distance_calibration(self):
        """Verify negative starting grid offsets are calibrated in Distance_Corrected."""
        cleaned_df, dist_rep, _ = clean_telemetry_signals(self.tel_df)
        self.assertIn("Distance_Corrected", cleaned_df.columns)
        self.assertIn("Distance_Anomaly_Flag", cleaned_df.columns)

        # Distance_Corrected must be strictly non-negative
        self.assertGreaterEqual(cleaned_df["Distance_Corrected"].min(), 0.0)

        # Raw distance had negative values
        self.assertLess(cleaned_df["Distance"].min(), 0.0)

    def test_distance_monotonicity_enforcement(self):
        """
        Verify non-monotonic steps are eliminated and Distance_Corrected is monotonically non-decreasing.
        Note: Pandas is_monotonic_increasing checks non-decreasing (Δd >= 0), which permits stationary
        plateaus (Δd == 0) observed during starting grid holds and jitter clamping.
        """
        cleaned_df, _, _ = clean_telemetry_signals(self.tel_df)
        non_mono_laps = 0
        total_zero_deltas = 0
        for (_, _), grp in cleaned_df.groupby(["Driver", "LapNumber"]):
            diffs = np.diff(grp["Distance_Corrected"].values)
            if not (diffs >= 0.0).all():
                non_mono_laps += 1
            total_zero_deltas += int(np.sum(diffs == 0.0))

        # All 96 laps are non-decreasing
        self.assertEqual(non_mono_laps, 0)
        # Exactly 152 legitimate zero-delta occurrences (91 grid stationary holds + 61 jitter clamps)
        self.assertEqual(total_zero_deltas, 152)

    def test_throttle_clipping_and_flags(self):
        """Verify throttle overtravel is flagged and clipped safely."""
        cleaned_df, _, signal_rep = clean_telemetry_signals(self.tel_df)
        self.assertIn("Throttle_Clipped", cleaned_df.columns)
        self.assertIn("Throttle_Overtravel_Flag", cleaned_df.columns)

        # Raw throttle has values up to 104.0%
        self.assertEqual(cleaned_df["Throttle"].max(), 104.0)

        # Clipped throttle is strictly bounded in [0, 100]
        self.assertLessEqual(cleaned_df["Throttle_Clipped"].max(), 100.0)
        self.assertGreaterEqual(cleaned_df["Throttle_Clipped"].min(), 0.0)

        # Exactly 32 samples flagged
        self.assertEqual(cleaned_df["Throttle_Overtravel_Flag"].sum(), 32)

    def test_brake_features_derivation(self):
        """Verify IsBraking and Brake_Int derivation."""
        cleaned_df, _, _ = clean_telemetry_signals(self.tel_df)
        self.assertIn("IsBraking", cleaned_df.columns)
        self.assertIn("Brake_Int", cleaned_df.columns)

        self.assertTrue(pd.api.types.is_bool_dtype(cleaned_df["IsBraking"]))
        self.assertTrue(set(cleaned_df["Brake_Int"].unique()).issubset({0, 1}))
        self.assertEqual(cleaned_df["Brake_Int"].sum(), 14089)


if __name__ == "__main__":
    unittest.main()
