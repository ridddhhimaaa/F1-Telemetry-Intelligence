"""
Unit Tests for Missing-Value Handling Module (Phase 4, Stage 2)
===============================================================
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
from src.preprocessing.handle_missing_values import handle_missing_values


class TestMissingValues(unittest.TestCase):
    """Test suite for missing-value treatment and structural derivations."""

    @classmethod
    def setUpClass(cls):
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.tel_path = cls.raw_dir / "2024_abu_dhabi_race_telemetry.parquet"
        cls.laps_path = cls.raw_dir / "2024_abu_dhabi_race_laps.parquet"
        cls.tel_df = load_dataset(cls.tel_path)
        cls.laps_df = load_dataset(cls.laps_path)

    def test_raw_missing_preservation(self):
        """Verify original missing values are preserved (never replaced with 0 or mean)."""
        tel_proc, laps_proc, treatment_df = handle_missing_values(self.tel_df, self.laps_df)
        self.assertEqual(tel_proc["DistanceToDriverAhead"].isna().sum(), 2957)
        self.assertEqual(tel_proc["DriverAhead"].isna().sum(), 4019)

    def test_is_leading_race_derivation(self):
        """Verify IsLeadingRace boolean indicator is derived accurately for NOR."""
        tel_proc, _, _ = handle_missing_values(self.tel_df, self.laps_df)
        self.assertIn("IsLeadingRace", tel_proc.columns)

        # Norris on Laps 2-5 must be confirmed race leader
        nor_lead = tel_proc[(tel_proc["Driver"] == "NOR") & (tel_proc["LapNumber"].isin([2, 3, 4, 5]))]
        self.assertTrue(nor_lead["IsLeadingRace"].all())

        # Non-NOR drivers must not be flagged as leading
        non_nor = tel_proc[tel_proc["Driver"] != "NOR"]
        self.assertFalse(non_nor["IsLeadingRace"].any())

    def test_timing_line_boundary_flag(self):
        """Verify IsTimingLineBoundary flag identifies start-finish boundary samples."""
        tel_proc, _, _ = handle_missing_values(self.tel_df, self.laps_df)
        self.assertIn("IsTimingLineBoundary", tel_proc.columns)
        boundary_pts = tel_proc[tel_proc["IsTimingLineBoundary"]]
        self.assertTrue((boundary_pts["Distance"] <= 25.0).all())

    def test_treatment_report_generation(self):
        """Verify missing value treatment report records justified policies."""
        _, _, treatment_df = handle_missing_values(self.tel_df, self.laps_df)
        self.assertGreaterEqual(len(treatment_df), 4)
        cols_treated = treatment_df["Column Name"].tolist()
        self.assertIn("DistanceToDriverAhead", cols_treated)
        self.assertIn("DriverAhead", cols_treated)


if __name__ == "__main__":
    unittest.main()
