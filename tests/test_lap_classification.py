"""
Unit Tests for Lap Classification Module (Phase 4, Stage 5)
============================================================
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
from src.preprocessing.classify_laps import classify_all_laps


class TestLapClassification(unittest.TestCase):
    """Test suite for lap taxonomy, Perez DNF handling, and lap completeness."""

    @classmethod
    def setUpClass(cls):
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.tel_path = cls.raw_dir / "2024_abu_dhabi_race_telemetry.parquet"
        cls.laps_path = cls.raw_dir / "2024_abu_dhabi_race_laps.parquet"
        cls.tel_df = load_dataset(cls.tel_path)
        cls.laps_df = load_dataset(cls.laps_path)

    def test_perez_lap1_dnf_classification(self):
        """Verify Sergio Perez Lap 1 is classified as DNF_OR_RETIRED."""
        tel_annot, laps_annot, class_df, rep_df = classify_all_laps(self.tel_df, self.laps_df)

        per_lap1 = class_df[(class_df["Driver"] == "PER") & (class_df["LapNumber"] == 1)]
        self.assertEqual(len(per_lap1), 1)
        self.assertEqual(per_lap1["LapClassification"].iloc[0], "DNF_OR_RETIRED")
        self.assertFalse(per_lap1["IsCompleteLap"].iloc[0])

        # Perez samples in telemetry must be tagged as incomplete
        per_tel = tel_annot[tel_annot["Driver"] == "PER"]
        self.assertFalse(per_tel["IsCompleteLap"].all())

    def test_standing_start_lap1_classification(self):
        """Verify non-retiring drivers on Lap 1 are classified as START_LAP."""
        _, _, class_df, _ = classify_all_laps(self.tel_df, self.laps_df)
        non_per_lap1 = class_df[(class_df["Driver"] != "PER") & (class_df["LapNumber"] == 1)]
        self.assertEqual(len(non_per_lap1), 19)
        self.assertTrue((non_per_lap1["LapClassification"] == "START_LAP").all())
        self.assertTrue(non_per_lap1["IsCompleteLap"].all())

    def test_vsc_affected_laps_classification(self):
        """Verify Laps 2 and 3 under VSC are classified as VSC_AFFECTED_LAP."""
        _, _, class_df, _ = classify_all_laps(self.tel_df, self.laps_df)
        vsc_laps = class_df[class_df["LapNumber"].isin([2, 3])]
        self.assertTrue((vsc_laps["LapClassification"] == "VSC_AFFECTED_LAP").all())

    def test_complete_flying_laps_classification(self):
        """Verify green-flag racing laps (Laps 4 and 5) are classified as COMPLETE_FLYING_LAP."""
        _, _, class_df, _ = classify_all_laps(self.tel_df, self.laps_df)
        flying_laps = class_df[class_df["LapNumber"].isin([4, 5])]
        self.assertTrue((flying_laps["LapClassification"] == "COMPLETE_FLYING_LAP").all())


if __name__ == "__main__":
    unittest.main()
