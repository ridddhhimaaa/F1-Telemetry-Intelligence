"""
Unit Tests for Data Validation Module (Phase 4, Stage 1)
=========================================================
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
from src.preprocessing.validate_data import validate_datasets, DataValidator


class TestDataValidation(unittest.TestCase):
    """Test suite for schema and domain validation logic."""

    @classmethod
    def setUpClass(cls):
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.tel_path = cls.raw_dir / "2024_abu_dhabi_race_telemetry.parquet"
        cls.laps_path = cls.raw_dir / "2024_abu_dhabi_race_laps.parquet"
        cls.tel_df = load_dataset(cls.tel_path)
        cls.laps_df = load_dataset(cls.laps_path)

    def test_actual_datasets_validation(self):
        """Validate actual raw telemetry and race-lap datasets."""
        val_tel, val_laps, summary_df, schema_df = validate_datasets(self.tel_df, self.laps_df)
        self.assertEqual(len(val_tel), 72476)
        self.assertEqual(len(val_laps), 1035)

        # Confirm rules passed
        col_presence = summary_df[summary_df["rule_id"] == "VAL_TEL_COL_PRESENCE"]
        self.assertEqual(col_presence["status"].iloc[0], "PASS")

        drivers_check = summary_df[summary_df["rule_id"] == "VAL_LAPS_DRIVER_ROSTER"]
        self.assertEqual(drivers_check["status"].iloc[0], "PASS")

        cross_drivers = summary_df[summary_df["rule_id"] == "VAL_CROSS_DRIVERS"]
        self.assertEqual(cross_drivers["status"].iloc[0], "PASS")

        cross_laps = summary_df[summary_df["rule_id"] == "VAL_CROSS_DRIVER_LAPS"]
        self.assertEqual(cross_laps["status"].iloc[0], "PASS")

    def test_missing_required_column_detection(self):
        """Verify validator flags missing columns as ERROR/FAIL."""
        corrupted_tel = self.tel_df.drop(columns=["Speed", "RPM"])
        _, _, summary_df, _ = validate_datasets(corrupted_tel, self.laps_df)
        col_rule = summary_df[summary_df["rule_id"] == "VAL_TEL_COL_PRESENCE"]
        self.assertEqual(col_rule["status"].iloc[0], "FAIL")
        self.assertEqual(col_rule["severity"].iloc[0], "ERROR")

    def test_negative_speed_detection(self):
        """Verify validator detects negative speed."""
        synthetic_tel = self.tel_df.head(50).copy()
        synthetic_tel.loc[0, "Speed"] = -15.0
        _, _, summary_df, _ = validate_datasets(synthetic_tel, self.laps_df)
        speed_rule = summary_df[summary_df["rule_id"] == "VAL_TEL_SPEED_RANGE"]
        self.assertEqual(speed_rule["status"].iloc[0], "FAIL")
        self.assertEqual(speed_rule["affected_records"].iloc[0], 1)

    def test_throttle_overtravel_detection(self):
        """Verify validator flags throttle overtravel as WARNING."""
        _, _, summary_df, _ = validate_datasets(self.tel_df, self.laps_df)
        throttle_rule = summary_df[summary_df["rule_id"] == "VAL_TEL_THROTTLE_NOMINAL"]
        self.assertEqual(throttle_rule["status"].iloc[0], "WARNING")
        self.assertEqual(throttle_rule["severity"].iloc[0], "WARNING")
        self.assertEqual(throttle_rule["affected_records"].iloc[0], 32)

    def test_empty_dataset_handling(self):
        """Verify validator handles empty datasets gracefully."""
        empty_tel = pd.DataFrame()
        empty_laps = pd.DataFrame()
        _, _, summary_df, schema_df = validate_datasets(empty_tel, empty_laps)
        self.assertFalse(summary_df.empty)


if __name__ == "__main__":
    unittest.main()
