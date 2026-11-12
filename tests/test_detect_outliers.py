"""
Unit Tests for Outlier Detection Module (Phase 4, Stage 6)
===========================================================
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
from src.preprocessing.detect_outliers import detect_telemetry_outliers


class TestDetectOutliers(unittest.TestCase):
    """Test suite for non-destructive outlier detection."""

    @classmethod
    def setUpClass(cls):
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.tel_path = cls.raw_dir / "2024_abu_dhabi_race_telemetry.parquet"
        cls.tel_df = load_dataset(cls.tel_path)

    def test_zero_records_deleted(self):
        """Verify outlier detection does not delete any observations."""
        annotated_df, outlier_rep = detect_telemetry_outliers(self.tel_df)
        self.assertEqual(len(annotated_df), len(self.tel_df))
        self.assertIn("IsOutlierFlag", annotated_df.columns)

    def test_rpm_zero_outlier_detection(self):
        """Verify zero RPM observations are audited in outlier report."""
        _, outlier_rep = detect_telemetry_outliers(self.tel_df)
        rpm_outliers = outlier_rep[outlier_rep["Column"] == "RPM"]
        self.assertFalse(rpm_outliers.empty)
        self.assertIn("PER", rpm_outliers["Driver"].values)

    def test_throttle_overtravel_outlier_detection(self):
        """Verify throttle overtravel is audited in outlier report."""
        _, outlier_rep = detect_telemetry_outliers(self.tel_df)
        th_outliers = outlier_rep[outlier_rep["Column"] == "Throttle"]
        self.assertFalse(th_outliers.empty)


if __name__ == "__main__":
    unittest.main()
