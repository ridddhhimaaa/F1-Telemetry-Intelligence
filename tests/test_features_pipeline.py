"""
Tests for Feature Engineering Pipeline Orchestrator (Phase 6)
=============================================================
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.features.feature_pipeline import (
    run_feature_pipeline,
    build_feature_dictionary
)


class TestFeaturePipeline(unittest.TestCase):
    """Integration and validation test suite for Phase 6 Feature Engineering Pipeline."""

    @classmethod
    def setUpClass(cls):
        # Create isolated temporary directory for test dataset output
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls._tmp_dir.name)

        # Run pipeline once into isolated directory
        cls.tel_feat, cls.lap_feat, cls.sec_feat, cls.corn_feat, cls.dict_df = run_feature_pipeline(
            output_dir=cls.test_dir,
            save_datasets=True
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def test_pipeline_output_shapes(self):
        """Verify pipeline output row counts match canonical Phase 4 dimensions."""
        # Telemetry: exactly 72,476 rows
        self.assertEqual(len(self.tel_feat), 72476)
        # 96 driver-laps in lap summary
        self.assertEqual(len(self.lap_feat), 96)
        # Sectors: 96 laps * 3 sectors = 288 sector records (or >= 280 given PER DNF Lap 1)
        self.assertGreaterEqual(len(self.sec_feat), 280)
        # Corners: 16 corners * 96 laps ~ 1500 corner records
        self.assertGreater(len(self.corn_feat), 1000)

    def test_essential_feature_columns_present(self):
        """Verify all mandatory engineered feature columns exist in output DataFrame."""
        mandatory_cols = [
            "Accel_Longitudinal_mps2", "G_Longitudinal", "SpeedChangeRate_kmh_per_s",
            "Accel_Lateral_mps2", "G_Lateral", "G_Combined",
            "IsAccelerating", "IsDecelerating", "IsCoasting", "DynamicsRegime",
            "AccelPhaseDuration_s", "DecelPhaseDuration_s", "Speed_Rolling_Std_1s",
            "X_m", "Y_m", "Heading_deg", "HeadingChange_deg",
            "Curvature_m_inv", "Radius_of_Curvature_m", "Corner_ID", "IsCornerZone", "IsCornerApex",
            "TurnDirection", "Throttle_Pct", "Throttle_dPct_per_s", "Throttle_Application_Rate",
            "Throttle_Release_Rate", "IsFullThrottle", "FullThrottle_Duration_s", "FullThrottle_Cumulative_Pct",
            "Throttle_Transition_Count", "Braking_Event_ID", "IsBrakingOnset", "BrakingDuration_s",
            "BrakingDistance_m", "BrakeEvent_Peak_G", "IsBrakeReleasePhase", "IsPedalOverlap",
            "TimeSinceBrakeRelease_s", "Corner_Entry_Speed_kmh", "Corner_Apex_Speed_kmh",
            "Corner_Exit_Speed_kmh", "Corner_Speed_Loss_kmh", "Corner_Exit_Accel_kmh",
            "Ratio_RPM_per_kmh", "RPM_Band_Pct", "RPM_Change_Rate_per_s", "IsGearShift",
            "IsUpshift", "IsDownshift", "Gear_Change_Count", "Upshift_Count", "Downshift_Count",
            "Sector", "Distance_In_Sector_m"
        ]
        for col in mandatory_cols:
            self.assertIn(col, self.tel_feat.columns, f"Missing mandatory feature column: {col}")

    def test_numerical_stability_no_inf(self):
        """Verify zero infinite values exist across all engineered numerical columns."""
        numeric_cols = self.tel_feat.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(self.tel_feat[col]).sum()
            self.assertEqual(inf_count, 0, f"Column {col} contains infinite values!")

    def test_saved_files_parquet_and_csv_consistency(self):
        """Verify Parquet and CSV files were written and have identical row/column dimensions."""
        for stem in ["telemetry_features", "lap_features", "sector_features", "corner_features"]:
            p_file = self.test_dir / f"{stem}.parquet"
            c_file = self.test_dir / f"{stem}.csv"
            self.assertTrue(p_file.is_file(), f"Missing {p_file.name}")
            self.assertTrue(c_file.is_file(), f"Missing {c_file.name}")

            df_p = pd.read_parquet(p_file)
            df_c = pd.read_csv(c_file)
            self.assertEqual(len(df_p), len(df_c))
            self.assertEqual(len(df_p.columns), len(df_c.columns))

    def test_feature_dictionary_coverage(self):
        """Verify feature dictionary documents at least 40 features with valid units and descriptions."""
        self.assertGreaterEqual(len(self.dict_df), 40)
        self.assertIn("Feature_Name", self.dict_df.columns)
        self.assertIn("Formula", self.dict_df.columns)
        self.assertIn("Unit", self.dict_df.columns)


if __name__ == "__main__":
    unittest.main()
