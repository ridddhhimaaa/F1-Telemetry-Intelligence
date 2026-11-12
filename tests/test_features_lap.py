"""
Tests for Lap-Level Features Module (Phase 6)
=============================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.lap_features import aggregate_lap_features


class TestLapFeatures(unittest.TestCase):
    """Unit tests verifying lap-level aggregation, summary metrics, and data quality."""

    def test_lap_level_aggregates_basic(self):
        """Verify lap summary calculates exact speed percentiles, WOT %, and counts."""
        df = pd.DataFrame({
            "Driver": ["VER"] * 10 + ["NOR"] * 10,
            "LapNumber": [4] * 10 + [4] * 10,
            "LapClassification": ["COMPLETE_FLYING_LAP"] * 20,
            "Distance_Corrected": list(np.linspace(0.0, 5000.0, 10)) * 2,
            "Time": [pd.Timedelta(seconds=i) for i in range(10)] * 2,
            "Speed": list(np.linspace(100.0, 300.0, 10)) * 2,
            "Throttle_Clipped": [100.0] * 5 + [0.0] * 5 + [80.0] * 10,
            "IsBraking": [0] * 5 + [1] * 5 + [0] * 10,
            "nGear": [7] * 10 + [6] * 10,
            "Gear_Change_Count": [5] * 10 + [8] * 10,
            "Upshift_Count": [3] * 10 + [4] * 10,
            "Downshift_Count": [2] * 10 + [4] * 10,
            "Throttle_Transition_Count": [2] * 10 + [3] * 10
        })

        lap_summary = aggregate_lap_features(df)

        # Must have exactly 2 rows (one for VER Lap 4, one for NOR Lap 4)
        self.assertEqual(len(lap_summary), 2)
        self.assertEqual(set(lap_summary["Driver"]), {"VER", "NOR"})

        ver_rec = lap_summary[lap_summary["Driver"] == "VER"].iloc[0]
        self.assertEqual(ver_rec["LapNumber"], 4)
        self.assertEqual(ver_rec["LapClassification"], "COMPLETE_FLYING_LAP")
        self.assertEqual(ver_rec["Speed_Mean_kmh"], 200.0)
        self.assertEqual(ver_rec["Speed_Max_kmh"], 300.0)
        self.assertEqual(ver_rec["Speed_Min_kmh"], 100.0)
        self.assertEqual(ver_rec["FullThrottle_Pct"], 50.0)
        self.assertEqual(ver_rec["Sample_Count"], 10)
        self.assertEqual(ver_rec["Most_Used_Gear"], 7)


if __name__ == "__main__":
    unittest.main()
