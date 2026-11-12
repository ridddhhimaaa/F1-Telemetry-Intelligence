"""
Tests for Sector Features Module (Phase 6)
==========================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.sector_features import (
    assign_sector_markers,
    compute_sector_aggregates,
    compute_sector_features,
    SECTOR_BOUNDARIES
)


class TestSectorFeatures(unittest.TestCase):
    """Unit tests verifying timing sector segmentation and sector aggregate metrics."""

    def test_sector_boundary_assignment(self):
        """Verify distance milestones map to Sectors 1, 2, and 3 accurately."""
        distances = pd.Series([0.0, 600.0, 1200.0, 1201.0, 2500.0, 3520.0, 3521.0, 5000.0])
        sec_id, dist_in_sec = assign_sector_markers(distances)

        self.assertEqual(sec_id.iloc[0], 1)
        self.assertEqual(sec_id.iloc[1], 1)
        self.assertEqual(sec_id.iloc[2], 1)
        self.assertEqual(sec_id.iloc[3], 2)
        self.assertEqual(sec_id.iloc[4], 2)
        self.assertEqual(sec_id.iloc[5], 2)
        self.assertEqual(sec_id.iloc[6], 3)
        self.assertEqual(sec_id.iloc[7], 3)

        # Distance in sector checks
        self.assertEqual(dist_in_sec.iloc[1], 600.0)
        self.assertEqual(dist_in_sec.iloc[3], 1.0)
        self.assertEqual(dist_in_sec.iloc[6], 1.0)

    def test_sector_aggregates_computation(self):
        """Verify sector aggregate metrics (speeds, WOT %, gear usage)."""
        df = pd.DataFrame({
            "Driver": ["NOR", "NOR", "NOR", "NOR"],
            "LapNumber": [4, 4, 4, 4],
            "Sector": [1, 1, 2, 2],
            "Distance": [100.0, 500.0, 1500.0, 2000.0],
            "_time_sec": [10.0, 15.0, 30.0, 40.0],
            "Speed": [150.0, 250.0, 280.0, 320.0],
            "Throttle": [50.0, 100.0, 100.0, 100.0],
            "Brake": [0, 0, 0, 0],
            "nGear": [4, 6, 7, 8],
            "IsGearShift": [0, 1, 0, 1],
            "Corner_ID": [1, 0, 0, 2]
        })

        sum_df = compute_sector_aggregates(df)
        self.assertEqual(len(sum_df), 2)  # Sector 1 and Sector 2

        s1 = sum_df[sum_df["Sector"] == 1].iloc[0]
        self.assertEqual(s1["Speed_Mean_kmh"], 200.0)
        self.assertEqual(s1["Speed_Max_kmh"], 250.0)
        self.assertEqual(s1["Speed_Min_kmh"], 150.0)
        self.assertEqual(s1["FullThrottle_Pct"], 50.0)

        s2 = sum_df[sum_df["Sector"] == 2].iloc[0]
        self.assertEqual(s2["Speed_Mean_kmh"], 300.0)
        self.assertEqual(s2["FullThrottle_Pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
