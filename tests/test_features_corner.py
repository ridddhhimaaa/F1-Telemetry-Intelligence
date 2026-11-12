"""
Tests for Corner Performance Feature Module (Phase 6)
=====================================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.corner_features import (
    compute_corner_kinematics,
    compute_corner_features
)


class TestCornerFeatures(unittest.TestCase):
    """Unit tests verifying corner entry, apex, exit speed, and summary table generation."""

    def test_corner_kinematics_synthetic_corner(self):
        """Verify speed profile through a synthetic corner: entry 250, apex 100, exit 200 km/h."""
        # 10 samples: 2 straight, 6 corner (apex at idx 4), 2 straight
        cid = pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
        apx = pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        spd = pd.Series([280.0, 270.0, 250.0, 180.0, 100.0, 130.0, 170.0, 200.0, 240.0, 260.0])
        dist = pd.Series([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0])
        t = pd.Series([0.0, 0.2, 0.4, 0.7, 1.1, 1.5, 1.9, 2.3, 2.6, 2.8])

        s_df, sum_df = compute_corner_kinematics(cid, apx, spd, dist, t)

        # In straightaway (idx 0), corner metrics must be zero
        self.assertEqual(s_df["Corner_Entry_Speed_kmh"].iloc[0], 0.0)

        # In corner (idx 2..7): entry is 250.0, apex is 100.0, exit is 200.0
        self.assertEqual(s_df["Corner_Entry_Speed_kmh"].iloc[2], 250.0)
        self.assertEqual(s_df["Corner_Apex_Speed_kmh"].iloc[4], 100.0)
        self.assertEqual(s_df["Corner_Exit_Speed_kmh"].iloc[7], 200.0)

        # Speed loss: 250 - 100 = 150 km/h
        self.assertEqual(s_df["Corner_Speed_Loss_kmh"].iloc[3], 150.0)

        # Exit accel: 200 - 100 = 100 km/h
        self.assertEqual(s_df["Corner_Exit_Accel_kmh"].iloc[5], 100.0)

        # Traversal duration: t[7] - t[2] = 2.3 - 0.4 = 1.9s
        self.assertAlmostEqual(s_df["Corner_Traversal_Duration_s"].iloc[2], 1.9, delta=1e-3)

        # Summary table should have exactly 1 corner record
        self.assertEqual(len(sum_df), 1)
        rec = sum_df.iloc[0]
        self.assertEqual(rec["Corner_ID"], 1)
        self.assertEqual(rec["EntrySpeed_kmh"], 250.0)
        self.assertEqual(rec["ApexSpeed_kmh"], 100.0)
        self.assertEqual(rec["ExitSpeed_kmh"], 200.0)
        self.assertEqual(rec["SpeedLoss_kmh"], 150.0)
        self.assertEqual(rec["ExitAccel_kmh"], 100.0)
        self.assertAlmostEqual(rec["CornerDuration_s"], 1.9, delta=1e-3)

    def test_pre_and_post_apex_zones(self):
        """Verify pre-apex is 1 before apex, 0 at/after apex; post-apex is 1 after apex."""
        cid = pd.Series([1, 1, 1, 1, 1])
        apx = pd.Series([0, 0, 1, 0, 0])  # apex at idx 2
        spd = pd.Series([200.0, 150.0, 100.0, 130.0, 160.0])
        dist = pd.Series([0.0, 20.0, 40.0, 60.0, 80.0])
        t = pd.Series([0.0, 0.3, 0.6, 0.9, 1.2])

        s_df, _ = compute_corner_kinematics(cid, apx, spd, dist, t)
        self.assertEqual(s_df["IsPreApexZone"].iloc[0], 1)
        self.assertEqual(s_df["IsPreApexZone"].iloc[1], 1)
        self.assertEqual(s_df["IsPreApexZone"].iloc[2], 0)  # at apex
        self.assertEqual(s_df["IsPostApexZone"].iloc[2], 0)  # at apex
        self.assertEqual(s_df["IsPostApexZone"].iloc[3], 1)
        self.assertEqual(s_df["IsPostApexZone"].iloc[4], 1)


if __name__ == "__main__":
    unittest.main()
