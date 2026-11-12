"""
Tests for Braking Dynamics Feature Module (Phase 6)
===================================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.braking_features import (
    segment_braking_events,
    compute_pedal_transition_and_overlap,
    compute_braking_features,
    GRAVITY
)


class TestBrakingFeatures(unittest.TestCase):
    """Unit tests verifying braking event segmentation, distance, peak G, and pedal overlap."""

    def test_braking_event_segmentation_and_distance(self):
        """Verify braking events segment into IDs with accurate duration and distance."""
        # 10 samples: 2 off-brake, 4 on-brake, 4 off-brake
        is_brk = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        dist = pd.Series([0.0, 10.0, 20.0, 50.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0])
        t = pd.Series([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
        # Deceleration during braking: -10, -40, -30, -10 m/s^2 (peak at idx 3 = -40 m/s^2 = ~4.08g)
        accel = pd.Series([0.0, 0.0, -10.0, -40.0, -30.0, -10.0, 5.0, 5.0, 5.0, 5.0])

        evt_id, is_onset, dur_s, dist_m, peak_g, is_rel = segment_braking_events(is_brk, dist, t, accel)

        # Event ID should be 1 during indices 2..5
        self.assertEqual(evt_id.iloc[0], 0)
        self.assertEqual(evt_id.iloc[2], 1)
        self.assertEqual(evt_id.iloc[5], 1)
        self.assertEqual(evt_id.iloc[6], 0)

        # Onset should be 1 strictly at idx 2
        self.assertEqual(is_onset.iloc[2], 1)
        self.assertEqual(is_onset.iloc[3], 0)

        # Braking distance at end of event: 100m - 20m = 80m
        self.assertAlmostEqual(dist_m.iloc[5], 80.0, delta=1e-2)

        # Braking duration at end of event: 0.8s - 0.2s = 0.6s
        self.assertAlmostEqual(dur_s.iloc[5], 0.6, delta=1e-3)

        # Peak G should be 40.0 / 9.80665 = 4.08g
        self.assertAlmostEqual(peak_g.iloc[3], 40.0 / GRAVITY, places=2)

        # Release phase should be 1 after peak (idx 4 and 5)
        self.assertEqual(is_rel.iloc[2], 0)
        self.assertEqual(is_rel.iloc[3], 0)  # peak itself
        self.assertEqual(is_rel.iloc[4], 1)  # after peak
        self.assertEqual(is_rel.iloc[5], 1)

    def test_multiple_braking_events_sequential_ids(self):
        """Verify consecutive discrete braking events get incremented IDs."""
        is_brk = pd.Series([1, 1, 0, 0, 1, 1, 0])
        dist = pd.Series([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
        t = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        accel = pd.Series([-20.0, -10.0, 0.0, 0.0, -30.0, -15.0, 0.0])

        evt_id, is_onset, dur_s, dist_m, peak_g, is_rel = segment_braking_events(is_brk, dist, t, accel)
        self.assertEqual(evt_id.iloc[0], 1)
        self.assertEqual(evt_id.iloc[2], 0)
        self.assertEqual(evt_id.iloc[4], 2)
        self.assertEqual(is_onset.iloc[0], 1)
        self.assertEqual(is_onset.iloc[4], 1)

    def test_pedal_overlap_and_release_time(self):
        """Verify detection of simultaneous throttle and brake overlap."""
        is_brk = pd.Series([1, 1, 0, 0])
        th = pd.Series([0.0, 30.0, 50.0, 100.0])  # overlap at idx 1
        t = pd.Series([0.0, 0.1, 0.2, 0.5])

        is_ovlp, t_rel = compute_pedal_transition_and_overlap(is_brk, th, t)
        self.assertEqual(is_ovlp.iloc[0], 0)
        self.assertEqual(is_ovlp.iloc[1], 1)  # overlap
        self.assertEqual(is_ovlp.iloc[2], 0)

        # Time since brake release at idx 3: 0.5 - 0.2 = 0.3s
        self.assertAlmostEqual(t_rel.iloc[3], 0.3, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
