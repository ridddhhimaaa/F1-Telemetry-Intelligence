"""
Tests for Throttle Dynamics Feature Module (Phase 6)
====================================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.throttle_features import (
    compute_throttle_derivatives,
    compute_throttle_regime_durations,
    compute_throttle_features,
    WOT_THRESHOLD,
    OFF_THROTTLE_THRESHOLD
)


class TestThrottleFeatures(unittest.TestCase):
    """Unit tests verifying throttle derivative rates, durations, and transitions."""

    def test_throttle_derivatives_ramp(self):
        """Verify derivative on a linear 0 to 100% throttle ramp in 0.5s (200 %/s)."""
        t = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        th = pd.Series([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])

        dth, app_rate, rel_rate = compute_throttle_derivatives(th, t)
        for val in dth.iloc[1:]:
            self.assertAlmostEqual(val, 200.0, delta=1.0)
        for val in app_rate.iloc[1:]:
            self.assertAlmostEqual(val, 200.0, delta=1.0)
        for val in rel_rate.iloc[1:]:
            self.assertEqual(val, 0.0)

    def test_throttle_release_rate(self):
        """Verify negative derivative becomes positive release rate."""
        t = pd.Series([0.0, 0.1, 0.2])
        th = pd.Series([100.0, 50.0, 0.0])  # dropping by 50% in 0.1s -> -500 %/s

        dth, app_rate, rel_rate = compute_throttle_derivatives(th, t)
        self.assertAlmostEqual(dth.iloc[1], -500.0, delta=1.0)
        self.assertEqual(app_rate.iloc[1], 0.0)
        self.assertAlmostEqual(rel_rate.iloc[1], 500.0, delta=1.0)

    def test_wot_duration_and_transitions(self):
        """Verify WOT duration increases while throttle >= 98% and resets when lifted."""
        th = pd.Series([0.0, 100.0, 100.0, 100.0, 50.0, 100.0])
        t = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

        wot_dur, part_dur, off_dur, cum_pct, trans_cnt = compute_throttle_regime_durations(th, t)

        # WOT at t=0.5 (idx 1) is start -> 0.0; at t=1.0 -> 0.5; at t=1.5 -> 1.0; at t=2.0 lifted -> 0.0
        self.assertAlmostEqual(wot_dur.iloc[1], 0.0, delta=1e-3)
        self.assertAlmostEqual(wot_dur.iloc[2], 0.5, delta=1e-3)
        self.assertAlmostEqual(wot_dur.iloc[3], 1.0, delta=1e-3)
        self.assertAlmostEqual(wot_dur.iloc[4], 0.0, delta=1e-3)

        # Transitions: off -> WOT (+1), WOT -> part (+0), part -> WOT (+1) -> total 2
        self.assertEqual(trans_cnt.iloc[-1], 2)

    def test_full_throttle_pipeline(self):
        """Verify full feature DataFrame generation across laps."""
        df = pd.DataFrame({
            "Driver": ["HAM", "HAM", "HAM"],
            "LapNumber": [4, 4, 4],
            "Throttle": [10.0, 99.0, 100.0],
            "Time": [pd.Timedelta(seconds=1), pd.Timedelta(seconds=2), pd.Timedelta(seconds=3)]
        })

        res = compute_throttle_features(df)
        self.assertIn("Throttle_Application_Rate", res.columns)
        self.assertIn("FullThrottle_Duration_s", res.columns)
        self.assertIn("IsFullThrottle", res.columns)
        self.assertEqual(res["IsFullThrottle"].iloc[1], 1)
        self.assertEqual(res["IsFullThrottle"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
