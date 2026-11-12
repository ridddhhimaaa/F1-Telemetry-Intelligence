"""
Tests for Gear and RPM Feature Module (Phase 6)
==============================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.gear_rpm_features import (
    detect_shift_events,
    compute_rpm_derivatives,
    compute_gear_rpm_features
)


class TestGearRPMFeatures(unittest.TestCase):
    """Unit tests verifying transmission shift detection, engine acceleration, and drivetrain ratios."""

    def test_shift_detection_upshifts_and_downshifts(self):
        """Verify gear changes classify into upshifts and downshifts with monotonic counts."""
        gears = pd.Series([2, 2, 3, 4, 4, 3, 2])
        times = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

        is_shift, is_up, is_down, cum_s, cum_up, cum_down = detect_shift_events(gears, times)

        # Gear 2 -> 3 at idx 2 (upshift)
        self.assertEqual(is_shift.iloc[2], 1)
        self.assertEqual(is_up.iloc[2], 1)
        self.assertEqual(is_down.iloc[2], 0)

        # Gear 4 -> 3 at idx 5 (downshift)
        self.assertEqual(is_shift.iloc[5], 1)
        self.assertEqual(is_up.iloc[5], 0)
        self.assertEqual(is_down.iloc[5], 1)

        # Total counts at end: 2 upshifts, 2 downshifts -> 4 total shifts
        self.assertEqual(cum_up.iloc[-1], 2)
        self.assertEqual(cum_down.iloc[-1], 2)
        self.assertEqual(cum_s.iloc[-1], 4)

    def test_rpm_derivative(self):
        """Verify engine acceleration derivative d(RPM)/dt."""
        rpm = pd.Series([10000.0, 10500.0, 11000.0])
        t = pd.Series([0.0, 0.1, 0.2])  # 500 RPM / 0.1s = 5000 RPM/s

        drpm = compute_rpm_derivatives(rpm, t)
        self.assertAlmostEqual(drpm.iloc[1], 5000.0, delta=10.0)
        self.assertAlmostEqual(drpm.iloc[2], 5000.0, delta=10.0)

    def test_zero_speed_clamping(self):
        """Verify zero speed does not cause division by zero in RPM/speed ratio."""
        df = pd.DataFrame({
            "Driver": ["ALO"],
            "LapNumber": [4],
            "nGear": [1],
            "RPM": [4000.0],
            "Speed": [0.0],
            "Time": [pd.Timedelta(seconds=1)]
        })

        res = compute_gear_rpm_features(df)
        self.assertFalse(np.isinf(res["Ratio_RPM_per_kmh"].iloc[0]))
        self.assertFalse(np.isnan(res["Ratio_RPM_per_kmh"].iloc[0]))
        # 4000 / 5.0 (clamped) = 800.0
        self.assertEqual(res["Ratio_RPM_per_kmh"].iloc[0], 800.0)


if __name__ == "__main__":
    unittest.main()
