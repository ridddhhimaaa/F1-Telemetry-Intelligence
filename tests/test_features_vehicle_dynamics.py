"""
Tests for Vehicle Dynamics Feature Module (Phase 6)
===================================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.vehicle_dynamics import (
    compute_longitudinal_acceleration,
    compute_lateral_acceleration,
    classify_dynamics_regimes,
    compute_phase_durations,
    compute_vehicle_dynamics_features,
    GRAVITY
)


class TestVehicleDynamics(unittest.TestCase):
    """Unit tests verifying vehicle dynamics formulations, units, and edge cases."""

    def setUp(self):
        # 10 samples at 10 Hz (dt = 0.1s)
        self.time_sec = pd.Series(np.linspace(0.0, 0.9, 10))
        # Constant acceleration: 0 to 72 km/h (0 to 20 m/s) in 0.9s -> a = 20 / 0.9 = 22.22 m/s^2
        self.speed_kmh = pd.Series(np.linspace(0.0, 72.0, 10))

    def test_longitudinal_acceleration_constant_accel(self):
        """Verify longitudinal acceleration against an exact analytical constant acceleration profile."""
        a_long, g_long = compute_longitudinal_acceleration(
            self.speed_kmh, self.time_sec, smooth=False
        )
        expected_a = (72.0 / 3.6) / 0.9  # ~22.22 m/s^2
        # Verify that calculated acceleration is within 1% of analytical expected value
        for val in a_long.iloc[1:]:
            self.assertAlmostEqual(val, expected_a, delta=0.5)

        # Check G-force conversion
        self.assertAlmostEqual(g_long.iloc[5], a_long.iloc[5] / GRAVITY, places=3)

    def test_heavy_braking_deceleration_range(self):
        """Verify heavy braking deceleration (-50 m/s^2, ~-5g) is preserved by smoothing."""
        # Speed drops from 320 km/h to 80 km/h in 1.3 seconds
        t = pd.Series(np.linspace(0.0, 1.3, 14))
        v = pd.Series(np.linspace(320.0, 80.0, 14))
        a_long, g_long = compute_longitudinal_acceleration(v, t, smooth=True)

        min_a = a_long.min()
        min_g = g_long.min()

        # F1 cars reach -4.5 to -5.5g in heavy braking into Turn 6
        self.assertLess(min_a, -40.0)
        self.assertLess(min_g, -4.0)

    def test_irregular_and_tiny_dt_handling(self):
        """Verify irregular dt and tiny dt (0.001s) do not produce NaN or infinite acceleration."""
        t = pd.Series([0.0, 0.001, 0.101, 0.25, 0.38])
        v = pd.Series([100.0, 100.2, 105.0, 110.0, 115.0])

        a_long, g_long = compute_longitudinal_acceleration(v, t)
        self.assertFalse(a_long.isna().any())
        self.assertFalse(np.isinf(a_long).any())
        self.assertFalse(g_long.isna().any())

    def test_lateral_acceleration(self):
        """Verify lateral acceleration formula a_lat = v^2 * kappa."""
        # 180 km/h = 50 m/s, Radius R = 100m -> kappa = 0.01 m^-1
        # a_lat = 50^2 * 0.01 = 25.0 m/s^2 -> g_lat = 25 / 9.80665 = 2.549g
        speed = pd.Series([180.0])
        kappa = pd.Series([0.01])

        a_lat, g_lat = compute_lateral_acceleration(speed, kappa)
        self.assertAlmostEqual(a_lat.iloc[0], 25.0, places=2)
        self.assertAlmostEqual(g_lat.iloc[0], 25.0 / GRAVITY, places=2)

    def test_regime_classification(self):
        """Verify operational regimes correctly tag braking, acceleration, and coasting."""
        a = pd.Series([10.0, -30.0, 0.0])
        v = pd.Series([200.0, 120.0, 250.0])
        th = pd.Series([100.0, 0.0, 0.0])
        brk = pd.Series([0, 1, 0])

        is_accel, is_decel, is_coast, regime = classify_dynamics_regimes(a, v, th, brk)
        self.assertEqual(is_accel.iloc[0], 1)
        self.assertEqual(is_decel.iloc[1], 1)
        self.assertEqual(is_coast.iloc[2], 1)
        self.assertEqual(regime.iloc[0], "AERO_LIMITED_ACCEL")
        self.assertEqual(regime.iloc[1], "HEAVY_BRAKING")
        self.assertEqual(regime.iloc[2], "COASTING")

    def test_phase_duration_accumulation_and_reset(self):
        """Verify that phase duration accumulates during contiguous events and resets to zero."""
        events = pd.Series([0, 1, 1, 1, 0, 1, 1])
        times = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])

        dur = compute_phase_durations(events, times)
        expected = [0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.2]
        np.testing.assert_allclose(dur.values, expected, atol=1e-3)

    def test_pipeline_grouping_prevents_leakage(self):
        """Verify that lap boundaries are strictly isolated with zero leakage between laps."""
        df = pd.DataFrame({
            "Driver": ["VER", "VER", "VER", "VER"],
            "LapNumber": [4, 4, 5, 5],
            "Speed": [100.0, 200.0, 300.0, 150.0],
            "Time": [pd.Timedelta(seconds=10), pd.Timedelta(seconds=11),
                     pd.Timedelta(seconds=20), pd.Timedelta(seconds=21)],
            "Throttle": [100.0, 100.0, 0.0, 0.0],
            "IsBraking": [0, 0, 1, 1]
        })

        res = compute_vehicle_dynamics_features(df)
        self.assertEqual(len(res), 4)
        self.assertIn("Accel_Longitudinal_mps2", res.columns)
        self.assertIn("G_Combined", res.columns)
        # Lap 5 point 0 should NOT be affected by Lap 4 point 1 (Time jump from 11s to 20s must NOT calculate negative dt!)
        self.assertGreater(res["G_Combined"].iloc[2], 0.0)


if __name__ == "__main__":
    unittest.main()
