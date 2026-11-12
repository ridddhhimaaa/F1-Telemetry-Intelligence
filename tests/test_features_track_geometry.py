"""
Tests for Track Geometry Feature Module (Phase 6)
=================================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.track_geometry import (
    compute_heading_and_curvature,
    detect_corners_from_curvature,
    compute_track_geometry_features,
    COORD_SCALE_FACTOR
)


class TestTrackGeometry(unittest.TestCase):
    """Unit tests verifying track geometry, curvature, and corner detection."""

    def test_coordinate_scaling(self):
        """Verify FastF1 coordinates in decimeters are scaled to meters by 0.1 factor."""
        self.assertEqual(COORD_SCALE_FACTOR, 0.1)
        raw_x = pd.Series([1000.0, 2000.0])
        raw_y = pd.Series([500.0, 1500.0])
        df = pd.DataFrame({"X": raw_x, "Y": raw_y, "Distance": [0.0, 100.0],
                           "Driver": ["NOR", "NOR"], "LapNumber": [4, 4]})
        res = compute_track_geometry_features(df)
        self.assertEqual(res["X_m"].iloc[0], 100.0)
        self.assertEqual(res["Y_m"].iloc[0], 50.0)

    def test_straight_line_geometry(self):
        """Verify straight line yields zero curvature, infinite radius, and STRAIGHT direction."""
        # 50 points along straight line parallel to X axis: X increases from 0 to 500m (0 to 5000 dm)
        x = pd.Series(np.linspace(0.0, 5000.0, 50))
        y = pd.Series(np.full(50, 1000.0))  # constant Y

        heading, dtheta, kappa, radius = compute_heading_and_curvature(x, y, smooth_coords=False)

        # In straight line along +X, heading is 0 deg
        for h in heading.iloc[2:-2]:
            self.assertAlmostEqual(h, 0.0, delta=1.0)
        for k in kappa.iloc[2:-2]:
            self.assertAlmostEqual(k, 0.0, delta=1e-4)
        for r in radius.iloc[2:-2]:
            self.assertGreaterEqual(r, 9000.0)

    def test_circular_arc_known_radius(self):
        """Verify circular arc of radius R = 50m yields curvature kappa ~ 0.02 m^-1 and radius ~ 50m."""
        # Circle of radius R = 50m (500 dm). Arc from theta = 0 to pi/2 (90 deg left turn)
        angles = np.linspace(0.0, np.pi / 2, 60)
        r_dm = 500.0  # 50 meters
        x_dm = pd.Series(r_dm * np.cos(angles))
        y_dm = pd.Series(r_dm * np.sin(angles))

        heading, dtheta, kappa, radius = compute_heading_and_curvature(x_dm, y_dm, smooth_coords=False)

        # Theoretical curvature kappa = 1 / 50m = 0.02 m^-1
        mean_k = np.mean(np.abs(kappa.iloc[5:-5]))
        mean_r = np.mean(radius.iloc[5:-5])

        self.assertAlmostEqual(mean_k, 0.02, delta=0.005)
        self.assertAlmostEqual(mean_r, 50.0, delta=10.0)

    def test_corner_detection_and_apex(self):
        """Verify corner detector flags corner interval, apex at maximum curvature, and turn direction."""
        # 100m straight, 100m corner with peak curvature at center, 100m straight
        dist = pd.Series(np.linspace(0.0, 300.0, 60))
        # Curvature Gaussian profile peaked at 150m with kappa_max = 0.03
        k_profile = 0.03 * np.exp(-((dist.values - 150.0) / 25.0) ** 2)
        kappa = pd.Series(k_profile)

        corner_id, is_apex, turn_dir = detect_corners_from_curvature(kappa, dist, threshold=0.005)

        # Should identify 1 corner
        self.assertGreater(corner_id.max(), 0)
        # Apex must be near distance 150m
        apex_idx = np.where(is_apex.values == 1)[0]
        self.assertEqual(len(apex_idx), 1)
        self.assertAlmostEqual(dist.iloc[apex_idx[0]], 150.0, delta=10.0)
        # Turn direction should be LEFT (positive curvature)
        self.assertEqual(turn_dir.iloc[apex_idx[0]], "LEFT")


if __name__ == "__main__":
    unittest.main()
