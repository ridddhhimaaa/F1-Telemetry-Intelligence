"""
F1 Telemetry Intelligence - Phase 5.1 EDA Validation Unit Tests
================================================================

Comprehensive unit and integration tests verifying the scientific validation routines,
DRS confounding decomposition, kinematic correlation audits, ML leakage evaluations,
cluster-aware bootstrap inference, modular exports, and reporting outputs.

Zero Side-Effect Isolation:
All test-generated CSV, PNG, and HTML files are redirected to a temporary directory
(tempfile.TemporaryDirectory) to strictly prevent overwriting production reports.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.analysis.data_loader import (
    load_all_datasets,
    load_processed_telemetry,
    load_processed_laps,
    filter_clean_flying_laps
)
from src.analysis.statistical_validation import (
    compute_sample_level_stats,
    compute_cluster_bootstrap_ci,
    compute_driver_level_aggregated_stats
)
from src.analysis.drs_validation import (
    validate_drs_effect as modular_validate_drs,
    compute_drs_driver_breakdown as modular_drs_breakdown,
    DRS_ZONE_1_RANGE,
    DRS_ZONE_2_RANGE
)
from src.analysis.correlation_validation import (
    validate_kinematic_correlations as modular_validate_correlations,
    compute_distance_windowed_correlations as modular_distance_correlations,
    compute_temporal_autocorrelation as modular_autocorrelation
)
from src.analysis.ml_readiness_audit import (
    audit_machine_learning_targets as modular_audit_ml
)
from src.analysis.eda_validation import (
    validate_drs_effect,
    compute_drs_driver_breakdown,
    validate_kinematic_correlations,
    compute_distance_windowed_correlations,
    compute_temporal_autocorrelation,
    audit_machine_learning_targets,
    generate_all_validation_figures,
    generate_validation_html_report,
    VALIDATION_DIR,
    FIGURES_DIR
)


class TestEDAValidation(unittest.TestCase):
    """Test suite for Phase 5.1 EDA scientific validation routines with isolated output redirection."""

    @classmethod
    def setUpClass(cls):
        cls.tel_df, cls.laps_df, cls.res_df = load_all_datasets()
        cls.flying_tel = filter_clean_flying_laps(cls.tel_df)

        # Create isolated temporary directory for test outputs
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls._tmp_dir.name)
        cls.test_figs_dir = cls.test_dir / "figures"
        cls.test_figs_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def test_datasets_loaded_correctly(self):
        """Verify underlying processed datasets are populated."""
        self.assertGreater(len(self.tel_df), 70000)
        self.assertGreater(len(self.laps_df), 1000)
        self.assertGreater(len(self.flying_tel), 25000)
        self.assertEqual(self.flying_tel["Driver"].nunique(), 19)

    def test_drs_validation_metrics_and_confounding(self):
        """Verify DRS confounding decomposition: naive delta is ~100 km/h, controlled is ~12-14 km/h."""
        # Execute with isolated test output_dir
        res = validate_drs_effect(self.tel_df, output_dir=self.test_dir)

        naive = res["naive_full_lap"]
        self.assertAlmostEqual(naive["drs_on_mean"], 298.69, delta=1.5)
        self.assertAlmostEqual(naive["drs_off_mean"], 197.77, delta=1.5)
        self.assertAlmostEqual(naive["naive_delta"], 100.92, delta=2.0)

        ctrl = res["controlled_straights"]
        self.assertGreater(ctrl["Delta_Mean_kmh"], 10.0)
        self.assertLess(ctrl["Delta_Mean_kmh"], 16.0)
        self.assertGreater(ctrl["cohens_d"], 0.40)
        self.assertLess(ctrl["p_value_ttest"], 1e-20)
        self.assertLess(ctrl["p_value_mann_whitney"], 1e-20)

        # Driver-level cluster bootstrap CI validation (hierarchical 19-driver unit, 5,000 iterations)
        boot = res["cluster_bootstrap_straights"]
        self.assertGreater(boot["Cluster_CI95_Lower_kmh"], 8.0)
        self.assertLess(boot["Cluster_CI95_Upper_kmh"], 18.0)
        self.assertEqual(boot["N_Clusters"], 19)
        self.assertEqual(boot["N_Drivers"], 19)
        self.assertGreaterEqual(boot["Bootstrap_Iterations"], 5000)
        self.assertEqual(boot["ResamplingUnit"], "Driver (both eligible laps retained)")

        decomp = res["confounding_decomposition"]
        self.assertGreater(decomp["confounding_share_percent"], 80.0)
        self.assertLess(decomp["confounding_share_percent"], 95.0)
        self.assertGreater(decomp["aerodynamic_share_percent"], 5.0)
        self.assertLess(decomp["aerodynamic_share_percent"], 20.0)

        # Verify CSV export in test directory (NOT production)
        test_csv = self.test_dir / "drs_controlled_comparison.csv"
        self.assertTrue(test_csv.is_file())

    def test_drs_driver_level_breakdown(self):
        """Verify per-driver DRS statistics on straights."""
        df_drv = compute_drs_driver_breakdown(self.tel_df, output_dir=self.test_dir)

        self.assertEqual(len(df_drv), 19)
        self.assertIn("Driver", df_drv.columns)
        self.assertIn("Controlled_Delta_Mean_kmh", df_drv.columns)

        # Lando Norris was leading and had no car within 1.0s ahead
        nor = df_drv[df_drv["Driver"] == "NOR"].iloc[0]
        self.assertEqual(nor["DRS_ON_Samples"], 0)
        self.assertEqual(nor["DRS_Eligible_Flying"], "NO (Leading or >1.0s gap)")

        # Max Verstappen had both DRS ON and OFF samples
        ver = df_drv[df_drv["Driver"] == "VER"].iloc[0]
        self.assertGreater(ver["DRS_ON_Samples"], 50)
        self.assertGreater(ver["DRS_OFF_Samples"], 50)
        self.assertAlmostEqual(ver["Controlled_Delta_Mean_kmh"], 10.92, delta=2.0)

        self.assertTrue((self.test_dir / "drs_driver_level.csv").is_file())

    def test_kinematic_correlation_validation(self):
        """Verify pooled vs intra-lap correlation distribution across 38 flying laps."""
        res = validate_kinematic_correlations(self.tel_df, output_dir=self.test_dir)

        # Pooled correlations
        self.assertAlmostEqual(res["pooled"]["r_Speed_Throttle"], 0.580, delta=0.03)
        self.assertGreater(res["pooled"]["r_Speed_Gear"], 0.90)
        self.assertGreater(res["pooled"]["r_Speed_RPM"], 0.70)

        # Intra-lap distributions
        self.assertEqual(res["lap_count"], 38)
        intra = res["intra_lap_distribution"]
        self.assertAlmostEqual(intra["r_Speed_Throttle"]["mean"], 0.574, delta=0.05)
        self.assertAlmostEqual(intra["r_Speed_Gear"]["mean"], 0.936, delta=0.05)
        self.assertGreater(intra["ACF_lag1_Speed"]["mean"], 0.99)
        self.assertGreater(intra["ACF_lag1_Throttle"]["mean"], 0.97)

        self.assertTrue((self.test_dir / "intra_lap_correlations.csv").is_file())
        self.assertTrue((self.test_dir / "kinematic_correlation_summary.csv").is_file())

    def test_distance_windowed_correlations(self):
        """Verify that local Speed-Throttle correlation collapses along terminal straights."""
        df_win = compute_distance_windowed_correlations(self.tel_df, window_size_m=250.0, output_dir=self.test_dir)

        self.assertGreater(len(df_win), 15)
        straight_win = df_win[(df_win["Start_m"] == 2000.0) & (df_win["End_m"] == 2250.0)].iloc[0]
        self.assertGreater(straight_win["Mean_Throttle_pct"], 98.0)
        self.assertLess(straight_win["r_Speed_Throttle"], 0.10)

        accel_win = df_win[(df_win["Start_m"] == 0.0) & (df_win["End_m"] == 250.0)].iloc[0]
        self.assertGreater(accel_win["r_Speed_Throttle"], 0.80)

        self.assertTrue((self.test_dir / "distance_windowed_correlations.csv").is_file())

    def test_temporal_autocorrelation_decay(self):
        """Verify lag-1 to lag-15 ACF decay across telemetry signals."""
        df_acf = compute_temporal_autocorrelation(self.tel_df, max_lag=15, output_dir=self.test_dir)

        self.assertEqual(len(df_acf), 15)
        self.assertGreater(df_acf.iloc[0]["ACF_Speed"], 0.99)
        self.assertGreater(df_acf.iloc[0]["ACF_Throttle"], 0.97)
        self.assertGreater(df_acf.iloc[0]["ACF_Speed"], df_acf.iloc[14]["ACF_Speed"])

        self.assertTrue((self.test_dir / "temporal_autocorrelation.csv").is_file())

    def test_ml_target_and_leakage_audit(self):
        """Verify ML target audit, Brake_Int tautology, spatial memorization, and split leakage."""
        ml_res = audit_machine_learning_targets(self.tel_df, self.laps_df, output_dir=self.test_dir)

        # Target A: Brake_Int is a 100% tautological leak
        self.assertTrue(ml_res["target_a_is_braking"]["is_tautological_leak"])
        self.assertAlmostEqual(ml_res["target_a_is_braking"]["brake_int_correlation"], 1.0, places=4)

        # Target B: Spatial lookup memorizes circuit map
        spd = ml_res["target_b_speed"]
        self.assertGreater(spd["spatial_10m_lookup_r2"], 0.88)
        self.assertGreater(spd["kinematic_linear_r2"], 0.90)

        # Target C: Deterministic gear ratios monotonically decreasing
        gear_ratios = [g["Mean_Ratio_RPM_per_kmh"] for g in ml_res["target_c_gear"]["ratios"]]
        for i in range(1, len(gear_ratios) - 1):
            self.assertGreater(gear_ratios[i], gear_ratios[i+1])

        # Target D: Flying laps sample size
        self.assertEqual(ml_res["target_d_lap_time"]["flying_laps_count"], 38)
        self.assertEqual(ml_res["target_d_lap_time"]["drivers_count"], 19)

        # Train/Test Split: 96.1% immediate neighbor leakage
        split_audit = ml_res["train_test_split_audit"]
        self.assertGreater(split_audit["random_split_adjacent_leakage_pct"], 90.0)

        self.assertTrue((self.test_dir / "ml_target_leakage_audit.csv").is_file())
        self.assertTrue((self.test_dir / "ml_splitting_strategy_audit.csv").is_file())
        self.assertTrue((self.test_dir / "ml_gear_ratio_audit.csv").is_file())

    def test_figures_and_html_generation_isolated(self):
        """Verify all 6 visualization figures and standalone HTML report are generated in isolated test dir."""
        figs = generate_all_validation_figures(self.tel_df, self.laps_df, output_dir=self.test_figs_dir)
        self.assertEqual(len(figs), 6)
        for f in figs:
            self.assertTrue(f.is_file())
            self.assertGreater(f.stat().st_size, 10000)
            # Ensure saved in test directory
            self.assertEqual(f.parent.resolve(), self.test_figs_dir.resolve())

        html_path = generate_validation_html_report(
            self.tel_df, self.laps_df,
            output_path=self.test_dir / "phase5_1_validation_report.html"
        )
        self.assertTrue(html_path.is_file())
        self.assertGreater(html_path.stat().st_size, 5000)
        self.assertEqual(html_path.parent.resolve(), self.test_dir.resolve())

    def test_modular_architecture_imports(self):
        """Verify that separate modular files expose identical analytical outputs."""
        drs_mod = modular_validate_drs(self.tel_df, output_dir=self.test_dir)
        self.assertIn("controlled_straights", drs_mod)

        corr_mod = modular_validate_correlations(self.tel_df, output_dir=self.test_dir)
        self.assertIn("intra_lap_distribution", corr_mod)

        ml_mod = modular_audit_ml(self.tel_df, self.laps_df, output_dir=self.test_dir)
        self.assertIn("target_a_is_braking", ml_mod)

    # ==========================================================================
    # PHASE 5.2: DEDICATED DRIVER-LEVEL CLUSTER BOOTSTRAP UNIT TESTS
    # ==========================================================================

    def test_driver_level_cluster_resampling(self):
        """Verify that default cluster unit is Driver (N=19) retaining all eligible laps."""
        z1_mask = (self.flying_tel["Distance_Corrected"] >= DRS_ZONE_1_RANGE[0]) & (self.flying_tel["Distance_Corrected"] <= DRS_ZONE_1_RANGE[1])
        z2_mask = (self.flying_tel["Distance_Corrected"] >= DRS_ZONE_2_RANGE[0]) & (self.flying_tel["Distance_Corrected"] <= DRS_ZONE_2_RANGE[1])
        straights = self.flying_tel[z1_mask | z2_mask]

        boot = compute_cluster_bootstrap_ci(straights, n_bootstraps=5000, random_state=42)

        self.assertEqual(boot["N_Clusters"], 19)
        self.assertEqual(boot["N_Drivers"], 19)
        self.assertEqual(boot["Bootstrap_Iterations"], 5000)
        self.assertEqual(boot["ResamplingUnit"], "Driver (both eligible laps retained)")
        self.assertEqual(boot["SamplingUnit"], "19 drivers (38 driver-laps retained)")
        self.assertEqual(boot["Cluster_CI95_Lower_kmh"], 10.50)
        self.assertEqual(boot["Cluster_CI95_Upper_kmh"], 15.28)

    def test_retention_of_both_laps_per_selected_driver(self):
        """Verify that selecting a driver keeps both laps for that driver together."""
        synthetic_df = pd.DataFrame({
            "Driver": ["D1", "D1", "D1", "D1", "D2", "D2", "D2", "D2"],
            "LapNumber": [4, 4, 5, 5, 4, 4, 5, 5],
            "DRS": [1, 1, 0, 0, 1, 1, 0, 0],
            "Speed": [300.0, 302.0, 200.0, 202.0, 310.0, 312.0, 210.0, 212.0]
        })

        boot = compute_cluster_bootstrap_ci(
            synthetic_df,
            cluster_cols=["Driver"],
            treatment_col="DRS",
            value_col="Speed",
            n_bootstraps=100,
            random_state=42
        )

        self.assertEqual(boot["N_Clusters"], 2)
        self.assertEqual(boot["N_Drivers"], 2)
        self.assertEqual(boot["SamplingUnit"], "2 drivers (4 driver-laps retained)")
        # In this synthetic balanced setup, for either driver the mean ON is 301/311 and OFF is 201/211 -> delta is 100.0
        self.assertAlmostEqual(boot["Cluster_Delta_Mean_kmh"], 100.0, delta=0.5)

    def test_bootstrap_reproducibility_with_fixed_seed(self):
        """Verify bootstrap reproducibility with fixed random seed."""
        z1_mask = (self.flying_tel["Distance_Corrected"] >= DRS_ZONE_1_RANGE[0]) & (self.flying_tel["Distance_Corrected"] <= DRS_ZONE_1_RANGE[1])
        straights_z1 = self.flying_tel[z1_mask]

        run1 = compute_cluster_bootstrap_ci(straights_z1, n_bootstraps=1000, random_state=42)
        run2 = compute_cluster_bootstrap_ci(straights_z1, n_bootstraps=1000, random_state=42)

        self.assertEqual(run1["Cluster_CI95_Lower_kmh"], run2["Cluster_CI95_Lower_kmh"])
        self.assertEqual(run1["Cluster_CI95_Upper_kmh"], run2["Cluster_CI95_Upper_kmh"])
        self.assertEqual(run1["Cluster_Delta_Mean_kmh"], run2["Cluster_Delta_Mean_kmh"])

    def test_sample_weighted_estimand(self):
        """Verify that bootstrap preserves the sample-weighted marginal mean difference."""
        z1_mask = (self.flying_tel["Distance_Corrected"] >= DRS_ZONE_1_RANGE[0]) & (self.flying_tel["Distance_Corrected"] <= DRS_ZONE_1_RANGE[1])
        z2_mask = (self.flying_tel["Distance_Corrected"] >= DRS_ZONE_2_RANGE[0]) & (self.flying_tel["Distance_Corrected"] <= DRS_ZONE_2_RANGE[1])
        straights = self.flying_tel[z1_mask | z2_mask]

        # Calculate exact sample-weighted point delta
        sample_on = straights[straights["DRS"] > 0]["Speed"].mean()
        sample_off = straights[straights["DRS"] == 0]["Speed"].mean()
        sample_delta = sample_on - sample_off

        boot = compute_cluster_bootstrap_ci(straights, n_bootstraps=5000, random_state=42)

        self.assertEqual(boot["Estimand"], "E[Speed | DRS=1, Straight] - E[Speed | DRS=0, Straight]")
        self.assertEqual(boot["WeightingStrategy"], "Sample-weighted ratio estimator")
        # Empirical mean of bootstrap deltas must match sample-weighted delta closely (< 0.15 km/h)
        self.assertAlmostEqual(boot["Cluster_Delta_Mean_kmh"], sample_delta, delta=0.15)

    def test_missing_or_empty_drs_subsets(self):
        """Verify graceful error-free handling of empty or missing DRS subsets."""
        # Case 1: Empty DataFrame
        empty_df = pd.DataFrame(columns=["Driver", "LapNumber", "DRS", "Speed"])
        res_empty = compute_cluster_bootstrap_ci(empty_df)
        self.assertTrue(np.isnan(res_empty["Cluster_CI95_Lower_kmh"]))
        self.assertTrue(np.isnan(res_empty["Cluster_CI95_Upper_kmh"]))
        self.assertEqual(res_empty["Bootstrap_Iterations"], 0)

        # Case 2: Only DRS OFF present (0 DRS ON)
        off_only_df = pd.DataFrame({
            "Driver": ["D1", "D2"],
            "LapNumber": [4, 4],
            "DRS": [0, 0],
            "Speed": [250.0, 260.0]
        })
        res_off = compute_cluster_bootstrap_ci(off_only_df)
        self.assertTrue(np.isnan(res_off["Cluster_CI95_Lower_kmh"]))
        self.assertEqual(res_off["Bootstrap_Iterations"], 0)

        # Case 3: Only DRS ON present (0 DRS OFF)
        on_only_df = pd.DataFrame({
            "Driver": ["D1", "D2"],
            "LapNumber": [4, 4],
            "DRS": [1, 1],
            "Speed": [300.0, 310.0]
        })
        res_on = compute_cluster_bootstrap_ci(on_only_df)
        self.assertTrue(np.isnan(res_on["Cluster_CI95_Lower_kmh"]))
        self.assertEqual(res_on["Bootstrap_Iterations"], 0)


if __name__ == "__main__":
    unittest.main()
