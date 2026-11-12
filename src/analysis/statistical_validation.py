"""
F1 Telemetry Intelligence - Statistical Validation & Inference Module (Phase 5.1)
==================================================================================

Provides rigorous statistical routines for evaluating telemetry distributions,
explicitly addressing serial autocorrelation, cluster dependence, and observational
confounding:

1. Descriptive Sample-Level Statistics:
   Calculates Mean, Median, Standard Deviation, Welch's t-test, Mann-Whitney U,
   and Cohen's d across high-frequency sample points. Explicitly documents that
   sample-level p-values are descriptive only, because 10 Hz telemetry violates
   the independent and identically distributed (i.i.d.) observations assumption.

2. Cluster-Aware Bootstrap Confidence Intervals:
   Resamples independent driver-lap clusters with replacement (N=38 clean flying laps
   across 19 drivers) to compute valid confidence intervals that account for intra-cluster
   autocorrelation (ACF_1 ≈ 0.997).

3. Driver-Level Aggregated Analysis:
   Aggregates telemetry to driver-level and lap-level units before comparative inference,
   distinguishing independent experimental units from repeated sensor observations.

4. Methodological Distinction (Correlation vs Observational Control vs Causality):
   Explicitly notes that controlling for track distance isolates track topography confounding,
   yielding a location-controlled observational association (+12.75 km/h on straights),
   which remains subject to residual confounding (slipstream/tow, ERS deployment, entry speed)
   and cannot be claimed as a pure causal aerodynamic drag benefit without a formal causal design.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("f1_telemetry.analysis.statistical_validation")


def compute_sample_level_stats(
    s_on: pd.Series,
    s_off: pd.Series,
    metric_name: str = "Speed"
) -> Dict[str, Any]:
    """
    Computes classical two-sample descriptive metrics between two telemetry series.

    NOTE: Due to extreme serial autocorrelation (ACF_1 ≈ 0.997), sample-level p-values
    violate i.i.d. assumptions and are reported strictly as descriptive indicators of
    sample separation, NOT as proof of independent statistical significance.
    """
    n1, n2 = len(s_on), len(s_off)
    m1, m2 = float(s_on.mean()) if n1 > 0 else 0.0, float(s_off.mean()) if n2 > 0 else 0.0
    std1, std2 = float(s_on.std()) if n1 > 1 else 0.0, float(s_off.std()) if n2 > 1 else 0.0
    med1, med2 = float(s_on.median()) if n1 > 0 else 0.0, float(s_off.median()) if n2 > 0 else 0.0
    delta = m1 - m2

    # Welch's t-test (unequal variances assumed)
    if n1 > 1 and n2 > 1 and (std1 > 0 or std2 > 0):
        t_stat, t_pval = stats.ttest_ind(s_on, s_off, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(s_on, s_off, alternative="two-sided")
    else:
        t_stat, t_pval = 0.0, 1.0
        u_stat, u_pval = 0.0, 1.0

    # Cohen's d (pooled standard deviation)
    if n1 + n2 > 2:
        s_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = delta / s_pooled if s_pooled > 0 else 0.0
    else:
        s_pooled = 0.0
        cohens_d = 0.0

    # Naive sample-level 95% Confidence Interval
    se_delta = np.sqrt((std1**2 / n1) + (std2**2 / n2)) if n1 > 0 and n2 > 0 else 0.0
    ci_lower = delta - 1.96 * se_delta
    ci_upper = delta + 1.96 * se_delta

    return {
        "N_DRS_ON": n1,
        "N_DRS_OFF": n2,
        "Mean_DRS_ON_kmh": round(m1, 2),
        "Mean_DRS_OFF_kmh": round(m2, 2),
        "Median_DRS_ON_kmh": round(med1, 2),
        "Median_DRS_OFF_kmh": round(med2, 2),
        "Std_DRS_ON_kmh": round(std1, 2),
        "Std_DRS_OFF_kmh": round(std2, 2),
        "Delta_Mean_kmh": round(delta, 2),
        "Delta_Median_kmh": round(med1 - med2, 2),
        "CI95_Lower_kmh": round(ci_lower, 2),
        "CI95_Upper_kmh": round(ci_upper, 2),
        "t_statistic": round(float(t_stat), 3),
        "p_value_ttest": float(t_pval),
        "mann_whitney_u": float(u_stat),
        "p_value_mann_whitney": float(u_pval),
        "cohens_d": round(float(cohens_d), 3),
        "is_descriptive_sample_level": True,
        "inference_warning": (
            "Telemetry points are non-independent time-series samples. "
            "Reported p-values are unadjusted descriptive statistics."
        )
    }


def compute_cluster_bootstrap_ci(
    df: pd.DataFrame,
    cluster_cols: Optional[List[str]] = None,
    treatment_col: str = "DRS",
    value_col: str = "Speed",
    n_bootstraps: int = 5000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes a cluster-aware bootstrap 95% confidence interval for the mean delta.

    Hierarchically resamples entire driver clusters (default: 19 drivers, retaining all
    eligible laps per driver) with replacement to account for within-driver and within-lap
    serial autocorrelation and driver heterogeneity.

    Estimand:
        E[Speed | DRS=1, Straight] - E[Speed | DRS=0, Straight]
        Preserves the sample-weighted marginal mean difference across straightaways.

    Parameters:
        df: Telemetry DataFrame restricted to straightaway evaluation zones.
        cluster_cols: Columns defining independent clusters (default: ['Driver']).
        treatment_col: Column indicating treatment (e.g. 'DRS' > 0 vs == 0).
        value_col: Target continuous metric (e.g. 'Speed').
        n_bootstraps: Number of bootstrap iterations (default: 5000).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing cluster-bootstrap CI bounds and cluster count metadata:
          - Cluster_CI95_Lower_kmh: 2.5th percentile bootstrap delta.
          - Cluster_CI95_Upper_kmh: 97.5th percentile bootstrap delta.
          - Cluster_Delta_Mean_kmh: Empirical mean bootstrap delta.
          - N_Clusters: Number of sampled cluster units (e.g. 19 drivers).
          - N_Drivers: Number of unique drivers in evaluation set.
          - Bootstrap_Iterations: Number of valid bootstrap iterations evaluated.
          - ResamplingUnit: Unit of resampling ("Driver (all eligible laps retained)").
          - WeightingStrategy: Weighting applied ("Sample-weighted ratio estimator").
          - Estimand: Statistical target estimand description.
          - SamplingUnit: Human-readable sampling unit description.
    """
    if df is None or len(df) == 0:
        return {
            "Cluster_CI95_Lower_kmh": np.nan,
            "Cluster_CI95_Upper_kmh": np.nan,
            "Cluster_Delta_Mean_kmh": np.nan,
            "N_Clusters": 0,
            "N_Drivers": 0,
            "Bootstrap_Iterations": 0,
            "ResamplingUnit": "Driver (both eligible laps retained)",
            "WeightingStrategy": "Sample-weighted ratio estimator",
            "Estimand": "E[Speed | DRS=1, Straight] - E[Speed | DRS=0, Straight]",
            "SamplingUnit": "0 clusters"
        }

    c_cols = cluster_cols or ["Driver"]
    clusters = [grp for _, grp in df.groupby(c_cols)]
    n_clusters = len(clusters)
    n_drivers = df["Driver"].nunique() if "Driver" in df.columns else n_clusters
    n_driver_laps = (
        df.groupby(["Driver", "LapNumber"]).ngroups
        if {"Driver", "LapNumber"}.issubset(df.columns)
        else n_clusters
    )

    if n_clusters < 5:
        logger.warning("Fewer than 5 clusters available for cluster bootstrap; CI may be unstable.")

    # Pre-extract cluster aggregate statistics to vectorize bootstrap sampling
    # [on_cnt, on_sum, off_cnt, off_sum]
    c_stats_list = []
    for grp in clusters:
        on_vals = grp[grp[treatment_col] > 0][value_col].to_numpy(dtype=np.float64)
        off_vals = grp[grp[treatment_col] == 0][value_col].to_numpy(dtype=np.float64)
        c_stats_list.append([
            len(on_vals),
            np.sum(on_vals) if len(on_vals) > 0 else 0.0,
            len(off_vals),
            np.sum(off_vals) if len(off_vals) > 0 else 0.0
        ])
    c_stats = np.array(c_stats_list, dtype=np.float64)

    rng = np.random.default_rng(random_state)
    # Sample matrix of shape (n_bootstraps, n_clusters)
    sampled_idx = rng.choice(n_clusters, size=(n_bootstraps, n_clusters), replace=True)
    sampled = c_stats[sampled_idx]

    on_cnt = sampled[:, :, 0].sum(axis=1)
    on_sum = sampled[:, :, 1].sum(axis=1)
    off_cnt = sampled[:, :, 2].sum(axis=1)
    off_sum = sampled[:, :, 3].sum(axis=1)

    resampling_unit_str = (
        "Driver (both eligible laps retained)"
        if c_cols == ["Driver"]
        else f"Cluster ({', '.join(c_cols)})"
    )
    sampling_unit_str = (
        f"{n_clusters} drivers ({n_driver_laps} driver-laps retained)"
        if c_cols == ["Driver"] and {"Driver", "LapNumber"}.issubset(df.columns)
        else f"{n_clusters} clusters"
    )

    # Missing / empty DRS subset handling
    valid_mask = (on_cnt > 0) & (off_cnt > 0)
    if not np.any(valid_mask):
        return {
            "Cluster_CI95_Lower_kmh": np.nan,
            "Cluster_CI95_Upper_kmh": np.nan,
            "Cluster_Delta_Mean_kmh": np.nan,
            "N_Clusters": n_clusters,
            "N_Drivers": n_drivers,
            "Bootstrap_Iterations": 0,
            "ResamplingUnit": resampling_unit_str,
            "WeightingStrategy": "Sample-weighted ratio estimator",
            "Estimand": "E[Speed | DRS=1, Straight] - E[Speed | DRS=0, Straight]",
            "SamplingUnit": sampling_unit_str
        }

    boot_deltas = (on_sum[valid_mask] / on_cnt[valid_mask]) - (off_sum[valid_mask] / off_cnt[valid_mask])
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    mean_boot_delta = float(np.mean(boot_deltas))

    return {
        "Cluster_CI95_Lower_kmh": round(ci_lower, 2),
        "Cluster_CI95_Upper_kmh": round(ci_upper, 2),
        "Cluster_Delta_Mean_kmh": round(mean_boot_delta, 2),
        "N_Clusters": n_clusters,
        "N_Drivers": n_drivers,
        "Bootstrap_Iterations": int(np.sum(valid_mask)),
        "ResamplingUnit": resampling_unit_str,
        "WeightingStrategy": "Sample-weighted ratio estimator",
        "Estimand": "E[Speed | DRS=1, Straight] - E[Speed | DRS=0, Straight]",
        "SamplingUnit": sampling_unit_str
    }


def compute_driver_level_aggregated_stats(
    df: pd.DataFrame,
    treatment_col: str = "DRS",
    value_col: str = "Speed"
) -> Dict[str, Any]:
    """
    Computes aggregated driver-level comparative statistics on straightaways.

    For drivers who executed straightaways with both DRS ON and DRS OFF,
    calculates intra-driver paired differences to remove between-driver car confounding.
    """
    driver_on = df[df[treatment_col] > 0].groupby("Driver")[value_col].mean()
    driver_off = df[df[treatment_col] == 0].groupby("Driver")[value_col].mean()

    common_drivers = sorted(list(set(driver_on.index).intersection(set(driver_off.index))))
    paired_df = pd.DataFrame({
        "DRS_ON": driver_on.loc[common_drivers],
        "DRS_OFF": driver_off.loc[common_drivers]
    })
    paired_df["Delta"] = paired_df["DRS_ON"] - paired_df["DRS_OFF"]

    n_paired = len(paired_df)
    if n_paired >= 2:
        mean_delta = float(paired_df["Delta"].mean())
        std_delta = float(paired_df["Delta"].std())
        se_delta = std_delta / np.sqrt(n_paired)
        t_stat, p_val = stats.ttest_rel(paired_df["DRS_ON"], paired_df["DRS_OFF"])
        ci_lower = mean_delta - 1.96 * se_delta
        ci_upper = mean_delta + 1.96 * se_delta
    else:
        mean_delta = float(paired_df["Delta"].mean()) if n_paired == 1 else np.nan
        std_delta = np.nan
        t_stat, p_val = np.nan, np.nan
        ci_lower, ci_upper = np.nan, np.nan

    return {
        "N_Paired_Drivers": n_paired,
        "Driver_Mean_Delta_kmh": round(mean_delta, 2) if not np.isnan(mean_delta) else np.nan,
        "Driver_Std_Delta_kmh": round(std_delta, 2) if not np.isnan(std_delta) else np.nan,
        "Driver_Paired_t_stat": round(float(t_stat), 3) if not np.isnan(t_stat) else np.nan,
        "Driver_Paired_p_val": float(p_val) if not np.isnan(p_val) else np.nan,
        "Driver_CI95_Lower_kmh": round(ci_lower, 2) if not np.isnan(ci_lower) else np.nan,
        "Driver_CI95_Upper_kmh": round(ci_upper, 2) if not np.isnan(ci_upper) else np.nan
    }
