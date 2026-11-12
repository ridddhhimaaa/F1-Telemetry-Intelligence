"""
F1 Telemetry Intelligence - Driver Performance Analysis Module (Phase 5 EDA)
=============================================================================

Computes driver-level comparative performance metrics: speed distributions,
throttle usage profiles, braking engagement percentages, and lap-time consistency
across comparable green-flag racing laps (Laps 4 & 5).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "eda_report" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "eda_report" / "tables"


def compute_driver_telemetry_metrics(
    tel_df: pd.DataFrame,
    only_flying_laps: bool = True
) -> pd.DataFrame:
    """
    Compute comprehensive driver-by-driver kinematic performance metrics.
    When only_flying_laps is True, evaluates strictly COMPLETE_FLYING_LAP (Laps 4 & 5).
    """
    df = tel_df.copy()
    if only_flying_laps and "LapClassification" in df.columns:
        df = df[df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    records = []
    for drv, grp in df.groupby("Driver"):
        sp = grp["Speed"]
        thr = grp["Throttle_Clipped"] if "Throttle_Clipped" in grp else grp["Throttle"]
        brk = grp["IsBraking"] if "IsBraking" in grp else (grp["Brake"] > 0)
        rpm = grp["RPM"]
        gear = grp["nGear"]

        full_thr_pct = (thr >= 98.0).mean() * 100.0
        braking_pct = brk.mean() * 100.0

        records.append({
            "Driver": drv,
            "LapsEvaluated": grp["LapNumber"].nunique(),
            "SampleCount": len(grp),
            "MeanSpeed_kmh": round(float(sp.mean()), 2),
            "MedianSpeed_kmh": round(float(sp.median()), 2),
            "MaxSpeed_kmh": round(float(sp.max()), 2),
            "StdSpeed_kmh": round(float(sp.std()), 2),
            "MeanThrottle_pct": round(float(thr.mean()), 2),
            "FullThrottle_pct": round(float(full_thr_pct), 2),
            "Braking_pct": round(float(braking_pct), 2),
            "MeanRPM": round(float(rpm.mean()), 1),
            "MaxRPM": int(rpm.max()),
            "MeanGear": round(float(gear.mean()), 2)
        })

    out_df = pd.DataFrame(records).sort_values("MeanSpeed_kmh", ascending=False).reset_index(drop=True)
    out_df["SpeedRank"] = out_df["MeanSpeed_kmh"].rank(ascending=False, method="min").astype(int)
    out_df.to_csv(TABLES_DIR / "03_driver_performance_metrics.csv", index=False)
    return out_df


def compute_driver_lap_time_stats(
    laps_df: pd.DataFrame,
    only_accurate_green: bool = True
) -> pd.DataFrame:
    """Compute official lap-time statistics per driver from processed_race_laps."""
    df = laps_df.copy()
    # Filter to first 5 laps (ingestion window)
    df = df[df["LapNumber"] <= 5]
    if only_accurate_green:
        df = df[df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    records = []
    for drv, grp in df.groupby("Driver"):
        valid_times = grp["LapTimeSeconds"].dropna()
        if len(valid_times) == 0:
            continue
        team = grp["team"].iloc[0] if "team" in grp else (grp["Team"].iloc[0] if "Team" in grp else "Unknown")
        records.append({
            "Driver": drv,
            "Team": team,
            "FlyingLaps": len(valid_times),
            "BestLapTime_s": round(float(valid_times.min()), 3),
            "MeanLapTime_s": round(float(valid_times.mean()), 3),
            "StdLapTime_s": round(float(valid_times.std()), 3) if len(valid_times) > 1 else 0.0
        })

    out_df = pd.DataFrame(records).sort_values("BestLapTime_s").reset_index(drop=True)
    out_df["PaceRank"] = out_df["BestLapTime_s"].rank(ascending=True, method="min").astype(int)
    out_df.to_csv(TABLES_DIR / "03_driver_lap_time_stats.csv", index=False)
    return out_df


def plot_driver_speed_distribution(
    tel_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """Boxplot of driver speed distributions during clean flying laps (Laps 4 & 5)."""
    df = tel_df[tel_df["LapClassification"] == "COMPLETE_FLYING_LAP"]
    order = df.groupby("Driver")["Speed"].median().sort_values(ascending=False).index

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    sns.boxplot(data=df, x="Driver", y="Speed", order=order, ax=ax,
                hue="Driver", palette="mako", legend=False, boxprops=dict(alpha=0.8), showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "#e10600", "markeredgecolor": "#ffffff"})

    ax.set_title("Driver Speed Distributions Across Clean Flying Laps (Laps 4 & 5, N=38 laps)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Driver (Ordered by Median Speed)", fontsize=11, labelpad=10)
    ax.set_ylabel("Speed (km/h)", fontsize=11, labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "03_driver_speed_boxplot.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_driver_throttle_vs_braking(
    driver_metrics_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """Scatter comparison of Full Throttle % vs Braking % per driver."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = driver_metrics_df["FullThrottle_pct"]
    y = driver_metrics_df["Braking_pct"]
    labels = driver_metrics_df["Driver"]

    scatter = ax.scatter(x, y, c=driver_metrics_df["MeanSpeed_kmh"], cmap="turbo",
                         s=140, edgecolors="#ffffff", alpha=0.9)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Mean Speed (km/h)", fontsize=10, labelpad=8)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x.iloc[i], y.iloc[i]), xytext=(5, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#ffffff")

    ax.set_title("Driver Control Regimes: Full Throttle % vs Braking Engagement % (Laps 4 & 5)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Full Throttle Usage (% of Lap with Throttle >= 98%)", fontsize=10, labelpad=8)
    ax.set_ylabel("Braking Engagement (% of Lap with IsBraking = True)", fontsize=10, labelpad=8)
    ax.grid(alpha=0.2, color="#94a3b8")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "03_driver_throttle_vs_braking.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
