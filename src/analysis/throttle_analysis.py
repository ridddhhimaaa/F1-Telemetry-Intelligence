"""
F1 Telemetry Intelligence - Throttle Analytics Module (Phase 5 EDA)
==================================================================

Analyzes throttle application regimes (full throttle, modulation, off-throttle),
corner exit traction characteristics, and driver acceleration aggressiveness.
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


def compute_throttle_regimes_by_driver(
    tel_df: pd.DataFrame,
    only_flying: bool = True
) -> pd.DataFrame:
    """
    Categorize throttle samples into three fundamental racing regimes:
      1. Off-Throttle (<= 5.0%)
      2. Partial Modulation (5.0% - 98.0%)
      3. Full Throttle (>= 98.0%)
    """
    df = tel_df.copy()
    if only_flying and "LapClassification" in df.columns:
        df = df[df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    records = []
    for drv, grp in df.groupby("Driver"):
        thr = grp["Throttle_Clipped"] if "Throttle_Clipped" in grp else grp["Throttle"]
        n_total = len(thr)

        off_pct = (thr <= 5.0).mean() * 100.0
        part_pct = ((thr > 5.0) & (thr < 98.0)).mean() * 100.0
        full_pct = (thr >= 98.0).mean() * 100.0

        records.append({
            "Driver": drv,
            "TotalSamples": n_total,
            "OffThrottle_pct": round(float(off_pct), 2),
            "PartialThrottle_pct": round(float(part_pct), 2),
            "FullThrottle_pct": round(float(full_pct), 2),
            "MeanThrottle_pct": round(float(thr.mean()), 2),
            "MedianThrottle_pct": round(float(thr.median()), 2)
        })

    out_df = pd.DataFrame(records).sort_values("FullThrottle_pct", ascending=False).reset_index(drop=True)
    out_df.to_csv(TABLES_DIR / "06_driver_throttle_regimes.csv", index=False)
    return out_df


def plot_throttle_distribution_bimodal(
    tel_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """Plot bimodal throttle distribution (off-throttle vs full-throttle peaks)."""
    df_flying = tel_df[tel_df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    sns.histplot(df_flying["Throttle_Clipped"], bins=50, kde=True, color="#10b981",
                 ax=ax, edgecolor="#1e293b", alpha=0.8)

    ax.set_title("Bimodal Throttle Distribution on Clean Flying Laps (Laps 4 & 5)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Throttle Application (%)", fontsize=11, labelpad=10)
    ax.set_ylabel("Observation Count", fontsize=11, labelpad=10)
    ax.grid(alpha=0.2, color="#94a3b8")

    # Annotate the two peaks
    ax.annotate("Coasting / Braking Peak\n(Throttle = 0%)",
                xy=(2, 3500), xytext=(15, 4500),
                arrowprops=dict(facecolor="#f59e0b", shrink=0.05, width=1, headwidth=6),
                fontsize=9, fontweight="bold", color="#f59e0b")

    ax.annotate("Full Throttle Peak\n(Throttle = 100%)",
                xy=(98, 9000), xytext=(65, 8500),
                arrowprops=dict(facecolor="#10b981", shrink=0.05, width=1, headwidth=6),
                fontsize=9, fontweight="bold", color="#10b981")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "06_throttle_bimodal_distribution.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_throttle_regimes_stacked_bar(
    regimes_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """Stacked horizontal bar chart of Off-Throttle vs Partial vs Full Throttle."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    drivers = regimes_df["Driver"]
    off = regimes_df["OffThrottle_pct"]
    part = regimes_df["PartialThrottle_pct"]
    full = regimes_df["FullThrottle_pct"]

    p1 = ax.barh(drivers, full, color="#10b981", edgecolor="#1e293b", label="Full Throttle (>= 98%)")
    p2 = ax.barh(drivers, part, left=full, color="#f59e0b", edgecolor="#1e293b", label="Partial Modulation (5-98%)")
    p3 = ax.barh(drivers, off, left=full + part, color="#ef4444", edgecolor="#1e293b", label="Off-Throttle (<= 5%)")

    ax.set_title("Driver Throttle Operating Regimes (% of Flying Lap)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Percentage of Lap Distance", fontsize=11, labelpad=10)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", frameon=True, facecolor="#1e293b")
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#94a3b8")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "06_throttle_regimes_stacked_bar.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
