"""
F1 Telemetry Intelligence - Gear & RPM Analytics Module (Phase 5 EDA)
====================================================================

Analyzes internal combustion engine RPM operating bands, 8-speed transmission
gear utilization, shift patterns, and empirical speed-to-gear ratios.
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


def compute_gear_usage_statistics(
    tel_df: pd.DataFrame,
    only_flying: bool = True
) -> pd.DataFrame:
    """Analyze percentage time and speed range in each of the 8 forward gears."""
    df = tel_df.copy()
    if only_flying and "LapClassification" in df.columns:
        df = df[df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    records = []
    total_samples = len(df)
    for g, grp in df.groupby("nGear"):
        g = int(g)
        cnt = len(grp)
        pct = (cnt / total_samples) * 100.0
        records.append({
            "Gear": g,
            "SampleCount": cnt,
            "UsagePercentage": round(float(pct), 2),
            "MeanSpeed_kmh": round(float(grp["Speed"].mean()), 1),
            "MinSpeed_kmh": round(float(grp["Speed"].min()), 1),
            "MaxSpeed_kmh": round(float(grp["Speed"].max()), 1),
            "MeanRPM": round(float(grp["RPM"].mean()), 1),
            "MaxRPM": int(grp["RPM"].max())
        })

    out_df = pd.DataFrame(records).sort_values("Gear").reset_index(drop=True)
    out_df.to_csv(TABLES_DIR / "07_gear_usage_statistics.csv", index=False)
    return out_df


def compute_gear_shifts_per_lap(
    tel_df: pd.DataFrame
) -> pd.DataFrame:
    """Count total gear shifts (both upshifts and downshifts) per driver-lap sequence."""
    records = []
    for (drv, lap_no), grp in tel_df.groupby(["Driver", "LapNumber"]):
        gears = grp["nGear"].values
        shifts = np.sum(np.diff(gears) != 0)
        classification = grp["LapClassification"].iloc[0] if "LapClassification" in grp else "UNKNOWN"
        records.append({
            "Driver": drv,
            "LapNumber": lap_no,
            "LapClassification": classification,
            "TotalGearShifts": int(shifts)
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(TABLES_DIR / "07_gear_shifts_per_lap.csv", index=False)
    return out_df


def plot_speed_vs_rpm_by_gear(
    tel_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """
    Scatter plot of RPM vs Speed colored by gear.
    Renders the characteristic distinct rays of 8 transmission gear ratios.
    """
    df_clean = tel_df[(tel_df["LapClassification"] == "COMPLETE_FLYING_LAP") & (tel_df["RPM"] > 4000)].copy()

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)

    # Palette for gears 1-8
    gear_palette = {
        1: "#94a3b8", 2: "#64748b", 3: "#38bdf8", 4: "#06b6d4",
        5: "#10b981", 6: "#f59e0b", 7: "#f97316", 8: "#ef4444"
    }

    for g in sorted(df_clean["nGear"].unique()):
        sub = df_clean[df_clean["nGear"] == g]
        ax.scatter(sub["Speed"], sub["RPM"], c=gear_palette.get(g, "#ffffff"),
                   label=f"Gear {int(g)}", alpha=0.35, s=12, edgecolors="none")

    ax.set_title("Transmission Gear Ratio Map: Engine RPM vs Vehicle Speed (Gears 1–8)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Vehicle Speed (km/h)", fontsize=11, labelpad=10)
    ax.set_ylabel("Engine RPM (rev/min)", fontsize=11, labelpad=10)
    ax.set_ylim(4000, 13000)
    ax.set_xlim(50, 345)
    ax.grid(alpha=0.2, color="#94a3b8")
    ax.legend(loc="lower right", frameon=True, facecolor="#1e293b", title="Transmission Gear")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "07_speed_vs_rpm_gear_ratios.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_gear_usage_distribution(
    gear_stats_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """Bar chart of gear utilization percentages across the circuit."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    bars = ax.bar(gear_stats_df["Gear"].astype(str), gear_stats_df["UsagePercentage"],
                  color="#3b82f6", edgecolor="#ffffff", linewidth=0.8, alpha=0.85)

    ax.set_title("Gear Usage Distribution on Clean Flying Laps (% of Lap Time)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Transmission Gear (1 to 8)", fontsize=11, labelpad=10)
    ax.set_ylabel("Usage Percentage (%)", fontsize=11, labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#ffffff")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "07_gear_usage_bar.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
