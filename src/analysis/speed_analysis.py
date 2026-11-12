"""
F1 Telemetry Intelligence - Speed Analytics Module (Phase 5 EDA)
===============================================================

Analyzes velocity distributions, terminal speeds along straightaways, apex speeds
in slow corners, and DRS acceleration deltas across clean flying laps.
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


def compute_speed_percentiles_by_driver(
    tel_df: pd.DataFrame,
    only_flying: bool = True
) -> pd.DataFrame:
    """Compute detailed speed percentiles per driver."""
    df = tel_df.copy()
    if only_flying and "LapClassification" in df.columns:
        df = df[df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    records = []
    for drv, grp in df.groupby("Driver"):
        sp = grp["Speed"]
        records.append({
            "Driver": drv,
            "Mean_kmh": round(float(sp.mean()), 2),
            "Min_kmh": round(float(sp.min()), 2),
            "P25_kmh": round(float(sp.quantile(0.25)), 2),
            "Median_kmh": round(float(sp.median()), 2),
            "P75_kmh": round(float(sp.quantile(0.75)), 2),
            "P95_kmh": round(float(sp.quantile(0.95)), 2),
            "P99_kmh": round(float(sp.quantile(0.99)), 2),
            "Max_kmh": round(float(sp.max()), 2),
            "Std_kmh": round(float(sp.std()), 2)
        })

    out_df = pd.DataFrame(records).sort_values("MaxSpeed_kmh" if "MaxSpeed_kmh" in records[0] else "Max_kmh", ascending=False).reset_index(drop=True)
    out_df.to_csv(TABLES_DIR / "04_driver_speed_percentiles.csv", index=False)
    return out_df


def analyze_drs_speed_impact(tel_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate terminal speed and acceleration differences with DRS active vs inactive."""
    df = tel_df[tel_df["LapClassification"] == "COMPLETE_FLYING_LAP"].copy()
    # DRS active: 1 in processed data, or 10, 12, 14 in raw bitmasks
    drs_active = df[df["DRS"].isin([1, 10, 12, 14]) | (df["DRS"] > 0)]
    drs_inactive = df[df["DRS"] == 0]

    stats = {
        "DRS_Active_MeanSpeed": round(float(drs_active["Speed"].mean()), 2) if len(drs_active) > 0 else 0.0,
        "DRS_Active_MaxSpeed": round(float(drs_active["Speed"].max()), 2) if len(drs_active) > 0 else 0.0,
        "DRS_Inactive_MeanSpeed": round(float(drs_inactive["Speed"].mean()), 2),
        "DRS_Inactive_MaxSpeed": round(float(drs_inactive["Speed"].max()), 2),
        "DRS_Active_Samples": len(drs_active),
        "DRS_Inactive_Samples": len(drs_inactive)
    }
    pd.DataFrame([stats]).to_csv(TABLES_DIR / "04_drs_speed_impact.csv", index=False)
    return stats


def plot_speed_distribution_by_classification(
    tel_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """Compare speed density curves across Start Lap, VSC Laps, and Clean Flying Laps."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    palette = {
        "COMPLETE_FLYING_LAP": "#10b981",
        "VSC_AFFECTED_LAP": "#f59e0b",
        "START_LAP": "#3b82f6",
        "DNF_OR_RETIRED": "#ef4444"
    }

    for cat in ["COMPLETE_FLYING_LAP", "VSC_AFFECTED_LAP", "START_LAP"]:
        sub = tel_df[tel_df["LapClassification"] == cat]
        if len(sub) > 0:
            sns.kdeplot(sub["Speed"], label=f"{cat} (Mean: {sub['Speed'].mean():.1f} km/h)",
                        color=palette[cat], linewidth=2.2, ax=ax)

    ax.set_title("Speed Density Distribution by Tactical Lap Classification", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Vehicle Speed (km/h)", fontsize=11, labelpad=10)
    ax.set_ylabel("Kernel Density", fontsize=11, labelpad=10)
    ax.grid(alpha=0.2, color="#94a3b8")
    ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "04_speed_distribution_by_classification.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_speed_distance_trace(
    res_df: pd.DataFrame,
    drivers: List[str] = ["NOR", "VER", "LEC", "HAM"],
    lap_number: int = 5,
    save_path: Optional[Path] = None
) -> Path:
    """Plot synchronized speed vs distance traces along Yas Marina track."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)

    colors = {"NOR": "#ff8000", "VER": "#3671c6", "LEC": "#e80020", "HAM": "#27f4d2"}

    for drv in drivers:
        sub = res_df[(res_df["Driver"] == drv) & (res_df["LapNumber"] == lap_number)]
        if len(sub) > 0:
            ax.plot(sub["Distance_Meter"], sub["Speed"], label=f"{drv} (Lap {lap_number})",
                    color=colors.get(drv, "#ffffff"), linewidth=1.4, alpha=0.9)

    ax.set_title(f"Synchronized Spatial Speed Comparison (Lap {lap_number}, 1.0m Uniform Grid)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Circuit Distance along Track (metres)", fontsize=11, labelpad=10)
    ax.set_ylabel("Speed (km/h)", fontsize=11, labelpad=10)
    ax.set_xlim(0, 5281)
    ax.grid(alpha=0.2, color="#94a3b8")
    ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "04_speed_distance_trace.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
