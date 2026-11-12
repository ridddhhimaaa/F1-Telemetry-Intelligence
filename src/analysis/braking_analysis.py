"""
F1 Telemetry Intelligence - Braking Analytics Module (Phase 5 EDA)
==================================================================

Identifies circuit braking zones, computes deceleration magnitudes,
braking zone lengths, entry/apex velocities, and compares driver braking points.
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


def detect_circuit_braking_zones(
    res_df: pd.DataFrame,
    lap_number: int = 5,
    min_length_m: float = 20.0
) -> pd.DataFrame:
    """
    Detect major braking zones along the circuit using median driver profile on Lap 5.
    Returns table of identified braking zones, start/end distance, and speed reduction.
    """
    lap5 = res_df[res_df["LapNumber"] == lap_number]
    if lap5.empty:
        lap5 = res_df[res_df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    # Compute consensus braking engagement along the 1m grid
    grid_summary = lap5.groupby("Distance_Meter").agg({
        "IsBraking": "mean",
        "Speed": "mean"
    }).reset_index()

    # Braking zone where > 40% of drivers are braking
    grid_summary["InBrakingZone"] = grid_summary["IsBraking"] > 0.40

    zones = []
    zone_active = False
    z_start = 0.0

    for idx, row in grid_summary.iterrows():
        d = row["Distance_Meter"]
        is_brk = row["InBrakingZone"]
        if is_brk and not zone_active:
            zone_active = True
            z_start = d
        elif not is_brk and zone_active:
            zone_active = False
            z_end = d
            length = z_end - z_start
            if length >= min_length_m:
                sub = grid_summary[(grid_summary["Distance_Meter"] >= z_start) & (grid_summary["Distance_Meter"] <= z_end)]
                v_entry = sub["Speed"].iloc[0]
                v_min = sub["Speed"].min()
                v_drop = v_entry - v_min
                zones.append({
                    "BrakingZoneID": f"BZ_{len(zones)+1}",
                    "StartDistance_m": z_start,
                    "EndDistance_m": z_end,
                    "BrakingLength_m": length,
                    "EntrySpeed_kmh": round(float(v_entry), 1),
                    "ApexSpeed_kmh": round(float(v_min), 1),
                    "SpeedDelta_kmh": round(float(v_drop), 1)
                })

    out_df = pd.DataFrame(zones)
    out_df.to_csv(TABLES_DIR / "05_circuit_braking_zones.csv", index=False)
    return out_df


def compare_turn6_braking_points(
    res_df: pd.DataFrame,
    lap_number: int = 5
) -> pd.DataFrame:
    """
    Compare driver braking points into Turn 6 (end of long back straight, ~2200m - 2550m).
    Turn 6 is the heaviest deceleration zone at Yas Marina (~330 km/h to ~80 km/h).
    """
    t6_slice = res_df[(res_df["LapNumber"] == lap_number) &
                      (res_df["Distance_Meter"] >= 2200.0) &
                      (res_df["Distance_Meter"] <= 2550.0)]

    records = []
    for drv, grp in t6_slice.groupby("Driver"):
        # First point where driver touches brake
        brk_pts = grp[grp["IsBraking"] == 1]
        if len(brk_pts) > 0:
            init_d = float(brk_pts["Distance_Meter"].iloc[0])
            init_v = float(grp[grp["Distance_Meter"] == init_d]["Speed"].iloc[0])
            min_v = float(grp["Speed"].min())
            records.append({
                "Driver": drv,
                "LapNumber": lap_number,
                "BrakingPoint_m": init_d,
                "EntrySpeed_kmh": round(init_v, 1),
                "ApexSpeed_kmh": round(min_v, 1),
                "SpeedDelta_kmh": round(init_v - min_v, 1)
            })

    if not records:
        return pd.DataFrame(columns=["Driver", "LapNumber", "BrakingPoint_m", "EntrySpeed_kmh", "ApexSpeed_kmh", "SpeedDelta_kmh", "BrakingRank"])

    out_df = pd.DataFrame(records).sort_values("BrakingPoint_m", ascending=False).reset_index(drop=True)
    out_df["BrakingRank"] = range(1, len(out_df) + 1)  # Higher distance = braked later!
    out_df.to_csv(TABLES_DIR / "05_turn6_braking_points.csv", index=False)
    return out_df


def plot_circuit_braking_map(
    res_df: pd.DataFrame,
    lap_number: int = 5,
    save_path: Optional[Path] = None
) -> Path:
    """Plot speed vs distance with overlaid braking zones across entire track."""
    mean_speed = res_df[res_df["LapNumber"] == lap_number].groupby("Distance_Meter")["Speed"].mean()
    brk_prob = res_df[res_df["LapNumber"] == lap_number].groupby("Distance_Meter")["IsBraking"].mean()

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2.5, 1]}, dpi=150)

    ax1.plot(mean_speed.index, mean_speed.values, color="#06b6d4", linewidth=1.8, label="Mean Speed (km/h)")
    # Highlight braking regions in red
    ax1.fill_between(mean_speed.index, 0, mean_speed.values, where=(brk_prob > 0.3),
                     color="#ef4444", alpha=0.35, label="Active Braking Zone")
    ax1.set_title(f"Circuit Velocity & Braking Zones (Lap {lap_number}, 2024 Abu Dhabi GP)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Speed (km/h)", fontsize=11, labelpad=10)
    ax1.set_ylim(40, 350)
    ax1.grid(alpha=0.2, color="#94a3b8")
    ax1.legend(loc="upper right", frameon=True, facecolor="#1e293b")

    ax2.fill_between(brk_prob.index, 0, brk_prob.values * 100, color="#ef4444", alpha=0.85)
    ax2.set_title("Fleet Braking Probability (%) along Distance Grid", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Circuit Distance (metres)", fontsize=11, labelpad=10)
    ax2.set_ylabel("% Braking", fontsize=10, labelpad=8)
    ax2.set_xlim(0, 5281)
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.2, color="#94a3b8")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "05_circuit_braking_map.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_turn6_braking_comparison(
    t6_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """Bar chart comparing Turn 6 braking points (latest to earliest brakers)."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Sort so latest braker is top
    df_plot = t6_df.sort_values("BrakingPoint_m", ascending=True)
    bars = ax.barh(df_plot["Driver"], df_plot["BrakingPoint_m"], color="#ef4444",
                   edgecolor="#ffffff", linewidth=0.6, alpha=0.85)

    ax.set_title("Turn 6 Heavy Braking Points (End of 1.2km Back Straight)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Braking Initiation Point along Circuit (metres)", fontsize=11, labelpad=10)
    # Zoom in to the braking zone
    min_d = df_plot["BrakingPoint_m"].min() - 20
    max_d = df_plot["BrakingPoint_m"].max() + 30
    ax.set_xlim(min_d, max_d)
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#94a3b8")

    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        w = bar.get_width()
        ax.annotate(f" {w:.0f}m ({row['EntrySpeed_kmh']:.0f} km/h)",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold", color="#ffffff")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "05_turn6_braking_comparison.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
