"""
F1 Telemetry Intelligence - Spatial Telemetry Analytics Module (Phase 5 EDA)
============================================================================

Maps telemetry onto spatial 2D track geometry (X, Y coordinates), segments
the 16-turn Yas Marina circuit into sectors and corners, and analyzes spatial profiles.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "eda_report" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "eda_report" / "tables"

# Yas Marina Circuit Key Landmarks (Post-2021 Reconfiguration, 5,281m)
CIRCUIT_SECTORS = {
    "Sector 1": {"start_m": 0.0, "end_m": 1470.0, "desc": "Main Straight, Turn 1, Fast Sweeps, Turn 5 Hairpin"},
    "Sector 2": {"start_m": 1470.0, "end_m": 3250.0, "desc": "1.2km Back Straight, Turn 6/7 Chicane, Second Straight, Turn 9 Bank"},
    "Sector 3": {"start_m": 3250.0, "end_m": 5281.0, "desc": "Marina & W Hotel Complex, Turns 12-15 Tech Chicanes, Turn 16"}
}


def compute_sector_performance_summary(
    res_df: pd.DataFrame,
    lap_number: int = 5
) -> pd.DataFrame:
    """Compute average speed, max speed, and full throttle percentage per sector."""
    lap5 = res_df[res_df["LapNumber"] == lap_number].copy()
    if lap5.empty:
        lap5 = res_df[res_df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    records = []
    for s_name, bounds in CIRCUIT_SECTORS.items():
        sub = lap5[(lap5["Distance_Meter"] >= bounds["start_m"]) & (lap5["Distance_Meter"] < bounds["end_m"])]
        records.append({
            "Sector": s_name,
            "DistanceRange_m": f"{bounds['start_m']:.0f}m - {bounds['end_m']:.0f}m",
            "MeanSpeed_kmh": round(float(sub["Speed"].mean()), 1),
            "MaxSpeed_kmh": round(float(sub["Speed"].max()), 1),
            "MinSpeed_kmh": round(float(sub["Speed"].min()), 1),
            "FullThrottle_pct": round(float((sub["Throttle_Clipped"] >= 98.0).mean() * 100), 1),
            "Braking_pct": round(float((sub["IsBraking"] == 1).mean() * 100), 1),
            "Description": bounds["desc"]
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(TABLES_DIR / "09_sector_performance_summary.csv", index=False)
    return out_df


def plot_2d_circuit_speed_map(
    res_df: pd.DataFrame,
    driver: str = "NOR",
    lap_number: int = 5,
    save_path: Optional[Path] = None
) -> Path:
    """
    Renders 2D GPS track layout colored by vehicle speed using (X, Y) telemetry.
    """
    lap_data = res_df[(res_df["Driver"] == driver.upper()) & (res_df["LapNumber"] == lap_number)]
    if lap_data.empty:
        lap_data = res_df[res_df["Driver"] == driver.upper()].iloc[:5281]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    scatter = ax.scatter(lap_data["X"], lap_data["Y"], c=lap_data["Speed"],
                         cmap="plasma", s=18, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Speed (km/h)", fontsize=11, labelpad=8)

    ax.set_title(f"2D Track Speed Heatmap: Yas Marina Circuit ({driver}, Lap {lap_number})",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Track Coordinate X", fontsize=10)
    ax.set_ylabel("Track Coordinate Y", fontsize=10)
    ax.axis("equal")
    ax.grid(alpha=0.15, color="#94a3b8")

    # Annotate key corners
    ax.annotate("Turn 6/7 Heavy Braking", xy=(lap_data["X"].iloc[2000], lap_data["Y"].iloc[2000]),
                xytext=(-1000, 500), textcoords="offset points",
                arrowprops=dict(facecolor="#ef4444", shrink=0.05, width=1, headwidth=5),
                fontsize=9, fontweight="bold", color="#ef4444")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / f"09_circuit_2d_speed_map_{driver}.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
