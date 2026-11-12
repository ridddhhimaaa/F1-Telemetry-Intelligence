"""
F1 Telemetry Intelligence - Lap Comparison & Delta Analytics Module (Phase 5 EDA)
================================================================================

Computes multi-channel telemetry deltas (Delta Speed, Delta Throttle, Time Delta)
between drivers on identical spatially aligned flying laps using resampled grids.
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


def compute_driver_lap_delta(
    res_df: pd.DataFrame,
    driver_ref: str = "NOR",
    driver_comp: str = "VER",
    lap_number: int = 5
) -> pd.DataFrame:
    """
    Compute synchronized spatial telemetry delta trace between two drivers.
    Positive Delta Speed means reference driver is faster.
    """
    ref = res_df[(res_df["Driver"] == driver_ref.upper()) & (res_df["LapNumber"] == lap_number)].set_index("Distance_Meter")
    comp = res_df[(res_df["Driver"] == driver_comp.upper()) & (res_df["LapNumber"] == lap_number)].set_index("Distance_Meter")

    common_dist = ref.index.intersection(comp.index)
    if len(common_dist) == 0:
        raise ValueError(f"No common spatial coordinates between {driver_ref} and {driver_comp} on Lap {lap_number}.")

    ref_common = ref.loc[common_dist]
    comp_common = comp.loc[common_dist]

    # Speed Delta (km/h)
    delta_v = ref_common["Speed"] - comp_common["Speed"]

    # Throttle Delta (%)
    delta_thr = ref_common["Throttle_Clipped"] - comp_common["Throttle_Clipped"]

    # Gear Delta
    delta_gear = ref_common["nGear"] - comp_common["nGear"]

    # Cumulative Estimated Time Delta (seconds)
    # dt = ds / v (converting km/h to m/s: v / 3.6)
    v_ref_ms = np.maximum(ref_common["Speed"].values / 3.6, 5.0)
    v_comp_ms = np.maximum(comp_common["Speed"].values / 3.6, 5.0)
    dt_ref = 1.0 / v_ref_ms
    dt_comp = 1.0 / v_comp_ms
    cum_time_delta = np.cumsum(dt_comp - dt_ref)  # Positive means ref driver is ahead in time!

    delta_df = pd.DataFrame({
        "Distance_Meter": common_dist,
        "Driver_Ref": driver_ref,
        "Driver_Comp": driver_comp,
        "Speed_Ref": ref_common["Speed"].values,
        "Speed_Comp": comp_common["Speed"].values,
        "Delta_Speed_kmh": delta_v.values,
        "Delta_Throttle_pct": delta_thr.values,
        "Delta_Gear": delta_gear.values,
        "Cumulative_Time_Delta_s": cum_time_delta
    })

    out_name = f"08_delta_{driver_ref}_vs_{driver_comp}_lap{lap_number}.csv"
    delta_df.to_csv(TABLES_DIR / out_name, index=False)
    return delta_df


def plot_head_to_head_telemetry(
    delta_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Path:
    """
    Generate professional multi-panel telemetry comparison chart:
      Panel 1: Speed Comparison & Delta Speed
      Panel 2: Cumulative Time Delta Trace
      Panel 3: Throttle Comparison
      Panel 4: Braking Comparison
      Panel 5: Gear Comparison
    """
    ref_name = delta_df["Driver_Ref"].iloc[0]
    comp_name = delta_df["Driver_Comp"].iloc[0]
    d = delta_df["Distance_Meter"]

    plt.style.use("dark_background")
    fig, (ax_sp, ax_dt, ax_th, ax_gr) = plt.subplots(
        4, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.8, 1.2]}, dpi=150
    )

    # 1. Speed Trace
    ax_sp.plot(d, delta_df["Speed_Ref"], color="#ff8000", linewidth=1.5, label=f"{ref_name} Speed (km/h)")
    ax_sp.plot(d, delta_df["Speed_Comp"], color="#3671c6", linewidth=1.5, label=f"{comp_name} Speed (km/h)")
    ax_sp.set_title(f"Head-to-Head Spatial Telemetry: {ref_name} vs {comp_name} (Lap 5, Yas Marina Circuit)",
                    fontsize=13, fontweight="bold", pad=12)
    ax_sp.set_ylabel("Speed (km/h)", fontsize=10)
    ax_sp.grid(alpha=0.2, color="#94a3b8")
    ax_sp.legend(loc="upper right", frameon=True, facecolor="#1e293b")

    # 2. Cumulative Time Delta Trace
    ax_dt.plot(d, delta_df["Cumulative_Time_Delta_s"], color="#10b981", linewidth=1.5)
    ax_dt.axhline(0, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_dt.set_ylabel(f"Time Gain (s)\n+{ref_name} Ahead", fontsize=9)
    ax_dt.grid(alpha=0.2, color="#94a3b8")

    # 3. Throttle Comparison
    ax_th.plot(d, delta_df["Delta_Throttle_pct"], color="#06b6d4", linewidth=1.2)
    ax_th.axhline(0, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_th.set_ylabel("Delta Throttle (%)", fontsize=10)
    ax_th.grid(alpha=0.2, color="#94a3b8")

    # 4. Gear Comparison
    ax_gr.plot(d, delta_df["Delta_Gear"], color="#a855f7", linewidth=1.2)
    ax_gr.axhline(0, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_gr.set_title("Gear Selection Delta", fontsize=10, pad=4)
    ax_gr.set_xlabel("Circuit Distance along Track (metres)", fontsize=11, labelpad=8)
    ax_gr.set_ylabel("Delta Gear", fontsize=10)
    ax_gr.set_xlim(0, 5281)
    ax_gr.grid(alpha=0.2, color="#94a3b8")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / f"08_head_to_head_{ref_name}_vs_{comp_name}.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
