"""
F1 Telemetry Intelligence - Dataset Overview Module (Phase 5 EDA)
================================================================

Computes comprehensive structural inventories, schema metrics, driver rosters,
lap classification distributions, and temporal sampling frequencies.
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
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_dataset_dimensions(
    tel_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    res_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute dimensional and memory metrics across all three processed entities."""
    records = [
        {
            "Dataset Entity": "Processed Telemetry",
            "Rows": len(tel_df),
            "Columns": len(tel_df.columns),
            "Memory (MB)": round(tel_df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "Unique Drivers": tel_df["Driver"].nunique() if "Driver" in tel_df else 0,
            "Unique Laps": tel_df["LapNumber"].nunique() if "LapNumber" in tel_df else 0,
            "Primary Granularity": "Time-domain intra-lap samples (~10 Hz)"
        },
        {
            "Dataset Entity": "Processed Race Laps",
            "Rows": len(laps_df),
            "Columns": len(laps_df.columns),
            "Memory (MB)": round(laps_df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "Unique Drivers": laps_df["Driver"].nunique() if "Driver" in laps_df else 0,
            "Unique Laps": laps_df["LapNumber"].nunique() if "LapNumber" in laps_df else 0,
            "Primary Granularity": "Lap-level summary records (58 laps total)"
        },
        {
            "Dataset Entity": "Resampled Spatial Telemetry",
            "Rows": len(res_df),
            "Columns": len(res_df.columns),
            "Memory (MB)": round(res_df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "Unique Drivers": res_df["Driver"].nunique() if "Driver" in res_df else 0,
            "Unique Laps": res_df["LapNumber"].nunique() if "LapNumber" in res_df else 0,
            "Primary Granularity": "Space-domain synchronized slices (1.0m grid)"
        }
    ]
    df_dim = pd.DataFrame(records)
    df_dim.to_csv(TABLES_DIR / "01_dataset_dimensions.csv", index=False)
    return df_dim


def compute_lap_classification_summary(tel_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize driver-lap sequences by their FIA tactical classification."""
    grouped = tel_df.groupby(["Driver", "LapNumber", "LapClassification"]).size().reset_index(name="SampleCount")
    summary = grouped["LapClassification"].value_counts().reset_index()
    summary.columns = ["Classification", "DriverLapSequences"]
    summary["Percentage"] = (summary["DriverLapSequences"] / len(grouped) * 100).round(2)

    total_samples = tel_df.groupby("LapClassification").size().to_dict()
    summary["TotalTelemetrySamples"] = summary["Classification"].map(total_samples)

    summary.to_csv(TABLES_DIR / "01_lap_classifications.csv", index=False)
    return summary


def compute_sampling_frequency_stats(tel_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute empirical time-delta statistics across telemetry observations."""
    deltas = []
    for (_, _), grp in tel_df.groupby(["Driver", "LapNumber"]):
        if "Time" in grp.columns:
            t = grp["Time"].values
            dt = np.diff(t).astype("timedelta64[ns]").astype(float) / 1e9  # seconds
            deltas.extend(dt[dt > 0])

    deltas = np.array(deltas)
    stats = {
        "Mean_DeltaT_s": float(np.mean(deltas)),
        "Median_DeltaT_s": float(np.median(deltas)),
        "Std_DeltaT_s": float(np.std(deltas)),
        "Min_DeltaT_s": float(np.min(deltas)),
        "Max_DeltaT_s": float(np.max(deltas)),
        "Nominal_Hz": float(1.0 / np.median(deltas)) if len(deltas) > 0 else 0.0,
        "Total_Sample_Transitions": len(deltas)
    }
    pd.DataFrame([stats]).to_csv(TABLES_DIR / "01_sampling_frequency.csv", index=False)
    return stats


def plot_lap_classification_distribution(tel_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Generate professional chart of lap taxonomy distributions."""
    grouped = tel_df.groupby(["Driver", "LapNumber", "LapClassification"]).size().reset_index()
    counts = grouped["LapClassification"].value_counts()

    palette = {
        "COMPLETE_FLYING_LAP": "#10b981",
        "VSC_AFFECTED_LAP": "#f59e0b",
        "START_LAP": "#3b82f6",
        "DNF_OR_RETIRED": "#ef4444"
    }
    colors = [palette.get(k, "#64748b") for k in counts.index]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="#ffffff", linewidth=0.8, alpha=0.9)

    ax.set_title("Distribution of Ingested Driver-Lap Sequences (96 Total)", fontsize=13, fontweight="bold", pad=14, color="#ffffff")
    ax.set_xlabel("Number of Driver-Lap Sequences", fontsize=11, labelpad=10, color="#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        w = bar.get_width()
        pct = (w / 96.0) * 100
        ax.annotate(f" {w} ({pct:.1f}%)",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, fontweight="bold", color="#ffffff")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "01_lap_classification_distribution.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_sampling_interval_distribution(tel_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot histogram and KDE of sampling time intervals."""
    deltas = []
    for (_, _), grp in tel_df.groupby(["Driver", "LapNumber"]):
        if "Time" in grp.columns:
            t = grp["Time"].values
            dt = np.diff(t).astype("timedelta64[ns]").astype(float) / 1e9
            deltas.extend(dt[(dt > 0) & (dt < 0.5)])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.histplot(deltas, bins=60, kde=True, color="#06b6d4", ax=ax, edgecolor="#1e293b", alpha=0.75)

    ax.axvline(np.median(deltas), color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Median $\\Delta t$: {np.median(deltas):.3f}s (~{1.0/np.median(deltas):.1f} Hz)")
    ax.set_title("Empirical Telemetry Sampling Time Delta Distribution ($\\Delta t < 0.5s$)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Time Delta between Successive Observations (seconds)", fontsize=11, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=11, labelpad=10)
    ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")
    ax.grid(alpha=0.2, color="#94a3b8")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "01_sampling_interval_distribution.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
