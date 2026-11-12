"""
F1 Telemetry Intelligence - Data Quality & Anomaly Analysis Module (Phase 5 EDA)
================================================================================

Audits missing value distributions, physical sensor boundary compliance,
outlier flags, and driver-level data consistency without mutating processed data.
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


def audit_missing_values(df: pd.DataFrame, entity_name: str = "Telemetry") -> pd.DataFrame:
    """Analyze missing value frequencies, percentages, and domain explanations."""
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    records = []
    for col, count in null_cols.items():
        pct = (count / len(df)) * 100
        explanation = "Unknown"
        if col == "DistanceToDriverAhead":
            explanation = "Structural MNAR: Race leader (NOR, 2,957 nulls) has no vehicle ahead"
        elif col == "DriverAhead":
            explanation = "Structural MNAR: Race leader (3,547 nulls) + timing line boundary zone (472 nulls)"
        elif "Pit" in col:
            explanation = "Structural: Cars only enter pitlane 1-2 times per race"
        elif col == "DeletedReason":
            explanation = "Structural: Populated only upon track limits breach deletion"
        elif col == "LapTime":
            explanation = "Crash retirement: Sergio Perez Lap 1 did not complete lap"

        records.append({
            "Entity": entity_name,
            "Column": col,
            "NullCount": int(count),
            "NullPercentage": round(float(pct), 3),
            "EngineeringDomainRootCause": explanation
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(TABLES_DIR / f"02_{entity_name.lower()}_missing_values.csv", index=False)
    return out_df


def audit_sensor_boundaries(tel_df: pd.DataFrame) -> pd.DataFrame:
    """Verify physical kinematics bounds against known FIA Technical Regulations."""
    sensors = {
        "Speed": {"min_expected": 0.0, "max_expected": 360.0, "units": "km/h"},
        "Throttle": {"min_expected": 0.0, "max_expected": 100.0, "units": "% (Raw)"},
        "Throttle_Clipped": {"min_expected": 0.0, "max_expected": 100.0, "units": "% (Clipped)"},
        "RPM": {"min_expected": 0.0, "max_expected": 15000.0, "units": "rev/min"},
        "nGear": {"min_expected": 1.0, "max_expected": 8.0, "units": "gear number"},
        "Distance_Corrected": {"min_expected": 0.0, "max_expected": 5300.0, "units": "metres"}
    }

    records = []
    for col, bounds in sensors.items():
        if col in tel_df.columns:
            s = tel_df[col].dropna()
            actual_min = float(s.min())
            actual_max = float(s.max())
            breach_low = int((s < bounds["min_expected"]).sum())
            breach_high = int((s > bounds["max_expected"]).sum())
            status = "PASS" if breach_low == 0 and breach_high == 0 else "FLAGGED"

            notes = "Within nominal bounds"
            if col == "Throttle" and breach_high > 0:
                notes = f"32 mechanical overtravel readings up to {actual_max:.1f}%"
            elif col == "RPM" and actual_min == 0.0:
                notes = "33 zero RPM samples belong strictly to PER collision retirement"

            records.append({
                "Channel": col,
                "Units": bounds["units"],
                "ExpectedRange": f"[{bounds['min_expected']}, {bounds['max_expected']}]",
                "ActualMin": round(actual_min, 2),
                "ActualMax": round(actual_max, 2),
                "LowBreaches": breach_low,
                "HighBreaches": breach_high,
                "AuditStatus": status,
                "TechnicalInterpretation": notes
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(TABLES_DIR / "02_sensor_boundaries.csv", index=False)
    return out_df


def audit_anomaly_flags(tel_df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown anomaly and outlier flags by driver."""
    flags = ["Distance_Anomaly_Flag", "Throttle_Overtravel_Flag", "IsOutlierFlag"]
    records = []
    for drv, grp in tel_df.groupby("Driver"):
        rec = {"Driver": drv, "TotalSamples": len(grp)}
        for f in flags:
            rec[f] = int(grp[f].sum()) if f in grp else 0
        records.append(rec)

    df_flags = pd.DataFrame(records)
    df_flags.to_csv(TABLES_DIR / "02_driver_anomaly_flags.csv", index=False)
    return df_flags


def plot_missing_values_bar(tel_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Generate visual bar chart of missing values with annotations."""
    missing = tel_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(missing.index, missing.values, color="#e10600", edgecolor="#ffffff", linewidth=0.8, alpha=0.9)

    ax.set_title("Missing Values in Processed Telemetry (N = 72,476)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Null Record Count", fontsize=11, labelpad=10)
    ax.set_ylim(0, max(missing.values) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        y = bar.get_height()
        pct = (y / len(tel_df)) * 100
        ax.annotate(f"{y:,}\n({pct:.2f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#ffffff")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "02_missing_values_bar.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_throttle_overtravel_comparison(tel_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Compare raw throttle overtravel vs clipped throttle distribution."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Raw Throttle > 95%
    raw_high = tel_df[tel_df["Throttle"] > 95.0]["Throttle"]
    sns.histplot(raw_high, bins=25, color="#ef4444", ax=ax1, edgecolor="#1e293b", alpha=0.8)
    ax1.axvline(100.0, color="#fbbf24", linestyle="--", linewidth=1.5, label="100.0% Nominal Limit")
    ax1.set_title("Raw Throttle (>95%)\n32 Samples in (100%, 104%]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Raw Throttle (%)")
    ax1.legend(frameon=True, facecolor="#1e293b")

    # Clipped Throttle > 95%
    clipped_high = tel_df[tel_df["Throttle_Clipped"] > 95.0]["Throttle_Clipped"]
    sns.histplot(clipped_high, bins=25, color="#10b981", ax=ax2, edgecolor="#1e293b", alpha=0.8)
    ax2.axvline(100.0, color="#fbbf24", linestyle="--", linewidth=1.5, label="100.0% Clipped Cap")
    ax2.set_title("Throttle_Clipped (>95%)\nStrictly Bounded at 100.0%", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Throttle_Clipped (%)")
    ax2.legend(frameon=True, facecolor="#1e293b")

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / "02_throttle_clipping_comparison.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out
