"""
F1 Telemetry Intelligence - Correlation & Multicollinearity Module (Phase 5 EDA)
================================================================================

Computes Pearson and Spearman rank correlation matrices, evaluates linear and
non-linear dependencies among kinematic sensors, and addresses correlation vs causation.
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


def compute_telemetry_correlation_matrix(
    tel_df: pd.DataFrame,
    method: str = "pearson",
    only_flying: bool = True
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix across numerical kinematic features.
    Filters to COMPLETE_FLYING_LAP to prevent distortion from VSC/safety car regimes.
    """
    df = tel_df.copy()
    if only_flying and "LapClassification" in df.columns:
        df = df[df["LapClassification"] == "COMPLETE_FLYING_LAP"]

    numeric_cols = [
        c for c in [
            "Speed", "Throttle_Clipped", "Brake_Int", "RPM", "nGear",
            "AccelerationX", "AccelerationY", "AccelerationZ", "Distance_Corrected"
        ] if c in df.columns
    ]

    corr = df[numeric_cols].corr(method=method).round(3)
    corr.to_csv(TABLES_DIR / f"10_{method}_correlation_matrix.csv")
    return corr


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    method_name: str = "Pearson",
    save_path: Optional[Path] = None
) -> Path:
    """Plot correlation matrix heatmap with dark motorsport styling."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.6,
                cbar_kws={"shrink": 0.8, "label": f"{method_name} Correlation ($r$)"}, ax=ax)

    ax.set_title(f"Kinematic Telemetry {method_name} Correlation Matrix (Clean Flying Laps)",
                 fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    out = save_path or (FIGURES_DIR / f"10_{method_name.lower()}_correlation_heatmap.png")
    plt.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return out


def plot_key_signal_pairplot(
    tel_df: pd.DataFrame,
    sample_size: int = 5000,
    save_path: Optional[Path] = None
) -> Path:
    """Generate pairplot of core control signals (Speed, Throttle, RPM, Gear)."""
    df_clean = tel_df[tel_df["LapClassification"] == "COMPLETE_FLYING_LAP"]
    if len(df_clean) > sample_size:
        df_clean = df_clean.sample(sample_size, random_state=42)

    cols = ["Speed", "Throttle_Clipped", "RPM", "nGear"]
    plt.style.use("dark_background")

    g = sns.pairplot(df_clean[cols], diag_kind="kde",
                     plot_kws={"alpha": 0.2, "s": 10, "color": "#06b6d4"},
                     diag_kws={"color": "#10b981", "fill": True})
    g.fig.suptitle("Pairwise Relationships Among Kinematic Variables (N=5,000 Flying Samples)",
                   y=1.02, fontsize=13, fontweight="bold")

    out = save_path or (FIGURES_DIR / "10_signal_pairplot.png")
    g.savefig(out, facecolor="#0f172a", edgecolor="none")
    plt.close(g.fig)
    return out
