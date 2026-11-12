"""
Script to generate notebooks/eda/12_eda_validation_audit.ipynb
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

# Title cell
nb.cells.append(nbf.v4.new_markdown_cell('''# 12: EDA Scientific Validation & Analytical Audit
**Project:** F1 Telemetry Intelligence  
**Grand Prix:** 2024 Abu Dhabi Grand Prix (Yas Marina Circuit)  
**Phase:** Phase 5.1 – Scientific Validation and Analytical Audit  

---

## 1. Executive Summary & Audit Scope
This audit performs rigorous empirical validation of exploratory findings from Phase 5 EDA:
1. **DRS Analytical Validation:** Deconstruct the reported raw speed difference between DRS active (298.69 km/h) and inactive (197.77 km/h) to isolate the true physical aerodynamic drag reduction (+12.75 km/h) from track-location confounding (ecological fallacy of comparing straights to hairpins and braking zones).
2. **Kinematic Correlation & Dynamic Validation:** Audit intra-lap vs pooled correlations, spatial distance-windowed dynamics, aerodynamic drag saturation at terminal velocity, and temporal autocorrelation persistence.
3. **Machine Learning Data Audit:** Audit target leakage (tautology of `Brake_Int` for `IsBraking`), spatial coordinate circuit memorization ($R^2 = 0.902$ lookup table vs vehicle dynamics), deterministic transmission gear ratios (RPM/Speed physics), sample size power for `LapTimeSeconds` ($N=38$), and prove why random 80/20 train/test splitting causes catastrophic optimism bias (96.1% immediate neighbor leakage).
4. **Statistical Significance Testing:** Provide peer-review-grade hypothesis tests (Welch's t-test, Mann-Whitney U test, Cohen's d, 95% confidence intervals).
'''))

# Setup cell
nb.cells.append(nbf.v4.new_markdown_cell('## 2. Setup & Environment Configuration'))
nb.cells.append(nbf.v4.new_code_cell('''import os
import sys
import tempfile
from pathlib import Path

# Headless matplotlib
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

# Configure Project Root
PROJECT_ROOT = Path.cwd().parents[1] if Path.cwd().name == "eda" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project Root: {PROJECT_ROOT}")

from src.analysis.data_loader import (
    load_all_datasets,
    filter_clean_flying_laps
)
from src.analysis.eda_validation import (
    validate_drs_effect,
    compute_drs_driver_breakdown,
    validate_kinematic_correlations,
    compute_distance_windowed_correlations,
    compute_temporal_autocorrelation,
    audit_machine_learning_targets,
    generate_all_validation_figures,
    generate_validation_html_report,
    VALIDATION_DIR,
    FIGURES_DIR
)
'''))

# Data loading cell
nb.cells.append(nbf.v4.new_markdown_cell('## 3. Load Processed Telemetry and Race Laps'))
nb.cells.append(nbf.v4.new_code_cell('''tel_df, laps_df, res_df = load_all_datasets()
flying_tel = filter_clean_flying_laps(tel_df)

print(f"Total Telemetry Samples: {len(tel_df):,}")
print(f"Clean Flying Lap Telemetry Samples: {len(flying_tel):,}")
print(f"Unique Drivers: {flying_tel['Driver'].nunique()}")
print(f"Race Laps: {len(laps_df):,}")
'''))

# Task 1 cell
nb.cells.append(nbf.v4.new_markdown_cell('''## 4. Task 1: DRS Analytical Validation & Confounding Decomposition

### Empirical Finding:
- **Naive Pooled Full-Lap Delta:** DRS Active (298.69 km/h) vs Inactive (197.77 km/h) = **+100.92 km/h**
- **Controlled Straightaway Delta:** DRS Active (298.31 km/h) vs Inactive (285.57 km/h) = **+12.75 km/h**
- **Confounding Breakdown:** **87.4% (88.17 km/h)** of the naive difference is an ecological fallacy caused by track geometry (hairpins, chicanes, braking zones). The isolated aerodynamic drag reduction benefit is **+12.75 km/h** (95% CI: [11.29, 14.21] km/h, p = 5.97e-64).
'''))
nb.cells.append(nbf.v4.new_code_cell('''drs_summary = validate_drs_effect(tel_df)
df_comp = pd.read_csv(VALIDATION_DIR / "drs_controlled_comparison.csv")
display(df_comp)
'''))

nb.cells.append(nbf.v4.new_markdown_cell('### Per-Driver DRS Speed Breakdown in Straightaway Activation Zones'))
nb.cells.append(nbf.v4.new_code_cell('''df_drv = compute_drs_driver_breakdown(tel_df)
display(df_drv)
'''))

# Task 2 cell
nb.cells.append(nbf.v4.new_markdown_cell('''## 5. Task 2: Kinematic Correlation & Dynamic Aerodynamic Drag Validation

### Empirical Finding:
- Pooled correlation r(Speed, Throttle) = 0.580, intra-lap mean r = 0.574 +/- 0.043.
- In distance windows, correlation collapses along high-speed straights (r = -0.15 to +0.05) because throttle is pegged at 100% while acceleration saturates due to quadratic aerodynamic drag.
- In corner exit acceleration zones, r(Speed, Throttle) reaches +0.75 to +0.95.
'''))
nb.cells.append(nbf.v4.new_code_cell('''corr_summary = validate_kinematic_correlations(tel_df)
df_corr_sum = pd.read_csv(VALIDATION_DIR / "kinematic_correlation_summary.csv")
display(df_corr_sum)
'''))

nb.cells.append(nbf.v4.new_markdown_cell('### Spatial Distance-Windowed Correlation & Dynamic Regimes (250m Bins)'))
nb.cells.append(nbf.v4.new_code_cell('''df_win = compute_distance_windowed_correlations(tel_df, window_size_m=250.0)
display(df_win[['Midpoint_m', 'Mean_Speed_kmh', 'Mean_Throttle_pct', 'r_Speed_Throttle', 'DynamicRegime']].head(10))
'''))

# Task 3 cell
nb.cells.append(nbf.v4.new_markdown_cell('''## 6. Task 3: Machine Learning Data Readiness & Leakage Audit

### Critical Leakage Identified:
1. **Target A (`IsBraking`):** `Brake_Int` is a direct integer cast (r = 1.000) -> 100% target tautology. Must be hard-dropped from feature matrix X.
2. **Target B (`Speed`):** Raw spatial coordinates (X, Y, d) construct a 10m spatial lookup table explaining **90.2% variance** ($R^2 = 0.902$) by memorizing the track layout rather than vehicle dynamics.
3. **Target C (`nGear`):** Ratio RPM/Speed uniquely maps to discrete physical transmission gear ratios.
4. **Target D (`LapTimeSeconds`):** Current 5-lap ingestion contains only $N = 38$ clean flying laps across 19 drivers (underpowered for high-dimensional models).
5. **Train/Test Splitting:** Random 80/20 row splitting places **96.12% of test samples** within +/- 0.125s (adjacent row) of a training sample. With lag-1 autocorrelation r = 0.997, random splitting causes catastrophic optimism bias. **`GroupKFold(by=['Driver', 'LapNumber'])` is strictly mandated.**
'''))
nb.cells.append(nbf.v4.new_code_cell('''ml_audit = audit_machine_learning_targets(tel_df, laps_df)
df_leak = pd.read_csv(VALIDATION_DIR / "ml_target_leakage_audit.csv")
display(df_leak)
'''))

nb.cells.append(nbf.v4.new_markdown_cell('### Dataset Splitting Strategies Audit'))
nb.cells.append(nbf.v4.new_code_cell('''df_split = pd.read_csv(VALIDATION_DIR / "ml_splitting_strategy_audit.csv")
display(df_split)
'''))

# Visualizations cell
nb.cells.append(nbf.v4.new_markdown_cell('## 7. Audit Visualizations & Verification Figures'))
nb.cells.append(nbf.v4.new_code_cell('''figures = generate_all_validation_figures(tel_df, laps_df)
for fig_path in figures:
    print(f"Generated: {fig_path.name}")
    display(Image(filename=str(fig_path)))
'''))

# Conclusion cell
nb.cells.append(nbf.v4.new_markdown_cell('''## 8. Synthesis & Mandatory Phase 6 Architectural Rules
1. **DRS Representation:** Reject naive speed delta (+100.92 km/h). When engineering features for Phase 6, interact DRS with track position or isolated straightaway intervals.
2. **Kinematic Feature Engineering:** Model throttle as dynamic acceleration intent rather than a linear predictor of speed across drag-saturated regimes.
3. **Feature Matrix Hygiene:** Hard-drop `Brake_Int` and `Brake` from any model predicting braking points.
4. **Spatial Coordinates Policy:** Prohibit raw $(X, Y, Z, \\text{Distance})$ in speed prediction models to prevent circuit memorization. Use curvature, delta-throttle, and historical lag tensors.
5. **Cross-Validation Standard:** Enforce `GroupKFold(by=['Driver', 'LapNumber'])` across all sample-level modeling.
'''))

out_nb_path = Path('notebooks/eda/12_eda_validation_audit.ipynb')
out_nb_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully created {out_nb_path} with {len(nb.cells)} cells.")
