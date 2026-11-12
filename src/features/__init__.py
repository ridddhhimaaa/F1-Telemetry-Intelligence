"""
F1 Telemetry Intelligence - Advanced Feature Engineering Package (Phase 6)
========================================================================

Modular, scientifically validated feature engineering pipeline for vehicle dynamics,
track geometry, throttle dynamics, braking behavior, cornering performance, transmission/engine,
sector summaries, and lap aggregates.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports" / "feature_engineering"

__all__ = [
    "PROJECT_ROOT",
    "FEATURES_DATA_DIR",
    "REPORTS_DIR",
]
