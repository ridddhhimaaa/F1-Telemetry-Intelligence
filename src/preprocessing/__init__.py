"""
F1 Telemetry Intelligence - Preprocessing Package (Phase 4)
===========================================================

Provides validation, missing-value handling, telemetry cleaning and normalization,
lap classification, non-destructive outlier detection, and spatial distance resampling.
"""

from src.preprocessing.validate_data import validate_datasets, DataValidator
from src.preprocessing.handle_missing_values import handle_missing_values, MissingValueHandler
from src.preprocessing.clean_telemetry import clean_telemetry_signals, TelemetryCleaner
from src.preprocessing.classify_laps import classify_all_laps, LapClassifier
from src.preprocessing.detect_outliers import detect_telemetry_outliers, OutlierDetector
from src.preprocessing.resample_telemetry import resample_telemetry_spatial, SpatialResampler
from src.preprocessing.preprocess_pipeline import run_preprocessing_pipeline

__all__ = [
    "validate_datasets",
    "DataValidator",
    "handle_missing_values",
    "MissingValueHandler",
    "clean_telemetry_signals",
    "TelemetryCleaner",
    "classify_all_laps",
    "LapClassifier",
    "detect_outliers",
    "OutlierDetector",
    "resample_telemetry_spatial",
    "SpatialResampler",
    "run_preprocessing_pipeline",
]
