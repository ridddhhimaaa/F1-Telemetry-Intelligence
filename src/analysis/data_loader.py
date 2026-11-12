"""
F1 Telemetry Intelligence - Data Loading & Filtering Module (Phase 5 EDA)
========================================================================

Provides robust, cached data loading functions for processed telemetry,
race laps, and resampled spatial grids with schema validation and domain filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("f1_telemetry.analysis.data_loader")

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "abu_dhabi_2024"
DEFAULT_TEL_PATH = PROCESSED_DIR / "processed_telemetry.parquet"
DEFAULT_LAPS_PATH = PROCESSED_DIR / "processed_race_laps.parquet"
DEFAULT_RES_PATH = PROCESSED_DIR / "resampled_telemetry.parquet"

# In-memory caches for interactive notebook performance
_CACHE: Dict[str, pd.DataFrame] = {}


def load_processed_telemetry(
    file_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Load feature-engineered, validated telemetry dataset.
    Default: data/processed/abu_dhabi_2024/processed_telemetry.parquet
    """
    path = Path(file_path) if file_path else DEFAULT_TEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Processed telemetry dataset not found at: {path}")

    cache_key = f"tel_{path}"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    logger.info("Loading processed telemetry: %s", path)
    df = pd.read_parquet(path)

    # Ensure categorical / temporal types
    if "Time" in df.columns and not pd.api.types.is_timedelta64_dtype(df["Time"]):
        df["Time"] = pd.to_timedelta(df["Time"])

    if use_cache:
        _CACHE[cache_key] = df

    return df.copy()


def load_processed_laps(
    file_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Load feature-engineered race-lap dataset.
    Default: data/processed/abu_dhabi_2024/processed_race_laps.parquet
    """
    path = Path(file_path) if file_path else DEFAULT_LAPS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Processed race-lap dataset not found at: {path}")

    cache_key = f"laps_{path}"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    logger.info("Loading processed race laps: %s", path)
    df = pd.read_parquet(path)

    if use_cache:
        _CACHE[cache_key] = df

    return df.copy()


def load_resampled_telemetry(
    file_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Load uniform 1.0-metre spatial distance resampled telemetry dataset.
    Default: data/processed/abu_dhabi_2024/resampled_telemetry.parquet
    """
    path = Path(file_path) if file_path else DEFAULT_RES_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Resampled telemetry dataset not found at: {path}")

    cache_key = f"res_{path}"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    logger.info("Loading resampled spatial telemetry: %s", path)
    df = pd.read_parquet(path)

    if use_cache:
        _CACHE[cache_key] = df

    return df.copy()


def load_all_datasets(
    processed_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three core analytical datasets in one unified call.
    Returns:
        (processed_telemetry, processed_laps, resampled_telemetry)
    """
    pdir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    tel_df = load_processed_telemetry(pdir / "processed_telemetry.parquet", use_cache=use_cache)
    laps_df = load_processed_laps(pdir / "processed_race_laps.parquet", use_cache=use_cache)
    res_df = load_resampled_telemetry(pdir / "resampled_telemetry.parquet", use_cache=use_cache)
    return tel_df, laps_df, res_df


def filter_clean_flying_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset strictly to valid, green-flag flying laps (TrackStatus 1, Laps 4 & 5).
    Excludes standing starts, VSC-affected laps, pit laps, and collision retirements.
    """
    if "LapClassification" in df.columns:
        return df[df["LapClassification"] == "COMPLETE_FLYING_LAP"].copy()
    elif "TrackStatus" in df.columns:
        return df[(df["TrackStatus"] == "1") & (df["LapNumber"] >= 4)].copy()
    return df.copy()


def filter_by_driver(df: pd.DataFrame, driver_code: str) -> pd.DataFrame:
    """Filter records for a specific driver code (e.g. 'VER', 'NOR', 'HAM')."""
    if "Driver" not in df.columns:
        raise ValueError("DataFrame does not contain a 'Driver' column.")
    return df[df["Driver"] == driver_code.upper()].copy()


def filter_by_lap(df: pd.DataFrame, lap_number: int) -> pd.DataFrame:
    """Filter records for a specific lap number (1 to 5)."""
    if "LapNumber" not in df.columns:
        raise ValueError("DataFrame does not contain a 'LapNumber' column.")
    return df[df["LapNumber"] == int(lap_number)].copy()


def clear_cache() -> None:
    """Clear in-memory DataFrame caches."""
    _CACHE.clear()
    logger.info("In-memory dataset cache cleared.")
