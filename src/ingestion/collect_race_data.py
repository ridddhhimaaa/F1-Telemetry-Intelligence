
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tif1


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

YEAR = 2024
EVENT = "Abu Dhabi"
SESSION_NAME = "Race"

# Initial test limits. Increase after successful validation.
MAX_DRIVERS = None
MAX_LAPS_PER_DRIVER = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
LOG_DIR = PROJECT_ROOT / "logs"

for directory in [RAW_DIR, METADATA_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingestion.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_name(value):
    """Convert event names into filesystem-safe names."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def save_dataset(df, filename):
    """Save a dataset in CSV and Parquet formats."""

    csv_path = RAW_DIR / f"{filename}.csv"
    parquet_path = RAW_DIR / f"{filename}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved Parquet: %s", parquet_path)


# --------------------------------------------------
# MAIN DATA COLLECTION
# --------------------------------------------------

def collect_race_data():

    logger.info("Starting F1 telemetry collection")

    logger.info(
        "Season: %s | Event: %s | Session: %s",
        YEAR, EVENT, SESSION_NAME
    )

    # 1. Retrieve the race session
    session = tif1.get_session(YEAR, EVENT, SESSION_NAME)

    # 2. Retrieve all available lap records
    laps_df = session.laps.copy()

    if laps_df.empty:
        raise RuntimeError("No lap data returned for this session.")

    logger.info("Total lap records retrieved: %s", len(laps_df))

    # 3. Identify available drivers
    drivers = sorted(laps_df["Driver"].dropna().unique())

    if MAX_DRIVERS is not None:
        drivers = drivers[:MAX_DRIVERS]

    logger.info("Drivers selected: %s", drivers)

    telemetry_frames = []
    failed_records = []

    successful_laps = 0

    # 4. Retrieve telemetry for individual driver laps
    for driver_code in drivers:

        logger.info("Processing driver: %s", driver_code)

        try:
            driver = session.get_driver(driver_code)
            driver_laps = driver.laps.copy()

            available_laps = (
                driver_laps.dropna(subset=["LapNumber"])
                .sort_values("LapNumber")
            )

            if MAX_LAPS_PER_DRIVER is not None:
                available_laps = available_laps.head(
                    MAX_LAPS_PER_DRIVER
                )

            for _, lap_record in available_laps.iterrows():

                lap_number = int(lap_record["LapNumber"])

                try:
                    logger.info(
                        "Fetching telemetry: Driver=%s Lap=%s",
                        driver_code,
                        lap_number
                    )

                    lap = driver.get_lap(lap_number)
                    telemetry = lap.telemetry.copy()

                    if telemetry.empty:
                        logger.warning(
                            "Empty telemetry: %s lap %s",
                            driver_code,
                            lap_number
                        )
                        continue

                    # Preserve race and lap identifiers
                    telemetry["Year"] = YEAR
                    telemetry["Event"] = EVENT
                    telemetry["Session"] = SESSION_NAME
                    telemetry["Driver"] = driver_code
                    telemetry["LapNumber"] = lap_number

                    telemetry_frames.append(telemetry)

                    successful_laps += 1

                    logger.info(
                        "Retrieved %s telemetry samples",
                        len(telemetry)
                    )

                except Exception as exc:
                    logger.exception(
                        "Failed: %s lap %s",
                        driver_code,
                        lap_number
                    )

                    failed_records.append({
                        "Driver": driver_code,
                        "LapNumber": lap_number,
                        "Error": str(exc)
                    })

        except Exception as exc:
            logger.exception(
                "Failed to process driver %s",
                driver_code
            )

            failed_records.append({
                "Driver": driver_code,
                "LapNumber": None,
                "Error": str(exc)
            })

    # 5. Combine telemetry
    if telemetry_frames:
        telemetry_df = pd.concat(
            telemetry_frames,
            ignore_index=True
        )
    else:
        telemetry_df = pd.DataFrame()

    # 6. Save lap and telemetry datasets
    event_slug = safe_name(EVENT)

    lap_filename = f"{YEAR}_{event_slug}_race_laps"
    telemetry_filename = f"{YEAR}_{event_slug}_race_telemetry"

    save_dataset(laps_df, lap_filename)

    if not telemetry_df.empty:
        save_dataset(telemetry_df, telemetry_filename)

    # 7. Save failed-record report
    failures_df = pd.DataFrame(failed_records)

    if not failures_df.empty:
        failures_df.to_csv(
            LOG_DIR / f"{YEAR}_{event_slug}_failures.csv",
            index=False
        )

    # 8. Save collection metadata
    metadata = {
        "year": YEAR,
        "event": EVENT,
        "session": SESSION_NAME,
        "collection_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_lap_records": len(laps_df),
        "selected_drivers": drivers,
        "successful_telemetry_laps": successful_laps,
        "telemetry_samples": len(telemetry_df),
        "failed_records": len(failed_records),
        "telemetry_columns": (
            telemetry_df.columns.tolist()
            if not telemetry_df.empty else []
        )
    }

    metadata_path = (
        METADATA_DIR / f"{YEAR}_{event_slug}_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, default=str)

    logger.info("Collection completed")
    logger.info("Successful telemetry laps: %s", successful_laps)
    logger.info("Total telemetry samples: %s", len(telemetry_df))
    logger.info("Failed records: %s", len(failed_records))
    logger.info("Metadata saved: %s", metadata_path)


if __name__ == "__main__":
    collect_race_data()
