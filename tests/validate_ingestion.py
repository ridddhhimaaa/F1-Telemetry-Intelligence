
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

laps_path = RAW / "2024_abu_dhabi_race_laps.parquet"
telemetry_path = RAW / "2024_abu_dhabi_race_telemetry.parquet"

laps = pd.read_parquet(laps_path)
telemetry = pd.read_parquet(telemetry_path)

print("\n========== LAP DATA ==========")
print("Shape:", laps.shape)
print("Unique drivers:", laps["Driver"].nunique())
print("Drivers:", sorted(laps["Driver"].dropna().unique()))

print("\n========== TELEMETRY DATA ==========")
print("Shape:", telemetry.shape)
print("Unique drivers:", telemetry["Driver"].nunique())
print("Drivers:", sorted(telemetry["Driver"].dropna().unique()))
print("Unique laps:", telemetry[["Driver", "LapNumber"]].drop_duplicates().shape[0])

print("\nSamples per driver and lap:")
print(
    telemetry.groupby(["Driver", "LapNumber"])
    .size()
    .to_string()
)

print("\nMissing values:")
print(telemetry.isnull().sum())

print("\nDuplicate rows:", telemetry.duplicated().sum())

print("\nTelemetry columns:")
print(telemetry.columns.tolist())
