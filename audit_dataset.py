
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

laps = pd.read_csv(DATA_DIR / "abu_dhabi_2024_laps.csv")
telemetry = pd.read_csv(DATA_DIR / "abu_dhabi_2024_telemetry.csv")

print("\n--- RACE DATA SUMMARY ---")

print("Total lap records:", len(laps))
print("Unique drivers:", laps["Driver"].nunique())
print("Drivers:", sorted(laps["Driver"].dropna().unique()))

print("\nLap records per driver:")
print(laps.groupby("Driver")["LapNumber"].count().sort_values(ascending=False))

print("\nValid lap-time records:")
valid_laps = laps.dropna(subset=["LapTimeSeconds"])
print(len(valid_laps))

print("\nAccurate lap records:")
print(laps["IsAccurate"].value_counts(dropna=False))

print("\n--- TELEMETRY SUMMARY ---")

print("Telemetry samples:", len(telemetry))
print("Drivers:", telemetry["Driver"].unique())
print("Lap numbers:", telemetry["LapNumber"].unique())

print("\nTelemetry time range:")
print(telemetry["Date"].min(), "to", telemetry["Date"].max())

print("\nTelemetry sample counts by driver and lap:")
print(
    telemetry.groupby(["Driver", "LapNumber"])
    .size()
    .sort_values(ascending=False)
)

print("\n--- DATA QUALITY ---")

print("Missing lap times:", laps["LapTimeSeconds"].isna().sum())
print("Missing speed values:", telemetry["Speed"].isna().sum())
print("Missing throttle values:", telemetry["Throttle"].isna().sum())
print("Missing brake values:", telemetry["Brake"].isna().sum())

print("Missing steering values:", telemetry["Steering"].isna().sum())
print("\nAudit complete.")
