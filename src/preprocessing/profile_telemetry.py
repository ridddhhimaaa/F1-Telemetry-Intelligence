
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "2024_abu_dhabi_race_telemetry.parquet"

df = pd.read_parquet(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("Memory usage (MB):", round(df.memory_usage(deep=True).sum() / 1024**2, 2))
print("Drivers:", df["Driver"].nunique())
print("Telemetry laps:", df[["Driver", "LapNumber"]].drop_duplicates().shape[0])

print("\n========== STATISTICAL SUMMARY ==========")
columns = ["Speed", "RPM", "Throttle", "Brake", "nGear", "Distance"]
print(df[columns].describe().round(2))

print("\n========== DRIVER-WISE SUMMARY ==========")
print(
    df.groupby("Driver")
    .agg(
        samples=("Speed", "count"),
        avg_speed=("Speed", "mean"),
        max_speed=("Speed", "max"),
        avg_throttle=("Throttle", "mean"),
        braking_samples=("Brake", "sum")
    )
    .round(2)
    .sort_values("avg_speed", ascending=False)
)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== VALUE VALIDATION ==========")
for col in ["Speed", "RPM", "Throttle", "Brake", "nGear"]:
    print(f"\n{col}")
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())

print("\n========== DISTANCE VALIDATION ==========")
print(
    df.groupby(["Driver", "LapNumber"])["Distance"]
    .agg(["min", "max", "count"])
    .head(20)
)

print("\n========== COMPLETED ==========")
