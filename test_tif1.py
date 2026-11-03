import tif1
from pathlib import Path

print("Loading 2024 Abu Dhabi Grand Prix...")

session = tif1.get_session(2024, "Abu Dhabi", "Race")

print("\nAvailable drivers:")
print(session.drivers_df.head())

print("\nLoading lap timings...")
laps = session.laps
print(laps.head())

print("\nLoading fastest lap telemetry...")
telemetry = session.get_fastest_lap_tel()

print("\nTelemetry sample:")
print(telemetry.head())

print("\nTelemetry columns:")
print(telemetry.columns.tolist())

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

laps.to_csv(output_dir / "abu_dhabi_2024_laps.csv", index=False)
telemetry.to_csv(output_dir / "abu_dhabi_2024_telemetry.csv", index=False)

print("\nSUCCESS: Data saved to data/raw/")
