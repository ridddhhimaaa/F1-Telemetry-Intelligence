
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

files = {
    "laps": DATA_DIR / "abu_dhabi_2024_laps.csv",
    "telemetry": DATA_DIR / "abu_dhabi_2024_telemetry.csv"
}

for name, path in files.items():
    print(f"\n{'=' * 50}")
    print(f"DATASET: {name.upper()}")
    print(f"{'=' * 50}")

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(15))

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nSample:")
    print(df.head(3).to_string())
