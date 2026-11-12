import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from sqlalchemy import text
from src.database.connection import get_engine

engine = get_engine()

tables = ["laps", "sectors", "corners", "telemetry_features"]

print("=== INDEPENDENT OCO RECORD VERIFICATION IN POSTGRESQL ===")
with engine.connect() as conn:
    for tbl in tables:
        # Check by driver_id column
        col = "driver_id"
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl} WHERE {col} = 'OCO';")).scalar()
        print(f"Table '{tbl}': {cnt} rows for driver_id = 'OCO'")

print("\n=== VERIFICATION IN PARQUET FEATURE DATASETS ===")
import pandas as pd
FEATURES_DIR = Path("data/processed/features")
for fname in ["lap_features.parquet", "sector_features.parquet", "corner_features.parquet", "telemetry_features.parquet"]:
    df = pd.read_parquet(FEATURES_DIR / fname)
    col = "Driver" if "Driver" in df.columns else "driver_id"
    oco_cnt = (df[col] == "OCO").sum()
    print(f"File '{fname}': {oco_cnt} rows for Driver = 'OCO'")
