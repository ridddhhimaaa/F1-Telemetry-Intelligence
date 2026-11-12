import pandas as pd
import numpy as np

def audit_df(name, path):
    df = pd.read_parquet(path)
    print(f"\n==========================================")
    print(f"=== {name} ({df.shape}) ===")
    print(f"==========================================")
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        print("Columns with NULL values:")
        for col, val in null_cols.items():
            print(f"  - {col}: {val} ({val/len(df)*100:.2f}%)")
    else:
        print("Columns with NULL values: None (100% complete)")

    num_cols = df.select_dtypes(include=[np.number]).columns
    infs = np.isinf(df[num_cols]).sum()
    inf_cols = infs[infs > 0]
    if len(inf_cols) > 0:
        print("Columns with INF values:")
        for col, val in inf_cols.items():
            print(f"  - {col}: {val}")
    else:
        print("Columns with INF values: None (0)")

    dups = df.duplicated().sum()
    print(f"Duplicate rows: {dups}")

audit_df("Telemetry Features", "data/processed/features/telemetry_features.parquet")
audit_df("Lap Features", "data/processed/features/lap_features.parquet")
audit_df("Sector Features", "data/processed/features/sector_features.parquet")
audit_df("Corner Features", "data/processed/features/corner_features.parquet")
