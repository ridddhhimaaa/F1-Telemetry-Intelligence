import pandas as pd
import numpy as np

# Load processed race laps and telemetry
laps_df = pd.read_parquet("data/processed/abu_dhabi_2024/processed_race_laps.parquet")
tel_df = pd.read_parquet("data/processed/features/telemetry_features.parquet")

# Filter complete flying laps
clean_laps = laps_df[laps_df["LapClassification"] == "COMPLETE_FLYING_LAP"].copy()

results = []

for _, lap_row in clean_laps.iterrows():
    driver = lap_row["Driver"]
    lap_num = int(lap_row["LapNumber"])
    
    if pd.isna(lap_row.get("Sector1Time")) or pd.isna(lap_row.get("Sector2Time")):
        continue
        
    s1_sec = lap_row["Sector1Time"].total_seconds()
    s2_sec = lap_row["Sector2Time"].total_seconds()
    s1_cum = s1_sec
    s2_cum = s1_sec + s2_sec
    
    tel = tel_df[(tel_df["Driver"] == driver) & (tel_df["LapNumber"] == lap_num)].copy()
    if len(tel) < 100:
        continue
        
    tel["t_sec"] = tel["Time"].dt.total_seconds()
    tel["t_rel"] = tel["t_sec"] - tel["t_sec"].iloc[0]
    
    # Nearest row to official timing beam
    idx_s1 = (tel["t_rel"] - s1_cum).abs().argmin()
    idx_s2 = (tel["t_rel"] - s2_cum).abs().argmin()
    
    dist_s1 = tel.iloc[idx_s1]["Distance"]
    dist_s2 = tel.iloc[idx_s2]["Distance"]
    
    results.append({
        "Driver": driver,
        "LapNumber": lap_num,
        "S1_Official_s": round(s1_sec, 3),
        "S2_Official_s": round(s2_sec, 3),
        "S1_Derived_Distance_m": round(dist_s1, 2),
        "S2_Derived_Distance_m": round(dist_s2, 2),
        "Delta_S1_from_1200m": round(dist_s1 - 1200.0, 2),
        "Delta_S2_from_3520m": round(dist_s2 - 3520.0, 2),
    })

res_df = pd.DataFrame(results)
print("=== Sector Boundary Analysis across Clean Flying Laps ===")
print(f"Total Clean Laps Analyzed: {len(res_df)}")
print("\nSample (first 10 laps):")
print(res_df.head(10)[["Driver", "LapNumber", "S1_Derived_Distance_m", "Delta_S1_from_1200m", "S2_Derived_Distance_m", "Delta_S2_from_3520m"]].to_string(index=False))

print("\nSummary Statistics for S1 Boundary (Fixed target: 1200 m):")
print(f"  Mean distance:   {res_df['S1_Derived_Distance_m'].mean():.2f} m")
print(f"  Median distance: {res_df['S1_Derived_Distance_m'].median():.2f} m")
print(f"  Min / Max:       [{res_df['S1_Derived_Distance_m'].min():.2f}, {res_df['S1_Derived_Distance_m'].max():.2f}] m")
print(f"  Std deviation:   {res_df['S1_Derived_Distance_m'].std():.2f} m")
print(f"  Mean deviation:  {res_df['Delta_S1_from_1200m'].mean():+.2f} m")

print("\nSummary Statistics for S2 Boundary (Fixed target: 3520 m):")
print(f"  Mean distance:   {res_df['S2_Derived_Distance_m'].mean():.2f} m")
print(f"  Median distance: {res_df['S2_Derived_Distance_m'].median():.2f} m")
print(f"  Min / Max:       [{res_df['S2_Derived_Distance_m'].min():.2f}, {res_df['S2_Derived_Distance_m'].max():.2f}] m")
print(f"  Std deviation:   {res_df['S2_Derived_Distance_m'].std():.2f} m")
print(f"  Mean deviation:  {res_df['Delta_S2_from_3520m'].mean():+.2f} m")
