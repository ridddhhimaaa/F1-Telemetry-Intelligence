"""
F1 Telemetry Intelligence: Relational Database Ingestion Pipeline
Loads processed and engineered feature datasets into the relational database
(PostgreSQL or SQLite fallback) with chunking, transaction management, and validation.
"""

import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from src.database.connection import get_engine, DEFAULT_SQLITE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
RAW_LAPS_PATH = PROJECT_ROOT / "data" / "processed" / "abu_dhabi_2024" / "processed_race_laps.parquet"


def initialize_schema(engine):
    """Execute DDL table creation scripts if tables do not exist."""
    sql_path = PROJECT_ROOT / "sql" / "02_create_tables.sql"
    if not sql_path.exists():
        logger.warning(f"DDL script not found at {sql_path}")
        return
    
    with open(sql_path, "r", encoding="utf-8") as f:
        ddl_sql = f.read()

    # If running against SQLite, adapt SERIAL / BIGSERIAL / DOUBLE PRECISION syntax
    if engine.dialect.name == "sqlite":
        ddl_sql = ddl_sql.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        ddl_sql = ddl_sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        ddl_sql = ddl_sql.replace("DOUBLE PRECISION", "REAL")
    
    statements = [stmt.strip() for stmt in ddl_sql.split(";") if stmt.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("Database schema initialized successfully.")


def load_drivers_and_races(engine):
    """Load metadata reference tables: drivers and races."""
    # Drivers table
    if RAW_LAPS_PATH.exists():
        raw_laps = pd.read_parquet(RAW_LAPS_PATH)
        driver_df = raw_laps[["Driver", "Team", "DriverNumber"]].drop_duplicates().copy()
        driver_df.rename(columns={
            "Driver": "driver_id",
            "Team": "team",
            "DriverNumber": "driver_number",
        }, inplace=True)
        driver_df["driver_name"] = driver_df["driver_id"]
    else:
        # Fallback driver codes
        driver_df = pd.DataFrame([
            {"driver_id": "VER", "driver_name": "Max Verstappen", "team": "Red Bull Racing", "driver_number": 1},
            {"driver_id": "HAM", "driver_name": "Lewis Hamilton", "team": "Mercedes", "driver_number": 44},
            {"driver_id": "LEC", "driver_name": "Charles Leclerc", "team": "Ferrari", "driver_number": 16},
            {"driver_id": "NOR", "driver_name": "Lando Norris", "team": "McLaren", "driver_number": 4},
        ])
    
    # Filter only drivers present in lap features to avoid orphan reference issues
    laps_path = FEATURES_DIR / "lap_features.parquet"
    if laps_path.exists():
        active_drivers = set(pd.read_parquet(laps_path)["Driver"].unique())
        driver_df = driver_df[driver_df["driver_id"].isin(active_drivers)].drop_duplicates(subset=["driver_id"])

    # Race table (canonical identifier: 2024_abu_dhabi)
    race_df = pd.DataFrame([{
        "race_id": "2024_abu_dhabi",
        "event_name": "Abu Dhabi Grand Prix",
        "circuit_name": "Yas Marina Circuit",
        "year": 2024,
        "round_number": 24,
        "circuit_length_m": 5281.0,
    }])

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM races WHERE race_id IN ('2024_abu_dhabi', 'ABU_DHABI_2024') OR (year = 2024 AND round_number = 24);"))
        race_df.to_sql("races", conn, if_exists="append", index=False)
        for _, row in driver_df.iterrows():
            if engine.dialect.name == "postgresql":
                conn.execute(text("""
                    INSERT INTO drivers (driver_id, driver_name, team, driver_number)
                    VALUES (:did, :dname, :team, :dnum)
                    ON CONFLICT (driver_id) DO UPDATE SET team = :team, driver_number = :dnum;
                """), {"did": row["driver_id"], "dname": row["driver_name"], "team": row["team"], "dnum": int(row["driver_number"])})
            else:
                conn.execute(text("""
                    INSERT OR REPLACE INTO drivers (driver_id, driver_name, team, driver_number)
                    VALUES (:did, :dname, :team, :dnum);
                """), {"did": row["driver_id"], "dname": row["driver_name"], "team": row["team"], "dnum": int(row["driver_number"])})
    logger.info(f"Loaded {len(driver_df)} drivers and 1 race reference record.")


def load_laps(engine):
    """Load lap features dataset into laps table."""
    laps_path = FEATURES_DIR / "lap_features.parquet"
    if not laps_path.exists():
        raise FileNotFoundError(f"Missing {laps_path}")
    
    df = pd.read_parquet(laps_path).copy()
    col_map = {
        "Driver": "driver_id",
        "LapNumber": "lap_number",
        "LapClassification": "lap_classification",
        "LapTime_s": "lap_time_s",
        "TelemetryDuration_s": "telemetry_duration_s",
        "LapDistance_m": "lap_distance_m",
        "Speed_Mean_kmh": "speed_mean_kmh",
        "Speed_Max_kmh": "speed_max_kmh",
        "Speed_Min_kmh": "speed_min_kmh",
        "Speed_Std_kmh": "speed_std_kmh",
        "Speed_Median_kmh": "speed_median_kmh",
        "FullThrottle_Pct": "full_throttle_pct",
        "Throttle_Mean_Pct": "throttle_mean_pct",
        "Throttle_Transitions": "throttle_transitions",
        "Braking_Duration_s": "braking_duration_s",
        "Braking_Pct": "braking_pct",
        "Braking_Events_Count": "braking_events_count",
        "Peak_Decel_G": "peak_decel_g",
        "Peak_Accel_G": "peak_accel_g",
        "Mean_Accel_mps2": "mean_accel_mps2",
        "Gear_Changes_Count": "gear_changes_count",
        "Upshifts_Count": "upshifts_count",
        "Downshifts_Count": "downshifts_count",
        "Most_Used_Gear": "most_used_gear",
        "Corners_Count": "corners_count",
        "Mean_Apex_Speed_kmh": "mean_apex_speed_kmh",
        "Min_Apex_Speed_kmh": "min_apex_speed_kmh",
        "Sector1_Duration_s": "sector1_duration_s",
        "Sector2_Duration_s": "sector2_duration_s",
        "Sector3_Duration_s": "sector3_duration_s",
        "Sample_Count": "sample_count",
        "Sampling_Frequency_Hz": "sampling_frequency_hz",
    }
    df.rename(columns=col_map, inplace=True)
    target_cols = [c for c in col_map.values() if c in df.columns]
    df = df[target_cols]

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM laps;"))
        df.to_sql("laps", conn, if_exists="append", index=False)
    logger.info(f"Loaded {len(df)} laps.")


def load_sectors(engine):
    """Load sector features dataset into sectors table."""
    sec_path = FEATURES_DIR / "sector_features.parquet"
    if not sec_path.exists():
        raise FileNotFoundError(f"Missing {sec_path}")
    
    df = pd.read_parquet(sec_path).copy()
    col_map = {
        "Driver": "driver_id",
        "LapNumber": "lap_number",
        "Sector": "sector",
        "SectorDuration_s": "sector_duration_s",
        "Speed_Mean_kmh": "speed_mean_kmh",
        "Speed_Max_kmh": "speed_max_kmh",
        "Speed_Min_kmh": "speed_min_kmh",
        "FullThrottle_Pct": "full_throttle_pct",
        "Throttle_Mean_Pct": "throttle_mean_pct",
        "Braking_Duration_s": "braking_duration_s",
        "Peak_Decel_G": "peak_decel_g",
        "Most_Used_Gear": "most_used_gear",
        "Gear_Shifts_Count": "gear_shifts_count",
        "Corners_Count": "corners_count",
        "Samples_Count": "samples_count",
    }
    df.rename(columns=col_map, inplace=True)
    target_cols = [c for c in col_map.values() if c in df.columns]
    df = df[target_cols]

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM sectors;"))
        df.to_sql("sectors", conn, if_exists="append", index=False)
    logger.info(f"Loaded {len(df)} sectors.")


def load_corners(engine):
    """Load corner features dataset into corners table."""
    cor_path = FEATURES_DIR / "corner_features.parquet"
    if not cor_path.exists():
        raise FileNotFoundError(f"Missing {cor_path}")
    
    df = pd.read_parquet(cor_path).copy()
    col_map = {
        "Driver": "driver_id",
        "LapNumber": "lap_number",
        "Corner_ID": "corner_id",
        "TurnDirection": "turn_direction",
        "StartDistance_m": "start_distance_m",
        "ApexDistance_m": "apex_distance_m",
        "EndDistance_m": "end_distance_m",
        "CornerLength_m": "corner_length_m",
        "EntrySpeed_kmh": "entry_speed_kmh",
        "ApexSpeed_kmh": "apex_speed_kmh",
        "ExitSpeed_kmh": "exit_speed_kmh",
        "SpeedLoss_kmh": "speed_loss_kmh",
        "ExitAccel_kmh": "exit_accel_kmh",
        "CornerDuration_s": "corner_duration_s",
        "MinRadius_m": "min_radius_m",
    }
    df.rename(columns=col_map, inplace=True)
    target_cols = [c for c in col_map.values() if c in df.columns]
    df = df[target_cols]

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM corners;"))
        df.to_sql("corners", conn, if_exists="append", index=False)
    logger.info(f"Loaded {len(df)} corners.")


def load_telemetry(engine, chunksize: int = 10000):
    """Load high-frequency telemetry features in chunks."""
    tel_path = FEATURES_DIR / "telemetry_features.parquet"
    if not tel_path.exists():
        raise FileNotFoundError(f"Missing {tel_path}")
    
    tel_df = pd.read_parquet(tel_path)
    
    # Build column mapping
    df = pd.DataFrame()
    df["driver_id"] = tel_df["Driver"]
    df["lap_number"] = tel_df["LapNumber"]
    
    if pd.api.types.is_timedelta64_dtype(tel_df["Time"]):
        df["time_seconds"] = tel_df["Time"].dt.total_seconds()
    else:
        df["time_seconds"] = tel_df["Time"].astype(float)
        
    df["distance_m"] = tel_df["Distance"].astype(float)
    df["speed_kmh"] = tel_df["Speed"].astype(float)
    df["rpm"] = tel_df["RPM"].astype(int)
    df["gear"] = tel_df["nGear"].astype(int)
    df["throttle_pct"] = (
        tel_df["Throttle_Clipped"].astype(float)
        if "Throttle_Clipped" in tel_df.columns
        else tel_df["Throttle"].clip(0.0, 100.0).astype(float)
    )
    df["is_braking"] = tel_df["Brake_Int"].astype(int)
    df["drs"] = tel_df["DRS"].astype(int)
    df["lap_classification"] = tel_df["LapClassification"]
    
    df["x_m"] = tel_df["X_m"].astype(float)
    df["y_m"] = tel_df["Y_m"].astype(float)
    df["heading_deg"] = tel_df["Heading_deg"].astype(float)
    df["curvature_m_inv"] = tel_df["Curvature_m_inv"].astype(float)
    df["radius_of_curvature_m"] = tel_df["Radius_of_Curvature_m"].astype(float)
    df["corner_id"] = tel_df["Corner_ID"].fillna(0).astype(int)
    df["is_corner_zone"] = tel_df["IsCornerZone"].astype(int)
    df["is_corner_apex"] = tel_df["IsCornerApex"].astype(int)
    df["turn_direction"] = tel_df["TurnDirection"]
    
    df["accel_longitudinal_mps2"] = tel_df["Accel_Longitudinal_mps2"].astype(float)
    df["g_longitudinal"] = tel_df["G_Longitudinal"].astype(float)
    df["accel_lateral_mps2"] = tel_df["Accel_Lateral_mps2"].astype(float)
    df["g_lateral"] = tel_df["G_Lateral"].astype(float)
    df["g_combined"] = tel_df["G_Combined"].astype(float)
    
    df["is_accelerating"] = tel_df["IsAccelerating"].astype(int)
    df["is_decelerating"] = tel_df["IsDecelerating"].astype(int)
    df["is_coasting"] = tel_df["IsCoasting"].astype(int)
    df["dynamics_regime"] = tel_df["DynamicsRegime"]
    df["accel_phase_duration_s"] = tel_df["AccelPhaseDuration_s"].astype(float)
    df["decel_phase_duration_s"] = tel_df["DecelPhaseDuration_s"].astype(float)
    
    df["throttle_application_rate"] = tel_df["Throttle_Application_Rate"].astype(float)
    df["throttle_release_rate"] = tel_df["Throttle_Release_Rate"].astype(float)
    df["is_full_throttle"] = tel_df["IsFullThrottle"].astype(int)
    df["full_throttle_duration_s"] = tel_df["FullThrottle_Duration_s"].astype(float)
    df["full_throttle_cumulative_pct"] = tel_df["FullThrottle_Cumulative_Pct"].astype(float)
    df["throttle_transition_count"] = tel_df["Throttle_Transition_Count"].astype(int)
    
    df["braking_event_id"] = tel_df["Braking_Event_ID"].fillna(0).astype(int)
    df["is_braking_onset"] = tel_df["IsBrakingOnset"].astype(int)
    df["braking_duration_s"] = tel_df["BrakingDuration_s"].astype(float)
    df["braking_distance_m"] = tel_df["BrakingDistance_m"].astype(float)
    df["brake_event_peak_g"] = tel_df["BrakeEvent_Peak_G"].astype(float)
    df["is_brake_release_phase"] = tel_df["IsBrakeReleasePhase"].astype(int)
    df["is_pedal_overlap"] = tel_df["IsPedalOverlap"].astype(int)
    
    df["ratio_rpm_per_kmh"] = tel_df["Ratio_RPM_per_kmh"].astype(float)
    df["rpm_change_rate_per_s"] = tel_df["RPM_Change_Rate_per_s"].astype(float)
    df["is_gear_shift"] = tel_df["IsGearShift"].astype(int)
    df["is_upshift"] = tel_df["IsUpshift"].astype(int)
    df["is_downshift"] = tel_df["IsDownshift"].astype(int)
    df["gear_change_count"] = tel_df["Gear_Change_Count"].astype(int)
    
    df["sector"] = tel_df["Sector"].astype(int)
    df["distance_in_sector_m"] = tel_df["Distance_In_Sector_m"].astype(float)

    total_rows = len(df)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM telemetry_features;"))
    
    logger.info(f"Loading {total_rows} telemetry feature rows in chunks of {chunksize}...")
    for start_idx in range(0, total_rows, chunksize):
        chunk = df.iloc[start_idx : start_idx + chunksize]
        chunk.to_sql("telemetry_features", engine, if_exists="append", index=False)
        logger.info(f"Loaded rows {start_idx} to {min(start_idx + chunksize, total_rows)} of {total_rows}")
    logger.info("Telemetry features loading completed.")


def load_all_features_to_db(engine=None):
    """Full orchestration: initialize schema and load all datasets."""
    if engine is None:
        engine = get_engine()
    
    logger.info(f"Starting data loading pipeline using dialect '{engine.dialect.name}'...")
    initialize_schema(engine)
    load_drivers_and_races(engine)
    load_laps(engine)
    load_sectors(engine)
    load_corners(engine)
    load_telemetry(engine)
    
    # Audit loaded counts
    with engine.connect() as conn:
        drivers_cnt = conn.execute(text("SELECT COUNT(*) FROM drivers;")).scalar()
        laps_cnt = conn.execute(text("SELECT COUNT(*) FROM laps;")).scalar()
        sectors_cnt = conn.execute(text("SELECT COUNT(*) FROM sectors;")).scalar()
        corners_cnt = conn.execute(text("SELECT COUNT(*) FROM corners;")).scalar()
        tel_cnt = conn.execute(text("SELECT COUNT(*) FROM telemetry_features;")).scalar()
    
    logger.info(f"--- Ingestion Verification ---")
    logger.info(f"Drivers count: {drivers_cnt} (expected 19)")
    logger.info(f"Laps count: {laps_cnt} (expected 96)")
    logger.info(f"Sectors count: {sectors_cnt} (expected 288)")
    logger.info(f"Corners count: {corners_cnt} (expected 1469)")
    logger.info(f"Telemetry count: {tel_cnt} (expected 72476)")
    return {
        "drivers": drivers_cnt,
        "laps": laps_cnt,
        "sectors": sectors_cnt,
        "corners": corners_cnt,
        "telemetry_features": tel_cnt,
    }


if __name__ == "__main__":
    load_all_features_to_db()
