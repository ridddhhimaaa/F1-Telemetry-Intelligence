import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import pandas as pd
from sqlalchemy import text
from src.database.connection import get_engine, is_postgres_available
from src.database.query_runner import run_query
from src.database.verify_sql_vs_pandas import run_all_cross_validations

print("=== 1. POSTGRESQL CONNECTIVITY & VERSION ===")
is_avail = is_postgres_available(timeout_sec=2)
print(f"is_postgres_available(): {is_avail}")
engine = get_engine()
print(f"Engine dialect: {engine.dialect.name}")

with engine.connect() as conn:
    ver = conn.execute(text("SELECT version();")).scalar()
    db_name = conn.execute(text("SELECT current_database();")).scalar()
    user_name = conn.execute(text("SELECT current_user;")).scalar()
    print(f"Version:  {ver}")
    print(f"Database: {db_name}")
    print(f"User:     {user_name}")

    print("\n=== 2. TABLE EXISTENCE & ROW COUNTS ===")
    tables = ["drivers", "races", "laps", "sectors", "corners", "telemetry_features"]
    counts = {}
    for tbl in tables:
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()
        counts[tbl] = cnt
        print(f"Table '{tbl}': {cnt} rows")

    print("\n=== 3. CONSTRAINTS & KEYS CHECK ===")
    pk_query = """
    SELECT 
        tc.table_name, 
        tc.constraint_name, 
        tc.constraint_type
    FROM information_schema.table_constraints tc
    WHERE tc.table_schema = 'public'
    ORDER BY tc.table_name, tc.constraint_type;
    """
    constraints_df = pd.read_sql(text(pk_query), conn)
    print(constraints_df.to_string(index=False))

    print("\n=== 4. ORPHAN RECORD VERIFICATION ===")
    orphan_laps = conn.execute(text("SELECT COUNT(*) FROM laps l LEFT JOIN drivers d ON l.driver_id = d.driver_id WHERE d.driver_id IS NULL;")).scalar()
    orphan_sectors = conn.execute(text("SELECT COUNT(*) FROM sectors s LEFT JOIN laps l ON s.driver_id = l.driver_id AND s.lap_number = l.lap_number WHERE l.driver_id IS NULL;")).scalar()
    orphan_corners = conn.execute(text("SELECT COUNT(*) FROM corners c LEFT JOIN laps l ON c.driver_id = l.driver_id AND c.lap_number = l.lap_number WHERE l.driver_id IS NULL;")).scalar()
    orphan_telemetry = conn.execute(text("SELECT COUNT(*) FROM telemetry_features t LEFT JOIN laps l ON t.driver_id = l.driver_id AND t.lap_number = l.lap_number WHERE l.driver_id IS NULL;")).scalar()
    print(f"Orphan Laps (no driver):      {orphan_laps}")
    print(f"Orphan Sectors (no lap):      {orphan_sectors}")
    print(f"Orphan Corners (no lap):      {orphan_corners}")
    print(f"Orphan Telemetry (no lap):    {orphan_telemetry}")

print("\n=== 5. SQL VS PANDAS FRESH CROSS-VALIDATION ===")
cross_val_pass = run_all_cross_validations(engine=engine)
print(f"Cross-Validation All Passed: {cross_val_pass}")
