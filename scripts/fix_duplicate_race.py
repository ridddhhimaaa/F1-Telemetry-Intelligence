from src.database.connection import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine()
with engine.begin() as conn:
    # Check if ABU_DHABI_2024 exists
    cnt = conn.execute(text("SELECT COUNT(*) FROM races WHERE race_id = 'ABU_DHABI_2024';")).scalar()
    print(f"Found {cnt} rows with race_id = 'ABU_DHABI_2024'")
    if cnt > 0:
        conn.execute(text("DELETE FROM races WHERE race_id = 'ABU_DHABI_2024';"))
        print("Deleted duplicate ABU_DHABI_2024 record.")
    
    # Check if constraint exists, if not add it
    chk = conn.execute(text("""
        SELECT COUNT(*) FROM pg_constraint 
        WHERE conname = 'uq_race_year_round' AND conrelid = 'races'::regclass;
    """)).scalar()
    if chk == 0:
        conn.execute(text("ALTER TABLE races ADD CONSTRAINT uq_race_year_round UNIQUE (year, round_number);"))
        print("Added constraint uq_race_year_round UNIQUE (year, round_number).")
    else:
        print("Constraint uq_race_year_round already exists.")

with engine.connect() as conn:
    df = pd.read_sql("SELECT * FROM races;", conn)
    print("Final races table contents:")
    print(df)
