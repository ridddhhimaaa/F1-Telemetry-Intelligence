# PostgreSQL Local Setup & Integration Guide

**Project:** F1 Telemetry Intelligence  
**Component:** Local Relational Database (`f1_telemetry`)  
**Database Engine:** PostgreSQL 16.8 (x86_64-pc-windows-msvc)  
**Host / Port:** `localhost:5432`  

---

## 1. Verified Local PostgreSQL Environment

PostgreSQL is installed and running locally on Windows without cloud dependencies or SQLite substitutions.

- **Binaries Directory:** `C:\Users\Riddhima\pgsql\bin`
- **Data Cluster Directory:** `C:\Users\Riddhima\pgsql\data`
- **Database Server Binary:** `C:\Users\Riddhima\pgsql\bin\postgres.exe`
- **Client Tool:** `C:\Users\Riddhima\pgsql\bin\psql.exe`
- **User / Superuser:** `postgres`
- **Default Database:** `f1_telemetry`
- **Port:** `5432`

---

## 2. Server Control Commands (PowerShell / VS Code Terminal)

### Start Database Server
To start the PostgreSQL daemon process in the background:
```powershell
C:\Users\Riddhima\pgsql\bin\pg_ctl.exe -D C:\Users\Riddhima\pgsql\data -l C:\Users\Riddhima\pgsql\data\server.log start
```

### Check Server Status
```powershell
C:\Users\Riddhima\pgsql\bin\pg_ctl.exe -D C:\Users\Riddhima\pgsql\data status
```

### Stop Database Server
```powershell
C:\Users\Riddhima\pgsql\bin\pg_ctl.exe -D C:\Users\Riddhima\pgsql\data stop
```

---

## 3. Configuration & Credentials

The database connection parameters are configured via `.env` in the project root:

```ini
DB_HOST=localhost
DB_PORT=5432
DB_NAME=f1_telemetry
DB_USER=postgres
DB_PASSWORD=postgres
USE_SQLITE_FALLBACK=0
```

- Template file `.env.example` is committed to source control.
- File `.env` contains local settings and is strictly excluded via `.gitignore`.
- No credentials or passwords are hardcoded or printed in execution logs.

---

## 4. Database Creation & Schema Initialization

### Step 1: Create Database
```powershell
C:\Users\Riddhima\pgsql\bin\createdb.exe -U postgres -h localhost -p 5432 f1_telemetry
```

### Step 2: Execute DDL & Constraints
Execute table definitions and integrity constraints using `psql`:
```powershell
C:\Users\Riddhima\pgsql\bin\psql.exe -U postgres -h localhost -d f1_telemetry -f sql/01_create_database.sql
C:\Users\Riddhima\pgsql\bin\psql.exe -U postgres -h localhost -d f1_telemetry -f sql/02_create_tables.sql
C:\Users\Riddhima\pgsql\bin\psql.exe -U postgres -h localhost -d f1_telemetry -f sql/03_constraints_and_indexes.sql
C:\Users\Riddhima\pgsql\bin\psql.exe -U postgres -h localhost -d f1_telemetry -f sql/04_data_loading.sql
```

### Step 3: Run Automated Python Ingestion
```powershell
.\.venv\Scripts\python.exe -m src.database.load_data
```

This populates:
- `drivers`: 21 drivers
- `races`: 1 race record (`2024_abu_dhabi`)
- `laps`: 96 laps
- `sectors`: 288 sectors
- `corners`: 1,469 corner traversals
- `telemetry_features`: 72,476 high-frequency samples in chunks of 10,000

---

## 5. Analytical Execution & Cross-Validation

### Execute All Analytical SQL Scripts (05 through 10)
```powershell
.\.venv\Scripts\python.exe -m src.database.query_runner
```
Outputs and benchmark logs are automatically exported to:
`reports/feature_engineering/sql_execution/`

### Run SQL vs. Pandas Mathematical Cross-Validation
```powershell
.\.venv\Scripts\python.exe -m src.database.verify_sql_vs_pandas
```
Validates 5 analytical suites (clean lap pace, sector benchmarks, apex speeds, speed traps, braking/throttle totals) with zero discrepancies.

### Run Database Unit Tests
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_database.py
```
All 5 database unit tests pass cleanly against the live PostgreSQL server.

---

## 6. VS Code Workflow & Extensions

1. **SQLTools Extension:**
   - Install `mtxr.sqltools` and `mtxr.sqltools-driver-pg`.
   - Add a connection pointing to `localhost:5432`, database `f1_telemetry`, username `postgres`.
2. **Integrated Terminal Execution:**
   - Use `psql -U postgres -h localhost -d f1_telemetry` for interactive queries.
   - Use `\i sql/<script_name>.sql` to execute scripts interactively.
3. **Export Query Results:**
   - `\copy (SELECT * FROM laps WHERE lap_classification = 'COMPLETE_FLYING_LAP') TO 'output.csv' WITH CSV HEADER;`
