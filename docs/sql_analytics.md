# SQL Analytics & Script Catalog

**Project:** F1 Telemetry Intelligence  
**Phase:** Phase 6 / 6.1 – Advanced Feature Engineering & Local PostgreSQL Analytics  
**Database:** `f1_telemetry` on PostgreSQL 16.8 (localhost:5432)  

---

## 1. Overview of the SQL Suite

The `sql/` directory contains 10 structured, modular SQL scripts executed directly against local PostgreSQL in VS Code.

| File | Script Name | Analytical Focus | Core SQL Concepts | Exec Time | Rows |
|---|---|---|---|---:|---:|
| `01_create_database.sql` | Database Initialization | Creates `f1_telemetry` database with UTF-8 encoding | `CREATE DATABASE`, conditional existence check | < 1 ms | - |
| `02_create_tables.sql` | Relational Schema | Defines 6 normalized tables with primary/foreign keys | Foreign keys, cascade rules, table constraints | 15 ms | - |
| `03_constraints_and_indexes.sql` | Constraints & Indexes | Enforces data bounds and composite B-tree indexes | Check constraints, composite B-tree indexes | 24 ms | - |
| `04_data_loading.sql` | Reference Ingestion | Seeds driver roster and race metadata | `INSERT ON CONFLICT`, CSV copy templates | 2 ms | 22 |
| `05_basic_analysis.sql` | Basic SQL Analysis | Pace filtering, driver grouping, corner severity | `SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `CASE WHEN` | 65 ms | 156 |
| `06_joins_and_aggregations.sql` | Joins & Aggregations | Multi-table joins across drivers, laps, sectors, corners | `INNER JOIN`, `LEFT JOIN`, multi-table aggregates | 17 ms | 140 |
| `07_cte_analysis.sql` | CTE Analytics | Benchmark deficits, sector dominance, braking work | Common Table Expressions (`WITH`), chained CTEs | 10 ms | 41 |
| `08_window_functions.sql` | Window Functions | Lap rankings, moving averages, speed deltas, quartiles | `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `NTILE` | 46 ms | 778 |
| `09_advanced_telemetry_analysis.sql` | Advanced Telemetry | Speed traps, apex pace deltas, head-to-head comparisons | Multi-CTE slices, high-speed trap extraction | 70 ms | 201 |
| `10_data_quality_checks.sql` | Data Quality Audits | Null checks, speed limits, gear ranges, orphan records | Table unions, conditional count sums | 215 ms | 23 |

All script outputs are recorded and persisted in `reports/feature_engineering/sql_execution/`.

---

## 2. Key Analytical Takeaways from PostgreSQL Queries

1. **Clean Flying Lap Dominance:**
   - Filtering strictly on `lap_classification = 'COMPLETE_FLYING_LAP'` isolates genuine vehicle performance from safety car and pit in/out regimes.
   - Max Verstappen (`VER`) and Charles Leclerc (`LEC`) set the benchmark clean flying lap times under 1:24.0.
   - Fast lap pacing in clean flying conditions: Carlos Sainz (`SAI`, 88.995 s) and Lando Norris (`NOR`, 89.010 s) recorded the fastest qualifying-simulation laps.

2. **Corner Apex Speeds & Severity Classification:**
   - Yas Marina Turn 5 (tight hairpin) exhibits the lowest minimum radius ($10.0\text{ m}$) and highest speed loss ($\sim 220\text{ km/h}$ deceleration, entry speed $288\text{ km/h} \to$ apex $68\text{ km/h}$).
   - High-speed Turn 2 and Turn 3 exhibit minimal apex speed loss, sustaining speeds $> 235\text{ km/h}$ and pulling over $3.5\text{G}$ lateral acceleration.

3. **Window Function Smoothing vs. Raw Derivative:**
   - 5-sample centered moving averages (`ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING`) filter out high-frequency sensor noise in $a_{\text{long}}$ without attenuating real braking spikes.
   - Sample-to-sample speed delta ($\Delta v = v_i - v_{i-1}$) tracking via `LAG()` correlates with instantaneous longitudinal acceleration at $r = 0.99$.

4. **Speed Trap Performance:**
   - Maximum speeds exceed $337\text{ km/h}$ at distance $1,800 - 2,050\text{ m}$ on the back straight leading into Turn 6, with DRS engaged in 8th gear.

---

## 3. Query Performance & Execution Summary

Against the indexed PostgreSQL database:
- **Single-row aggregations & simple joins:** 1.0 – 5.0 ms
- **Multi-table CTE rankings:** 2.0 – 6.0 ms
- **Window function operations:** 2.0 – 25.0 ms
- **Full telemetry table sequential & index scans (72,476 rows):** 25.0 – 60.0 ms
- **Comprehensive Data Quality Checks (10 audits across all tables):** 215 ms total
