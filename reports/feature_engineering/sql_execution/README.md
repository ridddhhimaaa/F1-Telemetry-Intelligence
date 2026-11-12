# SQL Execution Outputs and Benchmarks

**Project:** F1 Telemetry Intelligence  
**Phase:** Phase 6 / 6.1 – PostgreSQL Implementation & Analytics  
**Database:** `f1_telemetry` on PostgreSQL 16.8 (localhost:5432)  

This directory contains the actual outputs, benchmark timings, cross-validation records, and execution logs from executing all 10 SQL analytical suites against the live PostgreSQL database.

---

## Directory Contents

### Execution Logs & Benchmarks
- `sql_execution_log.txt`: Complete execution log with statement previews, row counts, timings, and formatted ASCII table previews.
- `sql_benchmarks.csv`: Machine-readable benchmarks for all 29 analytical queries across scripts 05 to 10.
- `sql_vs_pandas_validation.csv`: Audit results of the 5 mathematical comparisons between SQL and Pandas.

### Analytical Query Outputs (CSV Format)
- **05_basic_analysis:**
  - `05_basic_analysis_q01.csv`: Completed laps by driver and classification.
  - `05_basic_analysis_q02.csv`: Clean flying laps ranking by lap time.
  - `05_basic_analysis_q03.csv`: Driver average flying lap pace with HAVING filter.
  - `05_basic_analysis_q04.csv`: Corner severity classification using CASE WHEN.
  - `05_basic_analysis_q05.csv`: Telemetry dynamic regime distribution with percentage shares.
- **06_joins_and_aggregations:**
  - `06_joins_and_aggregations_q01.csv`: Multi-table join across drivers and laps.
  - `06_joins_and_aggregations_q02.csv`: Team-level flying lap pace aggregations.
  - `06_joins_and_aggregations_q03.csv`: Sector breakdown with driver and team metadata.
  - `06_joins_and_aggregations_q04.csv`: Corner kinematics and speed loss aggregates.
- **07_cte_analysis:**
  - `07_cte_analysis_q01.csv`: Driver pace deficit relative to session benchmark lap.
  - `07_cte_analysis_q02.csv`: Theoretical fastest driver per sector.
  - `07_cte_analysis_q03.csv`: Braking intensity and deceleration vs fleet average.
- **08_window_functions:**
  - `08_window_functions_q01.csv`: Lap ranking using RANK(), DENSE_RANK(), and ROW_NUMBER().
  - `08_window_functions_q02.csv`: Sample-to-sample speed delta and look-ahead acceleration with LAG/LEAD.
  - `08_window_functions_q03.csv`: 5-sample centered moving average smoothing.
  - `08_window_functions_q04.csv`: Running maximum and minimum speed accumulation.
  - `08_window_functions_q05.csv`: Corner apex speed quartiles (NTILE) and percentiles (CUME_DIST).
- **09_advanced_telemetry_analysis:**
  - `09_advanced_telemetry_analysis_q01.csv`: Exact speed trap locations and peak speeds per driver.
  - `09_advanced_telemetry_analysis_q02.csv`: Corner apex delta matrix against session benchmark.
  - `09_advanced_telemetry_analysis_q03.csv`: Sector-level pace deficits without regime mixing.
  - `09_advanced_telemetry_analysis_q04.csv`: Head-to-head telemetry slice comparison (VER vs HAM).
- **10_data_quality_checks:**
  - `10_data_quality_checks_q01.csv` to `q09.csv`: Null checks, bounds validation, gear checks, row count audits, orphan checks, and classification distributions.
