-- ==============================================================================
-- F1 Telemetry Intelligence: Phase 6 Local SQL Analytics
-- File: sql/10_data_quality_checks.sql
-- Description: Rigorous data quality verification queries:
--              1. Critical column null counts.
--              2. Negative / unphysical speed checks.
--              3. Gear range validation (1..8).
--              4. Row count audits across tables.
--              5. Foreign key integrity and orphan record detection.
--              6. Clean flying lap classification integrity.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Critical Column Null Checks in telemetry_features
--    Validates that essential identifiers and signals contain zero nulls
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_telemetry_rows,
    COUNT(*) - COUNT(driver_id) AS null_driver_id,
    COUNT(*) - COUNT(lap_number) AS null_lap_number,
    COUNT(*) - COUNT(distance_m) AS null_distance,
    COUNT(*) - COUNT(speed_kmh) AS null_speed_kmh,
    COUNT(*) - COUNT(throttle_pct) AS null_throttle_pct,
    COUNT(*) - COUNT(is_braking) AS null_is_braking,
    COUNT(*) - COUNT(gear) AS null_gear,
    COUNT(*) - COUNT(accel_longitudinal_mps2) AS null_accel_long,
    COUNT(*) - COUNT(curvature_m_inv) AS null_curvature
FROM telemetry_features;

-- ------------------------------------------------------------------------------
-- 2. Physical Bounds Validation: Negative Speeds and Extreme Accel
--    F1 cars cannot move backwards or exceed physical grip thresholds on dry asphalt
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_violations,
    SUM(CASE WHEN speed_kmh < 0 THEN 1 ELSE 0 END) AS negative_speed_count,
    SUM(CASE WHEN speed_kmh > 400 THEN 1 ELSE 0 END) AS excessive_speed_count,
    SUM(CASE WHEN throttle_pct < 0 OR throttle_pct > 105 THEN 1 ELSE 0 END) AS invalid_throttle_count,
    SUM(CASE WHEN is_braking NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_brake_count,
    SUM(CASE WHEN g_combined > 8.0 THEN 1 ELSE 0 END) AS extreme_g_count
FROM telemetry_features;

-- ------------------------------------------------------------------------------
-- 3. Gear Range Validation (1..8)
--    Forward gears in 2023 F1 hybrid power units are strictly 1 to 8
-- ------------------------------------------------------------------------------
SELECT 
    gear,
    COUNT(*) AS occurrence_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telemetry_features), 2) AS pct_of_total
FROM telemetry_features
GROUP BY gear
ORDER BY gear ASC;

-- ------------------------------------------------------------------------------
-- 4. Row Count Audit: Expected vs Actual across Relational Tables
-- ------------------------------------------------------------------------------
SELECT 'drivers' AS table_name, COUNT(*) AS row_count, 19 AS expected_count FROM drivers
UNION ALL
SELECT 'laps' AS table_name, COUNT(*) AS row_count, 96 AS expected_count FROM laps
UNION ALL
SELECT 'sectors' AS table_name, COUNT(*) AS row_count, 288 AS expected_count FROM sectors
UNION ALL
SELECT 'corners' AS table_name, COUNT(*) AS row_count, 1469 AS expected_count FROM corners
UNION ALL
SELECT 'telemetry_features' AS table_name, COUNT(*) AS row_count, 72476 AS expected_count FROM telemetry_features;

-- ------------------------------------------------------------------------------
-- 5. Foreign Key Integrity & Orphan Records Check
--    Verifies that all child rows link to an existing valid parent
-- ------------------------------------------------------------------------------
-- A. Laps without matching Driver
SELECT COUNT(*) AS orphan_laps_driver 
FROM laps l 
LEFT JOIN drivers d ON l.driver_id = d.driver_id 
WHERE d.driver_id IS NULL;

-- B. Sectors without matching Lap
SELECT COUNT(*) AS orphan_sectors_lap 
FROM sectors s 
LEFT JOIN laps l ON s.driver_id = l.driver_id AND s.lap_number = l.lap_number 
WHERE l.driver_id IS NULL;

-- C. Corners without matching Lap
SELECT COUNT(*) AS orphan_corners_lap 
FROM corners c 
LEFT JOIN laps l ON c.driver_id = l.driver_id AND c.lap_number = l.lap_number 
WHERE l.driver_id IS NULL;

-- D. Telemetry rows without matching Lap
SELECT COUNT(*) AS orphan_telemetry_lap 
FROM telemetry_features t 
LEFT JOIN laps l ON t.driver_id = l.driver_id AND t.lap_number = l.lap_number 
WHERE l.driver_id IS NULL;

-- ------------------------------------------------------------------------------
-- 6. Lap Classification Distribution
--    Ensures clean flying laps are strictly identified and accounted for
-- ------------------------------------------------------------------------------
SELECT 
    lap_classification,
    COUNT(*) AS lap_count,
    COUNT(DISTINCT driver_id) AS unique_drivers,
    ROUND(MIN(lap_time_s)::numeric, 3) AS min_lap_time,
    ROUND(AVG(lap_time_s)::numeric, 3) AS avg_lap_time,
    ROUND(MAX(lap_time_s)::numeric, 3) AS max_lap_time
FROM laps
GROUP BY lap_classification
ORDER BY lap_count DESC;
