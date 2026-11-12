-- ============================================================================
-- 05_basic_analysis.sql
-- F1 Telemetry Intelligence - Basic SQL Analytics
-- ============================================================================
-- Demonstrates:
--   SELECT, WHERE, ORDER BY, GROUP BY, HAVING, DISTINCT, CASE WHEN, NULL handling.
-- ============================================================================

-- Query 1: Distinct drivers and lap classifications present in the session
SELECT DISTINCT
    driver_id,
    lap_classification,
    COUNT(*) AS completed_laps
FROM laps
GROUP BY driver_id, lap_classification
ORDER BY lap_classification, driver_id;

-- Query 2: Clean Flying Laps ranking by lap time (WHERE, ORDER BY)
SELECT
    driver_id,
    lap_number,
    lap_time_s,
    speed_mean_kmh,
    speed_max_kmh,
    full_throttle_pct
FROM laps
WHERE lap_classification = 'COMPLETE_FLYING_LAP'
    AND lap_time_s IS NOT NULL
ORDER BY lap_time_s ASC;

-- Query 3: Driver average flying lap pace with HAVING filter (GROUP BY, HAVING)
SELECT
    driver_id,
    COUNT(*) AS flying_laps_count,
    ROUND(AVG(lap_time_s)::numeric, 3) AS avg_lap_time_s,
    ROUND(AVG(speed_mean_kmh)::numeric, 2) AS avg_speed_kmh,
    ROUND(MAX(speed_max_kmh)::numeric, 2) AS top_speed_kmh
FROM laps
WHERE lap_classification = 'COMPLETE_FLYING_LAP'
GROUP BY driver_id
HAVING COUNT(*) >= 2
ORDER BY avg_lap_time_s ASC;

-- Query 4: Categorizing cornering speeds using CASE WHEN
SELECT
    corner_id,
    turn_direction,
    ROUND(AVG(apex_speed_kmh)::numeric, 2) AS avg_apex_speed_kmh,
    ROUND(MIN(min_radius_m)::numeric, 1) AS min_radius_m,
    CASE
        WHEN AVG(apex_speed_kmh) < 100.0 THEN 'SLOW_HAIRPIN_CHICANE'
        WHEN AVG(apex_speed_kmh) BETWEEN 100.0 AND 180.0 THEN 'MEDIUM_SPEED_CORNER'
        ELSE 'HIGH_SPEED_SWEEPER'
    END AS corner_severity_class
FROM corners
GROUP BY corner_id, turn_direction
ORDER BY corner_id;

-- Query 5: Telemetry dynamic regime distribution with NULL safe handling
SELECT
    COALESCE(dynamics_regime, 'UNKNOWN') AS dynamics_regime,
    COUNT(*) AS sample_count,
    ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER())::numeric, 2) AS share_percentage,
    ROUND(AVG(speed_kmh)::numeric, 2) AS avg_speed_kmh,
    ROUND(AVG(accel_longitudinal_mps2)::numeric, 2) AS avg_accel_mps2
FROM telemetry_features
WHERE lap_classification = 'COMPLETE_FLYING_LAP'
GROUP BY dynamics_regime
ORDER BY sample_count DESC;
