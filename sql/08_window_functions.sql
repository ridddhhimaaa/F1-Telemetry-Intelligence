-- ==============================================================================
-- F1 Telemetry Intelligence: Phase 6 Local SQL Analytics
-- File: sql/08_window_functions.sql
-- Description: Analytical window functions including ROW_NUMBER, RANK, DENSE_RANK,
--              LAG, LEAD, moving averages, and cumulative distributions.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Lap Ranking: Rank all clean flying laps per driver and overall session rank
--    Uses ROW_NUMBER, RANK, and DENSE_RANK partitioned by classification
-- ------------------------------------------------------------------------------
SELECT 
    l.driver_id,
    d.team,
    l.lap_number,
    l.lap_time_s,
    l.lap_classification,
    -- Session overall rank among clean flying laps
    RANK() OVER (
        ORDER BY l.lap_time_s ASC
    ) AS session_lap_rank,
    -- Dense rank by team
    DENSE_RANK() OVER (
        PARTITION BY d.team 
        ORDER BY l.lap_time_s ASC
    ) AS team_lap_rank,
    -- Row number within driver laps
    ROW_NUMBER() OVER (
        PARTITION BY l.driver_id 
        ORDER BY l.lap_time_s ASC
    ) AS driver_personal_best_order
FROM laps l
JOIN drivers d ON l.driver_id = d.driver_id
WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
ORDER BY session_lap_rank ASC;

-- ------------------------------------------------------------------------------
-- 2. Telemetry Delta Tracking: Instantaneous speed deltas and acceleration using LAG / LEAD
--    Computes sample-to-sample speed delta and look-ahead acceleration
-- ------------------------------------------------------------------------------
SELECT 
    t.driver_id,
    t.lap_number,
    t.telemetry_id,
    t.distance_m,
    t.speed_kmh,
    -- Previous sample speed
    LAG(t.speed_kmh, 1) OVER (
        PARTITION BY t.driver_id, t.lap_number 
        ORDER BY t.telemetry_id
    ) AS prev_speed_kmh,
    -- Next sample speed
    LEAD(t.speed_kmh, 1) OVER (
        PARTITION BY t.driver_id, t.lap_number 
        ORDER BY t.telemetry_id
    ) AS next_speed_kmh,
    -- Speed delta from previous sample
    ROUND(
        (t.speed_kmh - LAG(t.speed_kmh, 1) OVER (
            PARTITION BY t.driver_id, t.lap_number 
            ORDER BY t.telemetry_id
        ))::numeric, 2
    ) AS delta_v_kmh,
    t.accel_longitudinal_mps2,
    t.throttle_pct,
    t.is_braking
FROM telemetry_features t
WHERE t.driver_id = 'VER' AND t.lap_number = 4
ORDER BY t.telemetry_id
LIMIT 50;

-- ------------------------------------------------------------------------------
-- 3. Rolling Smoothed Speed & Acceleration: 5-sample centered moving average
--    Calculates rolling window statistics to compare smoothed vs raw telemetry
-- ------------------------------------------------------------------------------
SELECT 
    t.driver_id,
    t.lap_number,
    t.telemetry_id,
    t.distance_m,
    t.speed_kmh AS raw_speed_kmh,
    ROUND(AVG(t.speed_kmh) OVER (
        PARTITION BY t.driver_id, t.lap_number 
        ORDER BY t.telemetry_id 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    )::numeric, 2) AS moving_avg_speed_kmh,
    t.accel_longitudinal_mps2 AS raw_accel_long,
    ROUND(AVG(t.accel_longitudinal_mps2) OVER (
        PARTITION BY t.driver_id, t.lap_number 
        ORDER BY t.telemetry_id 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    )::numeric, 3) AS moving_avg_accel_long
FROM telemetry_features t
WHERE t.driver_id = 'VER' AND t.lap_number = 4
ORDER BY t.telemetry_id
LIMIT 50;

-- ------------------------------------------------------------------------------
-- 4. Cumulative Work and Energy: Running maximum speed and cumulative throttle application
--    Uses unbounded preceding framing for progressive telemetry tracking
-- ------------------------------------------------------------------------------
SELECT 
    t.driver_id,
    t.lap_number,
    t.distance_m,
    t.speed_kmh,
    -- Maximum speed attained so far on this lap
    MAX(t.speed_kmh) OVER (
        PARTITION BY t.driver_id, t.lap_number 
        ORDER BY t.distance_m 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_max_speed_kmh,
    -- Minimum speed experienced so far on this lap
    MIN(t.speed_kmh) OVER (
        PARTITION BY t.driver_id, t.lap_number 
        ORDER BY t.distance_m 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_min_speed_kmh,
    t.drs
FROM telemetry_features t
WHERE t.driver_id = 'HAM' AND t.lap_number = 4
ORDER BY t.distance_m
LIMIT 50;

-- ------------------------------------------------------------------------------
-- 5. Percentile Distribution: NTILE and CUME_DIST of corner apex speeds
--    Divides driver apex performances into quartiles across all corners
-- ------------------------------------------------------------------------------
SELECT 
    c.corner_id,
    c.driver_id,
    c.apex_speed_kmh,
    NTILE(4) OVER (
        PARTITION BY c.corner_id 
        ORDER BY c.apex_speed_kmh DESC
    ) AS apex_speed_quartile,
    ROUND(CUME_DIST() OVER (
        PARTITION BY c.corner_id 
        ORDER BY c.apex_speed_kmh ASC
    )::numeric, 3) AS apex_speed_percentile
FROM corners c
JOIN laps l ON c.driver_id = l.driver_id AND c.lap_number = l.lap_number
WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
ORDER BY c.corner_id, apex_speed_percentile DESC;
