-- ==============================================================================
-- F1 Telemetry Intelligence: Phase 6 Local SQL Analytics
-- File: sql/09_advanced_telemetry_analysis.sql
-- Description: Advanced telemetry domain analytics: head-to-head comparisons,
--              speed trap rankings, corner apex delta matrix, braking signatures,
--              and strictly regime-separated sector pace deficits.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Maximum Speed Point & Speed Trap Analysis (Clean Flying Laps Only)
--    Identifies the exact track location and gear for peak speeds per driver
-- ------------------------------------------------------------------------------
WITH RankedSpeeds AS (
    SELECT 
        t.driver_id,
        d.team,
        t.lap_number,
        t.distance_m,
        t.speed_kmh,
        t.gear,
        t.drs,
        t.throttle_pct,
        ROW_NUMBER() OVER (
            PARTITION BY t.driver_id 
            ORDER BY t.speed_kmh DESC
        ) AS speed_rank
    FROM telemetry_features t
    JOIN drivers d ON t.driver_id = d.driver_id
    JOIN laps l ON t.driver_id = l.driver_id AND t.lap_number = l.lap_number
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
)
SELECT 
    driver_id,
    team,
    lap_number,
    ROUND(distance_m::numeric, 1) AS trap_distance_m,
    speed_kmh AS max_speed_kmh,
    gear,
    drs,
    throttle_pct
FROM RankedSpeeds
WHERE speed_rank = 1
ORDER BY max_speed_kmh DESC;

-- ------------------------------------------------------------------------------
-- 2. Corner Apex Delta Benchmark: Each driver's apex speed vs session benchmark
--    Evaluates minimum speed preservation through medium and high curvature turns
-- ------------------------------------------------------------------------------
WITH CornerBenchmarks AS (
    SELECT 
        c.corner_id,
        MAX(c.apex_speed_kmh) AS benchmark_apex_speed_kmh,
        ROUND(AVG(c.apex_speed_kmh)::numeric, 2) AS field_avg_apex_speed_kmh
    FROM corners c
    JOIN laps l ON c.driver_id = l.driver_id AND c.lap_number = l.lap_number
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY c.corner_id
),
DriverBestCorners AS (
    SELECT 
        c.corner_id,
        c.driver_id,
        MAX(c.apex_speed_kmh) AS driver_apex_speed_kmh,
        MIN(c.speed_loss_kmh) AS driver_min_speed_loss
    FROM corners c
    JOIN laps l ON c.driver_id = l.driver_id AND c.lap_number = l.lap_number
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY c.corner_id, c.driver_id
)
SELECT 
    dbc.corner_id,
    dbc.driver_id,
    dbc.driver_apex_speed_kmh,
    cb.benchmark_apex_speed_kmh AS benchmark_apex_speed_kmh,
    ROUND((dbc.driver_apex_speed_kmh - cb.benchmark_apex_speed_kmh)::numeric, 2) AS apex_delta_to_benchmark,
    ROUND((dbc.driver_apex_speed_kmh - cb.field_avg_apex_speed_kmh)::numeric, 2) AS apex_delta_to_average,
    dbc.driver_min_speed_loss
FROM DriverBestCorners dbc
JOIN CornerBenchmarks cb ON dbc.corner_id = cb.corner_id
WHERE dbc.driver_id IN ('VER', 'HAM', 'LEC', 'NOR', 'ALO')
ORDER BY dbc.corner_id ASC, dbc.driver_apex_speed_kmh DESC;

-- ------------------------------------------------------------------------------
-- 3. Sector-Level Deficits Without Regime Mixing
--    Computes sector deltas against the fastest sector in clean flying conditions only
-- ------------------------------------------------------------------------------
WITH CleanSectorBenchmarks AS (
    SELECT 
        s.sector,
        MIN(s.sector_duration_s) AS best_sector_time_seconds,
        ROUND(AVG(s.sector_duration_s)::numeric, 3) AS avg_sector_time_seconds
    FROM sectors s
    JOIN laps l ON s.driver_id = l.driver_id AND s.lap_number = l.lap_number
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY s.sector
),
DriverPersonalBestSectors AS (
    SELECT 
        s.sector,
        s.driver_id,
        d.team,
        MIN(s.sector_duration_s) AS personal_best_sector_time,
        ROUND(AVG(s.speed_mean_kmh)::numeric, 2) AS avg_speed_kmh,
        MAX(s.speed_max_kmh) AS max_speed_kmh,
        ROUND(AVG(s.full_throttle_pct)::numeric, 2) AS avg_wot_pct
    FROM sectors s
    JOIN drivers d ON s.driver_id = d.driver_id
    JOIN laps l ON s.driver_id = l.driver_id AND s.lap_number = l.lap_number
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY s.sector, s.driver_id, d.team
)
SELECT 
    dpb.sector,
    dpb.driver_id,
    dpb.team,
    dpb.personal_best_sector_time,
    csb.best_sector_time_seconds AS benchmark_sector_time,
    ROUND((dpb.personal_best_sector_time - csb.best_sector_time_seconds)::numeric, 3) AS deficit_seconds,
    dpb.avg_speed_kmh,
    dpb.max_speed_kmh,
    dpb.avg_wot_pct
FROM DriverPersonalBestSectors dpb
JOIN CleanSectorBenchmarks csb ON dpb.sector = csb.sector
ORDER BY dpb.sector ASC, deficit_seconds ASC;

-- ------------------------------------------------------------------------------
-- 4. Head-to-Head Telemetry Comparison: VER vs HAM on Best Clean Flying Lap
--    Computes telemetry dynamics differences across distance bins (100m slices)
-- ------------------------------------------------------------------------------
WITH SlicedTelemetry AS (
    SELECT 
        t.driver_id,
        FLOOR(t.distance_m / 100.0) * 100 AS distance_slice_m,
        AVG(t.speed_kmh) AS mean_speed_kmh,
        AVG(t.throttle_pct) AS mean_throttle_pct,
        AVG(t.is_braking) AS mean_brake_flag,
        AVG(t.accel_longitudinal_mps2) AS mean_accel_long
    FROM telemetry_features t
    JOIN laps l ON t.driver_id = l.driver_id AND t.lap_number = l.lap_number
    WHERE t.driver_id IN ('VER', 'HAM')
      AND l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY t.driver_id, FLOOR(t.distance_m / 100.0) * 100
)
SELECT 
    v.distance_slice_m,
    ROUND(v.mean_speed_kmh::numeric, 2) AS ver_speed_kmh,
    ROUND(h.mean_speed_kmh::numeric, 2) AS ham_speed_kmh,
    ROUND((v.mean_speed_kmh - h.mean_speed_kmh)::numeric, 2) AS delta_speed_kmh,
    ROUND(v.mean_throttle_pct::numeric, 1) AS ver_throttle_pct,
    ROUND(h.mean_throttle_pct::numeric, 1) AS ham_throttle_pct,
    ROUND(v.mean_accel_long::numeric, 2) AS ver_accel_long,
    ROUND(h.mean_accel_long::numeric, 2) AS ham_accel_long
FROM SlicedTelemetry v
JOIN SlicedTelemetry h ON v.distance_slice_m = h.distance_slice_m AND h.driver_id = 'HAM'
WHERE v.driver_id = 'VER'
ORDER BY v.distance_slice_m ASC
LIMIT 40;
