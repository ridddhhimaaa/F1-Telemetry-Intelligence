-- ============================================================================
-- 07_cte_analysis.sql
-- F1 Telemetry Intelligence - Common Table Expressions (CTEs)
-- ============================================================================
-- Demonstrates:
--   Single and Multiple CTEs, CTE chaining, benchmark deltas, quality auditing.
-- ============================================================================

-- Query 1: Session Pace Benchmark Delta using Chained CTEs
-- Identifies the fastest lap of the session (benchmark) and computes every driver's pace deficit.
WITH session_benchmark AS (
    SELECT
        driver_id,
        lap_number,
        lap_time_s AS benchmark_lap_time_s
    FROM laps
    WHERE lap_classification = 'COMPLETE_FLYING_LAP'
    ORDER BY lap_time_s ASC
    LIMIT 1
),
driver_best_laps AS (
    SELECT
        l.driver_id,
        d.team,
        MIN(l.lap_time_s) AS best_lap_time_s,
        MAX(l.speed_max_kmh) AS top_speed_kmh,
        AVG(l.full_throttle_pct) AS avg_wot_pct
    FROM laps l
    INNER JOIN drivers d ON l.driver_id = d.driver_id
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY l.driver_id, d.team
)
SELECT
    b.driver_id,
    b.team,
    b.best_lap_time_s,
    ROUND((b.best_lap_time_s - s.benchmark_lap_time_s)::numeric, 3) AS delta_to_benchmark_s,
    ROUND((((b.best_lap_time_s - s.benchmark_lap_time_s) / s.benchmark_lap_time_s) * 100.0)::numeric, 2) AS deficit_pct,
    b.top_speed_kmh,
    ROUND(b.avg_wot_pct::numeric, 1) AS avg_wot_pct
FROM driver_best_laps b
CROSS JOIN session_benchmark s
ORDER BY b.best_lap_time_s ASC;

-- Query 2: Sector-by-Sector Dominance Analysis using CTEs
-- Determines which driver set the fastest theoretical sector time in each sector.
WITH sector_bests AS (
    SELECT
        sector,
        MIN(sector_duration_s) AS best_sector_time_s
    FROM sectors s
    INNER JOIN laps l ON s.driver_id = l.driver_id AND s.lap_number = l.lap_number
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY sector
),
sector_driver_ranks AS (
    SELECT
        s.sector,
        s.driver_id,
        s.lap_number,
        s.sector_duration_s,
        sb.best_sector_time_s,
        ROUND((s.sector_duration_s - sb.best_sector_time_s)::numeric, 3) AS gap_to_best_s
    FROM sectors s
    INNER JOIN laps l ON s.driver_id = l.driver_id AND s.lap_number = l.lap_number
    INNER JOIN sector_bests sb ON s.sector = sb.sector
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
)
SELECT
    sector,
    driver_id,
    lap_number,
    sector_duration_s,
    best_sector_time_s,
    gap_to_best_s
FROM sector_driver_ranks
WHERE gap_to_best_s = 0.000
ORDER BY sector;

-- Query 3: Multi-CTE Braking Intensity & Deceleration Energy Audit
WITH braking_stats AS (
    SELECT
        l.driver_id,
        d.team,
        COUNT(*) AS total_braking_zones,
        ROUND(AVG(l.braking_duration_s)::numeric, 2) AS avg_braking_time_s,
        ROUND(AVG(l.peak_decel_g)::numeric, 2) AS avg_peak_decel_g,
        ROUND(MAX(l.peak_decel_g)::numeric, 2) AS max_decel_g
    FROM laps l
    INNER JOIN drivers d ON l.driver_id = d.driver_id
    WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
    GROUP BY l.driver_id, d.team
),
fleet_average AS (
    SELECT
        AVG(avg_peak_decel_g) AS fleet_avg_decel_g,
        AVG(avg_braking_time_s) AS fleet_avg_braking_time_s
    FROM braking_stats
)
SELECT
    b.driver_id,
    b.team,
    b.avg_braking_time_s,
    b.avg_peak_decel_g,
    b.max_decel_g,
    ROUND((b.avg_peak_decel_g - f.fleet_avg_decel_g)::numeric, 2) AS decel_g_vs_fleet
FROM braking_stats b
CROSS JOIN fleet_average f
ORDER BY b.avg_peak_decel_g DESC;
