-- ============================================================================
-- 06_joins_and_aggregations.sql
-- F1 Telemetry Intelligence - Relational Joins & Multi-Level Aggregations
-- ============================================================================
-- Demonstrates:
--   INNER JOIN, LEFT JOIN, multi-table joins, cross-table metric aggregation.
-- ============================================================================

-- Query 1: Multi-Table JOIN: Driver Details, Lap Performance & Team Pace
SELECT
    d.driver_id,
    d.driver_name,
    d.team,
    l.lap_number,
    l.lap_time_s,
    l.speed_mean_kmh,
    l.speed_max_kmh,
    l.full_throttle_pct,
    l.braking_duration_s
FROM laps l
INNER JOIN drivers d ON l.driver_id = d.driver_id
WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
ORDER BY l.lap_time_s ASC;

-- Query 2: Team Performance Aggregation across all clean flying laps
SELECT
    d.team,
    COUNT(DISTINCT d.driver_id) AS active_drivers,
    COUNT(l.lap_number) AS total_flying_laps,
    ROUND(AVG(l.lap_time_s)::numeric, 3) AS team_avg_lap_time_s,
    ROUND(MIN(l.lap_time_s)::numeric, 3) AS team_best_lap_time_s,
    ROUND(AVG(l.speed_mean_kmh)::numeric, 2) AS team_avg_speed_kmh,
    ROUND(MAX(l.speed_max_kmh)::numeric, 2) AS team_top_speed_kmh,
    ROUND(AVG(l.full_throttle_pct)::numeric, 1) AS team_avg_wot_pct
FROM laps l
INNER JOIN drivers d ON l.driver_id = d.driver_id
WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
GROUP BY d.team
ORDER BY team_best_lap_time_s ASC;

-- Query 3: Sector Breakdown JOIN with Driver Metadata
SELECT
    s.sector,
    d.team,
    d.driver_id,
    ROUND(AVG(s.sector_duration_s)::numeric, 3) AS avg_sector_duration_s,
    ROUND(AVG(s.speed_mean_kmh)::numeric, 2) AS avg_speed_kmh,
    ROUND(MAX(s.speed_max_kmh)::numeric, 2) AS max_speed_kmh,
    ROUND(AVG(s.full_throttle_pct)::numeric, 1) AS avg_full_throttle_pct
FROM sectors s
INNER JOIN drivers d ON s.driver_id = d.driver_id
INNER JOIN laps l ON s.driver_id = l.driver_id AND s.lap_number = l.lap_number
WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
GROUP BY s.sector, d.team, d.driver_id
ORDER BY s.sector, avg_sector_duration_s ASC;

-- Query 4: Telemetry and Corner Kinematics JOIN for Minimum Apex Speed Benchmarking
SELECT
    c.corner_id,
    c.turn_direction,
    ROUND(AVG(c.entry_speed_kmh)::numeric, 2) AS avg_entry_speed_kmh,
    ROUND(AVG(c.apex_speed_kmh)::numeric, 2) AS avg_apex_speed_kmh,
    ROUND(AVG(c.exit_speed_kmh)::numeric, 2) AS avg_exit_speed_kmh,
    ROUND(AVG(c.speed_loss_kmh)::numeric, 2) AS avg_speed_loss_kmh,
    ROUND(AVG(c.exit_accel_kmh)::numeric, 2) AS avg_exit_accel_kmh,
    ROUND(MIN(c.min_radius_m)::numeric, 1) AS min_radius_m
FROM corners c
INNER JOIN laps l ON c.driver_id = l.driver_id AND c.lap_number = l.lap_number
WHERE l.lap_classification = 'COMPLETE_FLYING_LAP'
GROUP BY c.corner_id, c.turn_direction
ORDER BY c.corner_id ASC;
