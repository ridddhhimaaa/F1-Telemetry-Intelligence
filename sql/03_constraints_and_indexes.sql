-- ============================================================================
-- 03_constraints_and_indexes.sql
-- F1 Telemetry Intelligence - Integrity Constraints & Query Optimization
-- ============================================================================
-- Description:
--   Establishes physical check constraints and targeted composite indexes
--   to accelerate analytical queries while preventing index bloating on high-volume tables.
-- ============================================================================

-- 1. INTEGRITY CHECK CONSTRAINTS
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_race_year_round') THEN
        ALTER TABLE races ADD CONSTRAINT uq_race_year_round UNIQUE (year, round_number);
    END IF;
END $$;

ALTER TABLE laps
    ADD CONSTRAINT check_lap_speed_positive CHECK (speed_min_kmh >= 0.0),
    ADD CONSTRAINT check_lap_throttle_range CHECK (full_throttle_pct >= 0.0 AND full_throttle_pct <= 100.0),
    ADD CONSTRAINT check_lap_gear_range CHECK (most_used_gear >= 0 AND most_used_gear <= 8);

ALTER TABLE sectors
    ADD CONSTRAINT check_sector_id_range CHECK (sector IN (1, 2, 3)),
    ADD CONSTRAINT check_sector_speed_positive CHECK (speed_min_kmh >= 0.0);

ALTER TABLE telemetry_features
    ADD CONSTRAINT check_tel_speed_positive CHECK (speed_kmh >= 0.0),
    ADD CONSTRAINT check_tel_throttle_bounds CHECK (throttle_pct >= 0.0 AND throttle_pct <= 100.0),
    ADD CONSTRAINT check_tel_gear_bounds CHECK (gear >= 0 AND gear <= 8);

-- 2. TARGETED ANALYTICAL INDEXES

-- Index for Driver & Lap queries across all analytical JOINs
CREATE INDEX IF NOT EXISTS idx_laps_driver_class
    ON laps (driver_id, lap_classification);

CREATE INDEX IF NOT EXISTS idx_sectors_driver_lap
    ON sectors (driver_id, lap_number, sector);

CREATE INDEX IF NOT EXISTS idx_corners_corner_id
    ON corners (corner_id, driver_id, lap_number);

-- High-volume telemetry indexes: composite (driver, lap, distance) for spatial trajectory lookups
CREATE INDEX IF NOT EXISTS idx_telemetry_driver_lap_dist
    ON telemetry_features (driver_id, lap_number, distance_m);

-- Index for filtering primary clean flying lap cohort
CREATE INDEX IF NOT EXISTS idx_telemetry_classification
    ON telemetry_features (lap_classification)
    WHERE lap_classification = 'COMPLETE_FLYING_LAP';

-- Index for corner kinematics and apex analysis
CREATE INDEX IF NOT EXISTS idx_telemetry_corner_apex
    ON telemetry_features (corner_id, is_corner_apex)
    WHERE corner_id > 0;
