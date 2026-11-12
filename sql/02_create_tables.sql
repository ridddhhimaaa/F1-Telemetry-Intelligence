-- ============================================================================
-- 02_create_tables.sql
-- F1 Telemetry Intelligence - Normalized Relational Schema
-- ============================================================================
-- Description:
--   Creates the normalized relational schema for drivers, races, laps,
--   timing sectors, corner performance events, and engineered telemetry.
--
-- Target Database: f1_telemetry
-- ============================================================================

-- 1. DRIVERS TABLE
CREATE TABLE IF NOT EXISTS drivers (
    driver_id VARCHAR(3) PRIMARY KEY,
    driver_name VARCHAR(100),
    team VARCHAR(50),
    driver_number INTEGER
);

-- 2. RACES TABLE
CREATE TABLE IF NOT EXISTS races (
    race_id VARCHAR(30) PRIMARY KEY,
    event_name VARCHAR(100) NOT NULL,
    circuit_name VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL,
    round_number INTEGER,
    circuit_length_m DOUBLE PRECISION DEFAULT 5281.0,
    CONSTRAINT uq_race_year_round UNIQUE (year, round_number)
);

-- 3. LAPS TABLE
CREATE TABLE IF NOT EXISTS laps (
    driver_id VARCHAR(3) NOT NULL REFERENCES drivers(driver_id) ON DELETE CASCADE,
    lap_number INTEGER NOT NULL,
    lap_classification VARCHAR(30) NOT NULL,
    lap_time_s DOUBLE PRECISION,
    telemetry_duration_s DOUBLE PRECISION,
    lap_distance_m DOUBLE PRECISION,
    speed_mean_kmh DOUBLE PRECISION,
    speed_max_kmh DOUBLE PRECISION,
    speed_min_kmh DOUBLE PRECISION,
    speed_std_kmh DOUBLE PRECISION,
    speed_median_kmh DOUBLE PRECISION,
    full_throttle_pct DOUBLE PRECISION,
    throttle_mean_pct DOUBLE PRECISION,
    throttle_transitions INTEGER,
    braking_duration_s DOUBLE PRECISION,
    braking_pct DOUBLE PRECISION,
    braking_events_count INTEGER,
    peak_decel_g DOUBLE PRECISION,
    peak_accel_g DOUBLE PRECISION,
    mean_accel_mps2 DOUBLE PRECISION,
    gear_changes_count INTEGER,
    upshifts_count INTEGER,
    downshifts_count INTEGER,
    most_used_gear INTEGER,
    corners_count INTEGER,
    mean_apex_speed_kmh DOUBLE PRECISION,
    min_apex_speed_kmh DOUBLE PRECISION,
    sector1_duration_s DOUBLE PRECISION,
    sector2_duration_s DOUBLE PRECISION,
    sector3_duration_s DOUBLE PRECISION,
    sample_count INTEGER,
    sampling_frequency_hz DOUBLE PRECISION,
    PRIMARY KEY (driver_id, lap_number)
);

-- 4. SECTORS TABLE
CREATE TABLE IF NOT EXISTS sectors (
    driver_id VARCHAR(3) NOT NULL,
    lap_number INTEGER NOT NULL,
    sector INTEGER NOT NULL,
    sector_duration_s DOUBLE PRECISION,
    speed_mean_kmh DOUBLE PRECISION,
    speed_max_kmh DOUBLE PRECISION,
    speed_min_kmh DOUBLE PRECISION,
    full_throttle_pct DOUBLE PRECISION,
    throttle_mean_pct DOUBLE PRECISION,
    braking_duration_s DOUBLE PRECISION,
    peak_decel_g DOUBLE PRECISION,
    most_used_gear INTEGER,
    gear_shifts_count INTEGER,
    corners_count INTEGER,
    samples_count INTEGER,
    PRIMARY KEY (driver_id, lap_number, sector),
    FOREIGN KEY (driver_id, lap_number) REFERENCES laps(driver_id, lap_number) ON DELETE CASCADE
);

-- 5. CORNERS TABLE
CREATE TABLE IF NOT EXISTS corners (
    corner_record_id SERIAL PRIMARY KEY,
    driver_id VARCHAR(3) NOT NULL,
    lap_number INTEGER NOT NULL,
    corner_id INTEGER NOT NULL,
    turn_direction VARCHAR(10),
    start_distance_m DOUBLE PRECISION,
    apex_distance_m DOUBLE PRECISION,
    end_distance_m DOUBLE PRECISION,
    corner_length_m DOUBLE PRECISION,
    entry_speed_kmh DOUBLE PRECISION,
    apex_speed_kmh DOUBLE PRECISION,
    exit_speed_kmh DOUBLE PRECISION,
    speed_loss_kmh DOUBLE PRECISION,
    exit_accel_kmh DOUBLE PRECISION,
    corner_duration_s DOUBLE PRECISION,
    min_radius_m DOUBLE PRECISION,
    FOREIGN KEY (driver_id, lap_number) REFERENCES laps(driver_id, lap_number) ON DELETE CASCADE
);

-- 6. TELEMETRY FEATURES TABLE (High-Frequency Instantaneous Samples)
CREATE TABLE IF NOT EXISTS telemetry_features (
    telemetry_id BIGSERIAL PRIMARY KEY,
    driver_id VARCHAR(3) NOT NULL,
    lap_number INTEGER NOT NULL,
    time_seconds DOUBLE PRECISION NOT NULL,
    distance_m DOUBLE PRECISION NOT NULL,
    speed_kmh DOUBLE PRECISION NOT NULL,
    rpm INTEGER,
    gear INTEGER,
    throttle_pct DOUBLE PRECISION,
    is_braking INTEGER,
    drs INTEGER,
    lap_classification VARCHAR(30),
    x_m DOUBLE PRECISION,
    y_m DOUBLE PRECISION,
    heading_deg DOUBLE PRECISION,
    curvature_m_inv DOUBLE PRECISION,
    radius_of_curvature_m DOUBLE PRECISION,
    corner_id INTEGER,
    is_corner_zone INTEGER,
    is_corner_apex INTEGER,
    turn_direction VARCHAR(10),
    accel_longitudinal_mps2 DOUBLE PRECISION,
    g_longitudinal DOUBLE PRECISION,
    accel_lateral_mps2 DOUBLE PRECISION,
    g_lateral DOUBLE PRECISION,
    g_combined DOUBLE PRECISION,
    is_accelerating INTEGER,
    is_decelerating INTEGER,
    is_coasting INTEGER,
    dynamics_regime VARCHAR(30),
    accel_phase_duration_s DOUBLE PRECISION,
    decel_phase_duration_s DOUBLE PRECISION,
    throttle_application_rate DOUBLE PRECISION,
    throttle_release_rate DOUBLE PRECISION,
    is_full_throttle INTEGER,
    full_throttle_duration_s DOUBLE PRECISION,
    full_throttle_cumulative_pct DOUBLE PRECISION,
    throttle_transition_count INTEGER,
    braking_event_id INTEGER,
    is_braking_onset INTEGER,
    braking_duration_s DOUBLE PRECISION,
    braking_distance_m DOUBLE PRECISION,
    brake_event_peak_g DOUBLE PRECISION,
    is_brake_release_phase INTEGER,
    is_pedal_overlap INTEGER,
    ratio_rpm_per_kmh DOUBLE PRECISION,
    rpm_change_rate_per_s DOUBLE PRECISION,
    is_gear_shift INTEGER,
    is_upshift INTEGER,
    is_downshift INTEGER,
    gear_change_count INTEGER,
    sector INTEGER,
    distance_in_sector_m DOUBLE PRECISION,
    FOREIGN KEY (driver_id, lap_number) REFERENCES laps(driver_id, lap_number) ON DELETE CASCADE
);
