-- ============================================================================
-- 04_data_loading.sql
-- F1 Telemetry Intelligence - Data Ingestion Scripts
-- ============================================================================
-- Description:
--   Loads engineered features and reference datasets into PostgreSQL tables.
--
-- Execution via psql terminal:
--   psql -U postgres -d f1_telemetry -f sql/04_data_loading.sql
--
-- Note:
--   For programmatic bulk ingestion with transaction safety and type casting,
--   execute the Python database loader: python -m src.database.load_data
-- ============================================================================

-- 1. POPULATE DRIVERS REFERENCE DATA
INSERT INTO drivers (driver_id, driver_name, team, driver_number) VALUES
('VER', 'Max Verstappen', 'Red Bull Racing', 1),
('NOR', 'Lando Norris', 'McLaren', 4),
('LEC', 'Charles Leclerc', 'Ferrari', 16),
('PIA', 'Oscar Piastri', 'McLaren', 81),
('SAI', 'Carlos Sainz', 'Ferrari', 55),
('RUS', 'George Russell', 'Mercedes', 63),
('HAM', 'Lewis Hamilton', 'Mercedes', 44),
('GAS', 'Pierre Gasly', 'Alpine', 10),
('ALB', 'Alexander Albon', 'Williams', 23),
('TSU', 'Yuki Tsunoda', 'RB', 22),
('HUL', 'Nico Hulkenberg', 'Haas', 27),
('MAG', 'Kevin Magnussen', 'Haas', 20),
('ALO', 'Fernando Alonso', 'Aston Martin', 14),
('STR', 'Lance Stroll', 'Aston Martin', 18),
('OCO', 'Esteban Ocon', 'Alpine', 31),
('ZHO', 'Guanyu Zhou', 'Kick Sauber', 24),
('BOT', 'Valtteri Bottas', 'Kick Sauber', 77),
('LAW', 'Liam Lawson', 'RB', 30),
('COL', 'Franco Colapinto', 'Williams', 43),
('DOO', 'Jack Doohan', 'Alpine', 61),
('PER', 'Sergio Perez', 'Red Bull Racing', 11)
ON CONFLICT (driver_id) DO UPDATE SET
    team = EXCLUDED.team,
    driver_number = EXCLUDED.driver_number;

-- 2. POPULATE RACES REFERENCE DATA
INSERT INTO races (race_id, event_name, circuit_name, year, round_number, circuit_length_m) VALUES
('2024_abu_dhabi', 'Abu Dhabi Grand Prix', 'Yas Marina Circuit', 2024, 24, 5281.0)
ON CONFLICT (race_id) DO NOTHING;

-- 3. BULK LOADING COMMANDS VIA PSQL \copy (Client-side CSV loading)
-- Replace [PROJECT_PATH] with your local project root:
-- \copy laps FROM 'data/processed/features/lap_features.csv' WITH (FORMAT csv, HEADER true);
-- \copy sectors FROM 'data/processed/features/sector_features.csv' WITH (FORMAT csv, HEADER true);
-- \copy corners FROM 'data/processed/features/corner_features.csv' WITH (FORMAT csv, HEADER true);
-- \copy telemetry_features FROM 'data/processed/features/telemetry_features.csv' WITH (FORMAT csv, HEADER true);
