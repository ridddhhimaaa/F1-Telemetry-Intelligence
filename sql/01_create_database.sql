-- ============================================================================
-- 01_create_database.sql
-- F1 Telemetry Intelligence - Phase 6 SQL Analytics
-- ============================================================================
-- Description:
--   Creates the target relational database 'f1_telemetry' if it does not already exist.
--   Enforces UTF-8 encoding and standard connection configuration.
--
-- Execution:
--   psql -U postgres -f sql/01_create_database.sql
--   or execute in VS Code SQL extension connected to the default 'postgres' database.
-- ============================================================================

SELECT 'Creating database f1_telemetry if not exists...' AS status;

-- Check existence before creation to avoid destructive errors
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'f1_telemetry' AND pid <> pg_backend_pid();

DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'f1_telemetry') THEN
      PERFORM dblink_exec('dbname=' || current_database(), 'CREATE DATABASE f1_telemetry WITH ENCODING = ''UTF8'' TEMPLATE = template0');
   END IF;
END
$do$;

-- Alternative standard creation (for interactive psql execution)
-- CREATE DATABASE f1_telemetry WITH ENCODING = 'UTF8';
