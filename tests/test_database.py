"""
Unit Tests for Database Connection, Ingestion, and Query Runner Modules
"""

import unittest
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from src.database.connection import get_engine, is_postgres_available
from src.database.load_data import load_drivers_and_races
from src.database.query_runner import run_query
from src.database.verify_sql_vs_pandas import run_all_cross_validations

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDatabaseIntegration(unittest.TestCase):
    """Test suite for local database interactions."""

    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine()

    def test_postgres_availability_check(self):
        """Verify is_postgres_available returns a boolean value."""
        avail = is_postgres_available(timeout_sec=1)
        self.assertIsInstance(avail, bool)

    def test_engine_creation(self):
        """Verify get_engine produces a valid operational engine."""
        self.assertIsNotNone(self.engine)
        with self.engine.connect() as conn:
            val = conn.execute(text("SELECT 1;")).scalar()
            self.assertEqual(val, 1)

    def test_database_table_row_counts(self):
        """Verify populated relational tables have expected row counts."""
        with self.engine.connect() as conn:
            races_cnt = conn.execute(text("SELECT COUNT(*) FROM races;")).scalar()
            drivers_cnt = conn.execute(text("SELECT COUNT(*) FROM drivers;")).scalar()
            laps_cnt = conn.execute(text("SELECT COUNT(*) FROM laps;")).scalar()
            sectors_cnt = conn.execute(text("SELECT COUNT(*) FROM sectors;")).scalar()
            corners_cnt = conn.execute(text("SELECT COUNT(*) FROM corners;")).scalar()
            tel_cnt = conn.execute(text("SELECT COUNT(*) FROM telemetry_features;")).scalar()
            
            self.assertEqual(races_cnt, 1)
            self.assertEqual(drivers_cnt, 21)
            self.assertEqual(laps_cnt, 96)
            self.assertEqual(sectors_cnt, 288)
            self.assertEqual(corners_cnt, 1469)
            self.assertEqual(tel_cnt, 72476)

    def test_race_identifier_standardization(self):
        """Verify races table has standardized 2024_abu_dhabi identifier without legacy duplicates."""
        with self.engine.connect() as conn:
            races = conn.execute(text("SELECT race_id, event_name, year, round_number FROM races;")).fetchall()
            self.assertEqual(len(races), 1)
            self.assertEqual(races[0][0], "2024_abu_dhabi")
            self.assertEqual(races[0][1], "Abu Dhabi Grand Prix")
            self.assertEqual(races[0][2], 2024)
            self.assertEqual(races[0][3], 24)
            
            # Verify legacy identifier does not exist
            legacy_cnt = conn.execute(text("SELECT COUNT(*) FROM races WHERE race_id = 'ABU_DHABI_2024';")).scalar()
            self.assertEqual(legacy_cnt, 0)

    def test_cannot_insert_duplicate_race_under_different_identifier(self):
        """Verify database rejects duplicate race insertion for the same year and round."""
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO races (race_id, event_name, circuit_name, year, round_number, circuit_length_m) "
                    "VALUES ('ABU_DHABI_2024', 'Abu Dhabi Grand Prix', 'Yas Marina Circuit', 2024, 24, 5281.0);"
                ))

    def test_loader_idempotency_preserves_single_race(self):
        """Verify re-running load_drivers_and_races maintains exactly one race record."""
        load_drivers_and_races(self.engine)
        with self.engine.connect() as conn:
            races_cnt = conn.execute(text("SELECT COUNT(*) FROM races;")).scalar()
            self.assertEqual(races_cnt, 1)
            canonical_id = conn.execute(text("SELECT race_id FROM races;")).scalar()
            self.assertEqual(canonical_id, "2024_abu_dhabi")

    def test_run_query_execution(self):
        """Verify run_query executes successfully and records execution duration."""
        df, elapsed = run_query("SELECT driver_id, lap_number, lap_time_s FROM laps LIMIT 5;", engine=self.engine)
        self.assertEqual(len(df), 5)
        self.assertIn("driver_id", df.columns)
        self.assertGreater(elapsed, 0.0)

    def test_all_cross_validations(self):
        """Verify that all 5 SQL vs Pandas mathematical validations pass."""
        passed = run_all_cross_validations(engine=self.engine)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
