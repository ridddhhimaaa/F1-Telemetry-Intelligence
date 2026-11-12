"""
Unit Tests for Apache Spark Session & Runtime Integration
Milestone 7.1: Focused Spark Initialization, Master, and Synthetic Aggregation Tests
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPARK_PYTHON = Path(r"C:\Spark\spark-env\Scripts\python.exe")


class TestSparkSessionNative(unittest.TestCase):
    """
    In-process Spark test suite executed directly within the dedicated Python 3.13.5 environment.
    """

    @classmethod
    def setUpClass(cls):
        # Only run in-process if we are running under the dedicated Python 3.13.5 runtime
        if Path(sys.executable).resolve() != SPARK_PYTHON.resolve():
            raise unittest.SkipTest("Not running inside dedicated Spark Python environment; handled via delegation.")
        
        from src.spark.session import get_spark_session, stop_spark_session
        stop_spark_session()  # Ensure clean starting state
        cls.spark = get_spark_session(
            app_name="F1_Milestone_7_1_SmokeTest",
            local_cores=2,
            driver_memory="1536m",
            executor_memory="1536m",
            force_new=True,
        )

    @classmethod
    def tearDownClass(cls):
        if Path(sys.executable).resolve() == SPARK_PYTHON.resolve():
            from src.spark.session import stop_spark_session
            stop_spark_session()

    def get_active_spark(self):
        """Helper to ensure we have a valid, active SparkSession."""
        from src.spark.session import get_spark_session
        return get_spark_session(
            app_name="F1_Milestone_7_1_SmokeTest",
            local_cores=2,
            driver_memory="1536m",
            executor_memory="1536m",
        )

    def test_01_python_runtime_is_3_13_5(self):
        """Verify the active Spark Python runtime is 3.13.5."""
        self.assertEqual(sys.version_info.major, 3)
        self.assertEqual(sys.version_info.minor, 13)
        self.assertEqual(sys.version_info.micro, 5)

    def test_02_pyspark_version_is_3_5_9(self):
        """Verify PySpark version is exactly 3.5.9."""
        import pyspark
        self.assertEqual(pyspark.__version__, "3.5.9")

    def test_03_spark_session_initialized_with_local2(self):
        """Verify SparkSession initialized with local[2] master and conservative parallelism."""
        spark = self.get_active_spark()
        self.assertIsNotNone(spark)
        sc = spark.sparkContext
        self.assertEqual(sc.master, "local[2]")
        self.assertEqual(sc.appName, "F1_Milestone_7_1_SmokeTest")
        self.assertEqual(spark.conf.get("spark.sql.shuffle.partitions"), "4")
        self.assertEqual(sc.defaultParallelism, 4)

    def test_04_synthetic_dataframe_and_aggregation(self):
        """Verify DataFrame creation, count(), show(), and simple aggregation."""
        spark = self.get_active_spark()
        data = [
            ("VER", 1, 325.4, 98.5),
            ("NOR", 1, 321.2, 95.0),
            ("HAM", 1, 319.8, 92.3),
            ("LEC", 1, 323.0, 96.2),
        ]
        columns = ["driver", "lap", "speed_kmh", "throttle_pct"]
        
        df = spark.createDataFrame(data, columns)
        
        # Action 1: count()
        self.assertEqual(df.count(), 4)
        
        # Action 2: show()
        df.show(5)
        
        # Action 3: aggregation (average and maximum speed)
        from pyspark.sql import functions as F
        agg_res = df.agg(
            F.round(F.avg("speed_kmh"), 2).alias("avg_speed"),
            F.round(F.max("speed_kmh"), 2).alias("max_speed"),
        ).collect()[0]
        
        self.assertAlmostEqual(float(agg_res["avg_speed"]), 322.35, places=2)
        self.assertAlmostEqual(float(agg_res["max_speed"]), 325.40, places=2)

    def test_05_spark_diagnostics_structure(self):
        """Verify get_spark_diagnostics returns runtime-verified values."""
        from src.spark.session import get_spark_diagnostics
        spark = self.get_active_spark()
        diag = get_spark_diagnostics(spark)
        
        self.assertTrue(diag["spark_session_live"])
        self.assertTrue(diag["python_runtime"]["is_dedicated_spark_python"])
        
        verified = diag["runtime_verified"]
        self.assertEqual(verified["spark_version"], "3.5.9")
        self.assertEqual(verified["master"], "local[2]")
        self.assertEqual(verified["shuffle_partitions"], "4")
        self.assertEqual(verified["driver_memory_conf"], "1536m")
        self.assertIn("21", verified["java_version_jvm"])
        self.assertGreater(verified["jvm_available_cores"], 0)
        self.assertGreater(verified["jvm_max_memory_mb"], 1000.0)

    def test_06_spark_session_reuse_and_shutdown(self):
        """Verify get_spark_session reuses active session and stop_spark_session terminates cleanly."""
        from src.spark.session import get_spark_session, stop_spark_session
        
        spark1 = get_spark_session()
        spark2 = get_spark_session()
        self.assertIs(spark1, spark2)
        
        # Stop session
        stop_spark_session()
        # After stop, sparkContext._jsc is set to None in PySpark
        self.assertIsNone(spark1.sparkContext._jsc)
        
        # Calling get_spark_session again should successfully spin up a new session
        spark3 = get_spark_session(local_cores=2, force_new=True)
        self.assertIsNotNone(spark3)
        self.assertIsNotNone(spark3.sparkContext._jsc)


class TestSparkSessionRunnerDelegation(unittest.TestCase):
    """
    Test case designed to run when tests are executed from the main project venv (Python 3.14).
    Delegates smoke-test execution to the dedicated Python 3.13.5 Spark environment via subprocess.
    """

    def test_spark_smoke_suite_via_dedicated_runtime(self):
        """Run the full TestSparkSessionNative suite using C:\\Spark\\spark-env\\Scripts\\python.exe."""
        self.assertTrue(
            SPARK_PYTHON.exists(),
            f"Dedicated Python executable does not exist at '{SPARK_PYTHON}'",
        )
        
        cmd = [
            str(SPARK_PYTHON),
            "-m",
            "unittest",
            "tests.test_spark_session.TestSparkSessionNative",
            "-v",
        ]
        
        env = os.environ.copy()
        env["PYSPARK_PYTHON"] = str(SPARK_PYTHON)
        env["PYSPARK_DRIVER_PYTHON"] = str(SPARK_PYTHON)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        # Print subprocess output if it failed
        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            
        self.assertEqual(
            result.returncode,
            0,
            f"Spark smoke test failed in dedicated runtime with exit code {result.returncode}",
        )
        self.assertIn("OK", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
