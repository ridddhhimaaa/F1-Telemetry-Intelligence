"""
F1 Telemetry Intelligence - Apache Spark Session & Runtime Integration
Milestone 7.1: Reusable SparkSession Factory & Environment Diagnostics

This module provides a managed, conservative SparkSession factory configured
specifically for the Windows local multi-core environment using the verified
Python 3.13.5 PySpark runtime at C:\\Spark\\spark-env.
"""

import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Verified Dedicated Spark Runtime Paths
SPARK_PYTHON_EXE = Path(r"C:\Spark\spark-env\Scripts\python.exe")
SPARK_INSTALL_DIR = Path(r"C:\Spark\spark-3.5.9-bin-hadoop3")
JAVA_HOME_DIR = Path(r"C:\Program Files\Microsoft\jdk-21.0.11.10-hotspot")

# Module-level singleton session reference
_ACTIVE_SPARK_SESSION = None


def validate_spark_environment() -> None:
    """
    Validate that Java, Apache Spark, and the dedicated Python runtime exist.
    Raises RuntimeError with clear resolution guidance if prerequisites are missing.
    """
    # 1. Validate dedicated Spark Python runtime
    if not SPARK_PYTHON_EXE.exists():
        raise RuntimeError(
            f"Dedicated Spark Python executable was not found at '{SPARK_PYTHON_EXE}'. "
            "Phase 7 requires the verified Python 3.13 runtime in C:\\Spark\\spark-env."
        )

    # 2. Validate current executing Python matches the dedicated runtime
    current_python = Path(sys.executable).resolve()
    target_python = SPARK_PYTHON_EXE.resolve()
    if current_python != target_python:
        logger.warning(
            "Spark session is being invoked from %s, but Spark workers must run under %s. "
            "Setting PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON to enforce the dedicated runtime.",
            current_python,
            target_python,
        )

    # 3. Validate Java Home
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        if JAVA_HOME_DIR.exists():
            os.environ["JAVA_HOME"] = str(JAVA_HOME_DIR)
            logger.info("Configured missing JAVA_HOME to verified path: %s", JAVA_HOME_DIR)
        else:
            raise RuntimeError(
                "JAVA_HOME environment variable is not set and Microsoft OpenJDK was not found at "
                f"'{JAVA_HOME_DIR}'."
            )

    # 4. Validate Spark Home
    spark_home = os.environ.get("SPARK_HOME")
    if not spark_home:
        if SPARK_INSTALL_DIR.exists():
            os.environ["SPARK_HOME"] = str(SPARK_INSTALL_DIR)
            logger.info("Configured missing SPARK_HOME to verified path: %s", SPARK_INSTALL_DIR)
        else:
            raise RuntimeError(
                "SPARK_HOME environment variable is not set and Apache Spark was not found at "
                f"'{SPARK_INSTALL_DIR}'."
            )

    # 5. Enforce dedicated Python for both driver and workers prior to PySpark import
    os.environ["PYSPARK_PYTHON"] = str(SPARK_PYTHON_EXE)
    os.environ["PYSPARK_DRIVER_PYTHON"] = str(SPARK_PYTHON_EXE)


def get_spark_session(
    app_name: str = "F1_Telemetry_Intelligence_Phase7",
    local_cores: int = 2,
    driver_memory: str = "1536m",
    executor_memory: str = "1536m",
    extra_configs: Optional[Dict[str, str]] = None,
    force_new: bool = False,
) -> Any:
    """
    Get or create a local Apache SparkSession configured for conservative memory use.

    Parameters:
    -----------
    app_name : str
        Human-readable application name for Spark monitoring.
    local_cores : int
        Number of local execution threads (defaults to 2 for local[2]).
    driver_memory : str
        Heap memory limit for the driver JVM (e.g. '1536m' or '2g').
        Note: Must be configured before Py4J launches the JVM.
    executor_memory : str
        Heap memory limit for local executors.
    extra_configs : Optional[Dict[str, str]]
        Additional Spark key-value configurations to append.
    force_new : bool
        If True, stops any existing active session and creates a fresh session.

    Returns:
    --------
    pyspark.sql.SparkSession
        Active, operational SparkSession.
    """
    global _ACTIVE_SPARK_SESSION

    # Reuse active session if valid and not stopped
    if not force_new and _ACTIVE_SPARK_SESSION is not None:
        try:
            if _ACTIVE_SPARK_SESSION._jsc is not None and not _ACTIVE_SPARK_SESSION._jsc.sc().isStopped():
                logger.debug("Reusing active SparkSession (app: %s)", _ACTIVE_SPARK_SESSION.sparkContext.appName)
                return _ACTIVE_SPARK_SESSION
        except Exception:
            _ACTIVE_SPARK_SESSION = None

    if force_new and _ACTIVE_SPARK_SESSION is not None:
        stop_spark_session()

    # Ensure environment variables are configured
    validate_spark_environment()

    # Import pyspark inside function to provide graceful error if running under incompatible venv
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError(
            f"PySpark is not installed in the active environment ({sys.executable}). "
            f"Phase 7 requires the dedicated environment at '{SPARK_PYTHON_EXE}'."
        ) from exc

    # Determine master string
    master_str = f"local[{local_cores}]" if local_cores > 0 else "local[*]"

    # Calculate conservative shuffle partition count
    # Defaulting 200 partitions for a local 2-core machine causes unnecessary task scheduling overhead.
    shuffle_partitions = max(2, local_cores * 2)

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_str)
        # Driver and executor memory limits (enforced at JVM startup)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        # Partition and parallelism defaults
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        # Disable web UI to eliminate unnecessary open ports in local sandbox
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        # Enable Arrow optimization for fast DataFrame serialization
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )

    if extra_configs:
        for k, v in extra_configs.items():
            builder = builder.config(k, v)

    logger.info("Initializing SparkSession: master=%s, memory=%s, shuffle_partitions=%s",
                master_str, driver_memory, shuffle_partitions)
    
    session = builder.getOrCreate()
    _ACTIVE_SPARK_SESSION = session
    return session


def stop_spark_session() -> None:
    """
    Gracefully terminate the active SparkSession and free JVM resources.
    """
    global _ACTIVE_SPARK_SESSION
    if _ACTIVE_SPARK_SESSION is not None:
        try:
            logger.info("Stopping active SparkSession (app: %s)", _ACTIVE_SPARK_SESSION.sparkContext.appName)
            _ACTIVE_SPARK_SESSION.stop()
        except Exception as e:
            logger.warning("Warning during SparkSession termination: %s", e)
        finally:
            _ACTIVE_SPARK_SESSION = None


def get_spark_diagnostics(spark: Optional[Any] = None) -> Dict[str, Any]:
    """
    Produce runtime diagnostic information distinguishing configured settings
    from verified live runtime properties.

    Parameters:
    -----------
    spark : Optional[SparkSession]
        Existing active session, or None to inspect active session / environment.

    Returns:
    --------
    dict: Structured diagnostics report.
    """
    active = spark or _ACTIVE_SPARK_SESSION
    diag: Dict[str, Any] = {
        "configured_environment": {
            "spark_python_exe": str(SPARK_PYTHON_EXE),
            "spark_install_dir": str(SPARK_INSTALL_DIR),
            "java_home_dir": str(JAVA_HOME_DIR),
            "pyspark_python_env": os.environ.get("PYSPARK_PYTHON"),
            "pyspark_driver_python_env": os.environ.get("PYSPARK_DRIVER_PYTHON"),
        },
        "python_runtime": {
            "current_executable": sys.executable,
            "version": sys.version,
            "is_dedicated_spark_python": Path(sys.executable).resolve() == SPARK_PYTHON_EXE.resolve(),
        },
        "spark_session_live": active is not None,
        "runtime_verified": {},
    }

    if active is not None:
        try:
            sc = active.sparkContext
            jvm = sc._jvm
            conf = active.conf
            
            java_ver = jvm.java.lang.System.getProperty("java.version") if jvm else "Unavailable"
            jvm_max_mem_mb = (jvm.java.lang.Runtime.getRuntime().maxMemory() / (1024 * 1024)) if jvm else 0.0
            jvm_avail_cores = jvm.java.lang.Runtime.getRuntime().availableProcessors() if jvm else 0

            diag["runtime_verified"] = {
                "spark_version": active.version,
                "app_name": sc.appName,
                "master": sc.master,
                "default_parallelism": sc.defaultParallelism,
                "shuffle_partitions": conf.get("spark.sql.shuffle.partitions", "not_set"),
                "driver_memory_conf": conf.get("spark.driver.memory", "not_set"),
                "executor_memory_conf": conf.get("spark.executor.memory", "not_set"),
                "ui_enabled_conf": conf.get("spark.ui.enabled", "not_set"),
                "arrow_enabled_conf": conf.get("spark.sql.execution.arrow.pyspark.enabled", "not_set"),
                "java_version_jvm": java_ver,
                "jvm_max_memory_mb": round(jvm_max_mem_mb, 2),
                "jvm_available_cores": jvm_avail_cores,
            }
        except Exception as err:
            diag["runtime_verified"]["diagnostic_error"] = str(err)

    return diag
