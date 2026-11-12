"""
F1 Telemetry Intelligence: Local Database Connection Management
Supports PostgreSQL via psycopg2 / SQLAlchemy with seamless fallback to SQLite
for continuous testing and environments without an active PostgreSQL daemon.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import psycopg2

logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "processed" / "f1_telemetry.db"


def get_postgres_config() -> dict:
    """Read PostgreSQL connection parameters from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "f1_telemetry"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }


def is_postgres_available(timeout_sec: int = 2) -> bool:
    """
    Check if the local PostgreSQL server is running and accepting connections.
    
    Returns:
        bool: True if connection succeeded, False otherwise.
    """
    cfg = get_postgres_config()
    try:
        conn = psycopg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            dbname=cfg["database"],
            user=cfg["user"],
            password=cfg["password"],
            connect_timeout=timeout_sec,
        )
        conn.close()
        return True
    except Exception as e:
        logger.debug(f"PostgreSQL connection check failed: {e}")
        return False


def get_engine(force_sqlite: bool = False, echo: bool = False) -> Engine:
    """
    Create and return an SQLAlchemy Engine.
    Attempts PostgreSQL connection first; if unavailable or force_sqlite=True,
    falls back to a local SQLite database for test execution and offline exploration.
    
    Args:
        force_sqlite (bool): Force using SQLite regardless of PostgreSQL status.
        echo (bool): Whether to log SQL queries.
        
    Returns:
        Engine: SQLAlchemy Engine connected to PostgreSQL or SQLite.
    """
    use_sqlite_env = os.getenv("USE_SQLITE_FALLBACK", "0").lower() in ("1", "true", "yes")
    
    if not force_sqlite and not use_sqlite_env and is_postgres_available():
        cfg = get_postgres_config()
        pg_url = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        logger.info(f"Connected to local PostgreSQL database: {cfg['database']}@{cfg['host']}:{cfg['port']}")
        return create_engine(pg_url, echo=echo, pool_pre_ping=True, pool_size=5, max_overflow=10)
    
    # Fallback to local SQLite
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sqlite_url = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
    logger.info(f"Using SQLite database engine at {DEFAULT_SQLITE_PATH}")
    return create_engine(sqlite_url, echo=echo)


def get_db_connection():
    """
    Get a raw DBAPI connection (psycopg2 or sqlite3 depending on availability).
    
    Returns:
        Connection: Active database connection.
    """
    engine = get_engine()
    return engine.raw_connection()
