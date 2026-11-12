"""
F1 Telemetry Intelligence: SQL Query Runner
Executes analytical SQL queries and scripts against PostgreSQL or SQLite,
benchmarks execution times, returns Pandas DataFrames, and exports results.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sqlalchemy import text
from src.database.connection import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_engineering" / "sql_execution"


def run_query(
    query_str: str,
    params: Optional[Dict[str, Any]] = None,
    engine=None
) -> Tuple[pd.DataFrame, float]:
    """
    Execute a single SQL query string and return the result as a DataFrame
    along with execution time in seconds.
    
    Args:
        query_str (str): The SQL statement.
        params (dict, optional): Parameter dictionary.
        engine (Engine, optional): SQLAlchemy engine.
        
    Returns:
        Tuple[pd.DataFrame, float]: (Results DataFrame, execution time in seconds)
    """
    if engine is None:
        engine = get_engine()
        
    start_time = time.perf_counter()
    with engine.connect() as conn:
        result_df = pd.read_sql_query(text(query_str), conn, params=params)
    elapsed_sec = time.perf_counter() - start_time
    
    return result_df, elapsed_sec


def execute_sql_file(
    file_path: Path,
    params: Optional[Dict[str, Any]] = None,
    engine=None
) -> List[Tuple[str, pd.DataFrame, float]]:
    """
    Parse and execute SQL statements inside a .sql file.
    Statements separated by semicolons are executed sequentially.
    
    Args:
        file_path (Path): Path to .sql file.
        params (dict, optional): Parameter mapping.
        engine (Engine, optional): SQLAlchemy engine.
        
    Returns:
        List[Tuple[str, pd.DataFrame, float]]: List of (statement_preview, df, elapsed_sec)
    """
    if engine is None:
        engine = get_engine()
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # If running on SQLite, replace Postgres-specific syntax like ::numeric
    if engine.dialect.name == "sqlite":
        # Replace ::numeric and ::numeric(p,s) casts
        import re
        content = re.sub(r'::numeric(\(\d+,\s*\d+\))?', '', content)
        
    raw_statements = [stmt.strip() for stmt in content.split(";") if stmt.strip()]
    results = []
    
    for stmt in raw_statements:
        # Skip pure comments or empty blocks
        lines = [line for line in stmt.splitlines() if not line.strip().startswith("--")]
        clean_stmt = "\n".join(lines).strip()
        if not clean_stmt:
            continue
            
        header = stmt.splitlines()[0][:60]
        try:
            df, elapsed = run_query(clean_stmt, params=params, engine=engine)
            results.append((header, df, elapsed))
            logger.info(f"Executed: [{header}] -> {len(df)} rows ({elapsed:.4f}s)")
        except Exception as e:
            logger.error(f"Error executing statement [{header}]: {e}")
            raise e
            
    return results


def export_query_results(df: pd.DataFrame, output_name: str, export_format: str = "csv") -> Path:
    """Export query results to CSV or JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if export_format == "csv":
        out_path = OUTPUT_DIR / f"{output_name}.csv"
        df.to_csv(out_path, index=False)
    elif export_format == "json":
        out_path = OUTPUT_DIR / f"{output_name}.json"
        df.to_json(out_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {export_format}")
    logger.info(f"Exported query results to {out_path}")
    return out_path


def run_all_analysis_scripts(engine=None) -> Dict[str, List[Tuple[str, pd.DataFrame, float]]]:
    """Execute all analysis scripts (05 to 10), export results, and benchmark them."""
    if engine is None:
        engine = get_engine()
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "sql_execution_log.txt"

    script_files = [
        "05_basic_analysis.sql",
        "06_joins_and_aggregations.sql",
        "07_cte_analysis.sql",
        "08_window_functions.sql",
        "09_advanced_telemetry_analysis.sql",
        "10_data_quality_checks.sql",
    ]
    
    benchmark_report = {}
    benchmark_rows = []
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"F1 Telemetry Intelligence - PostgreSQL Query Execution Log\n")
        lf.write(f"Dialect: {engine.dialect.name}\n")
        lf.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write("=" * 80 + "\n\n")

        for filename in script_files:
            path = SQL_DIR / filename
            if path.exists():
                logger.info(f"\n--- Running {filename} ---")
                lf.write(f"\n{'='*80}\nSCRIPT: {filename}\n{'='*80}\n")
                results = execute_sql_file(path, engine=engine)
                benchmark_report[filename] = results
                
                script_stem = Path(filename).stem
                for idx, (header, df, elapsed) in enumerate(results, start=1):
                    # Clean filename slug
                    csv_name = f"{script_stem}_q{idx:02d}"
                    out_csv = export_query_results(df, csv_name, export_format="csv")
                    benchmark_rows.append({
                        "script": filename,
                        "query_index": idx,
                        "query_header": header[:60],
                        "row_count": len(df),
                        "execution_time_s": round(elapsed, 4),
                        "output_file": out_csv.name,
                    })
                    
                    lf.write(f"\n--- Query {idx}: {header} ---\n")
                    lf.write(f"Rows: {len(df)} | Execution Time: {elapsed:.4f}s\n")
                    if len(df) > 0:
                        lf.write(df.head(10).to_string() + "\n")
                        if len(df) > 10:
                            lf.write(f"... ({len(df) - 10} more rows)\n")
                    lf.write("\n")

    # Export benchmark summary
    benchmarks_df = pd.DataFrame(benchmark_rows)
    bench_csv = OUTPUT_DIR / "sql_benchmarks.csv"
    benchmarks_df.to_csv(bench_csv, index=False)
    logger.info(f"Saved execution benchmarks to {bench_csv}")
    logger.info(f"Saved full execution log to {log_file}")
            
    return benchmark_report


if __name__ == "__main__":
    report = run_all_analysis_scripts()
