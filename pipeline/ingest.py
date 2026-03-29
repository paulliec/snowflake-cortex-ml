"""Load synthetic CSV into Bronze.EMPLOYEE_RAW."""

from pathlib import Path

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from pipeline.connection import get_connection, run_sql_file

SQL_DIR = Path(__file__).resolve().parent / "sql"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def setup_schema(conn):
    """Create database, schemas, tables if they don't exist."""
    print("  Setting up schema...")
    run_sql_file(conn, SQL_DIR / "01_setup_schema.sql")
    print("  Schema ready.")


def load_bronze(conn, csv_path=None):
    """Read CSV and write to Bronze staging table."""
    csv_path = csv_path or DATA_DIR / "synthetic_attrition_data.csv"
    print(f"  Loading {csv_path}...")

    df = pd.read_csv(csv_path, dtype=str)  # all VARCHAR in bronze
    df.columns = [c.upper() for c in df.columns]

    # truncate before reload
    conn.cursor().execute("TRUNCATE TABLE IF EXISTS BRONZE.EMPLOYEE_RAW")

    write_pandas(
        conn, df,
        table_name="EMPLOYEE_RAW",
        schema="BRONZE",
        database="ATTRITION_ML",
        auto_create_table=False,
    )

    row_count = conn.cursor().execute(
        "SELECT COUNT(*) FROM BRONZE.EMPLOYEE_RAW"
    ).fetchone()[0]
    print(f"  Bronze loaded: {row_count} rows")
    return row_count


def main():
    conn = get_connection()
    try:
        setup_schema(conn)
        load_bronze(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
