"""Run Bronze → Silver → Gold SQL transformations."""

from pathlib import Path

from pipeline.connection import get_connection, run_sql_file

SQL_DIR = Path(__file__).resolve().parent / "sql"


def bronze_to_silver(conn):
    """Type-cast, clean, and enrich with Cortex sentiment."""
    print("  Running Bronze → Silver...")
    run_sql_file(conn, SQL_DIR / "02_bronze_to_silver.sql")

    row_count = conn.cursor().execute(
        "SELECT COUNT(*) FROM SILVER.EMPLOYEE_CLEANSED"
    ).fetchone()[0]

    sentiment_count = conn.cursor().execute(
        "SELECT COUNT(*) FROM SILVER.EMPLOYEE_CLEANSED WHERE EXIT_SURVEY_SENTIMENT IS NOT NULL"
    ).fetchone()[0]

    print(f"  Silver loaded: {row_count} rows, {sentiment_count} with sentiment scores")
    return row_count


def silver_to_gold(conn):
    """Build ML-ready feature set with train/test split."""
    print("  Running Silver → Gold...")
    run_sql_file(conn, SQL_DIR / "03_silver_to_gold.sql")

    cur = conn.cursor()
    cur.execute("""
        SELECT SPLIT, COUNT(*), SUM(CHURNED), ROUND(AVG(CHURNED), 3)
        FROM GOLD.EMPLOYEE_ML_READY
        GROUP BY SPLIT
        ORDER BY SPLIT
    """)
    for split, total, churned, rate in cur.fetchall():
        print(f"  Gold [{split}]: {total} rows, {churned} churned ({rate:.1%})")
    cur.close()


def main():
    conn = get_connection()
    try:
        bronze_to_silver(conn)
        silver_to_gold(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
