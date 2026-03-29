"""Shared Snowflake connection factory. Reads creds from .env."""

import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

# load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_connection(**overrides):
    params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "database": os.environ.get("SNOWFLAKE_DATABASE", "ATTRITION_ML"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
    }
    params.update(overrides)
    return snowflake.connector.connect(**params)


def run_sql_file(conn, filepath):
    """Execute a SQL file, splitting on semicolons."""
    sql = Path(filepath).read_text()
    cur = conn.cursor()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)
    cur.close()
