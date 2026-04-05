"""Shared Snowflake connection factory. Reads creds from .env."""

import os
from pathlib import Path

import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load .env from project root
load_dotenv(_PROJECT_ROOT / ".env")


def _resolve_key_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    # relative paths are from project root (same folder as .env)
    candidate = (_PROJECT_ROOT / raw).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Private key not found: {raw} (tried cwd and {_PROJECT_ROOT})")


def _private_key_der(path: Path) -> bytes:
    raw = path.read_bytes()
    passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    pw = passphrase.encode() if passphrase else None
    p_key = serialization.load_pem_private_key(raw, password=pw, backend=default_backend())
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_connection(**overrides):
    params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "database": os.environ.get("SNOWFLAKE_DATABASE", "ATTRITION_ML"),
    }
    role = os.environ.get("SNOWFLAKE_ROLE")
    if role:
        params["role"] = role

    key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    if key_path:
        params["private_key"] = _private_key_der(_resolve_key_path(key_path))
    elif os.environ.get("SNOWFLAKE_PASSWORD"):
        params["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    else:
        raise ValueError(
            "Set SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY_PATH (and optional "
            "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE) in .env"
        )

    params.update(overrides)
    return snowflake.connector.connect(**params)


# Cortex ML classification models live in a schema; CREATE / !PREDICT need current schema set.
ML_MODEL_SCHEMA = "GOLD"


def use_ml_schema(cursor):
    db = os.environ.get("SNOWFLAKE_DATABASE", "ATTRITION_ML")
    cursor.execute(f"USE DATABASE {db}")
    cursor.execute(f"USE SCHEMA {ML_MODEL_SCHEMA}")


def _sql_for_semicolon_split(sql: str) -> str:
    # naive split on ";" breaks if a semicolon appears inside a -- comment line
    lines = []
    for line in sql.splitlines():
        if line.lstrip().startswith("--"):
            line = line.replace(";", " ")
        lines.append(line)
    return "\n".join(lines)


def run_sql_file(conn, filepath):
    """Execute a SQL file, splitting on semicolons."""
    sql = _sql_for_semicolon_split(Path(filepath).read_text())
    cur = conn.cursor()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)
    cur.close()
