"""Shared Snowflake connection factory. Reads from st.secrets (Streamlit Cloud) or .env (local)."""

import os
from pathlib import Path

import snowflake.connector
import streamlit as st
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load .env for local dev (no-op if file doesn't exist)
load_dotenv(_PROJECT_ROOT / ".env")


def _resolve_key_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    candidate = (_PROJECT_ROOT / raw).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Private key not found: {raw} (tried cwd and {_PROJECT_ROOT})")


def _pem_to_der(pem_bytes: bytes, passphrase: str = None) -> bytes:
    pw = passphrase.encode() if passphrase else None
    p_key = serialization.load_pem_private_key(pem_bytes, password=pw, backend=default_backend())
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_connection(**overrides):
    try:
        # Streamlit Cloud — read from st.secrets
        sf = st.secrets["snowflake"]
        account = sf["account"]
        user = sf["user"]
        warehouse = sf["warehouse"]
        database = sf.get("database", "ATTRITION_ML")
        role = sf.get("role", "ATTRITION_ROLE")
        private_key_pem = sf.get("private_key")
        password = sf.get("password")
        passphrase = sf.get("private_key_passphrase")
    except (KeyError, FileNotFoundError):
        # Local dev — read from os.environ / .env
        account = os.environ["SNOWFLAKE_ACCOUNT"]
        user = os.environ["SNOWFLAKE_USER"]
        warehouse = os.environ["SNOWFLAKE_WAREHOUSE"]
        database = os.environ.get("SNOWFLAKE_DATABASE", "ATTRITION_ML")
        role = os.environ.get("SNOWFLAKE_ROLE", "ATTRITION_ROLE")
        private_key_pem = os.environ.get("SNOWFLAKE_PRIVATE_KEY")
        password = os.environ.get("SNOWFLAKE_PASSWORD")
        passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    params = {
        "account": account,
        "user": user,
        "warehouse": warehouse,
        "database": database,
        "role": role,
    }

    key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    if key_path:
        params["private_key"] = _pem_to_der(
            _resolve_key_path(key_path).read_bytes(), passphrase
        )
    elif private_key_pem:
        params["private_key"] = _pem_to_der(private_key_pem.encode(), passphrase)
    elif password:
        params["password"] = password
    else:
        raise ValueError(
            "Set password or private_key in st.secrets[snowflake], "
            "or SNOWFLAKE_PASSWORD / SNOWFLAKE_PRIVATE_KEY_PATH in .env"
        )

    params.update(overrides)
    return snowflake.connector.connect(**params)


# Cortex ML classification models live in a schema; CREATE / !PREDICT need current schema set.
ML_MODEL_SCHEMA = "GOLD"


def use_ml_schema(cursor):
    try:
        db = st.secrets["snowflake"].get("database", "ATTRITION_ML")
    except (KeyError, FileNotFoundError):
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
