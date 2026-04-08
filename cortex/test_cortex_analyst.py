import json
import os

import requests
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_private_key,
)
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

# Load private key
with open("snowflake_rsa_key.p8", "rb") as f:
    private_key = load_pem_private_key(f.read(), password=None)

private_key_bytes = private_key.private_bytes(
    Encoding.DER, PrivateFormat.PKCS8, NoEncryption()
)

conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    private_key=private_key_bytes,
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    role=os.getenv("SNOWFLAKE_ROLE"),
)

# Inspect connection internals to find the token
rest = conn._rest
print("=== Connection internals ===")
print(f"type(rest): {type(rest)}")
print(f"dir(rest) token-related: {[a for a in dir(rest) if 'token' in a.lower()]}")

token = None
# Try common token locations
for attr in ("token", "_token", "master_token", "_master_token"):
    val = getattr(rest, attr, None)
    if val:
        print(f"Found token at rest.{attr}: {str(val)[:50]}...")
        if token is None:
            token = val

# Try _token_provider
tp = getattr(rest, "_token_provider", None)
if tp:
    print(f"type(_token_provider): {type(tp)}")
    print(f"dir(_token_provider): {[a for a in dir(tp) if not a.startswith('__')]}")
    try:
        tp_token = tp.get_token()
        print(f"_token_provider.get_token(): {str(tp_token)[:50]}...")
        if token is None:
            token = tp_token
    except Exception as e:
        print(f"_token_provider.get_token() failed: {e}")

if not token:
    print("ERROR: Could not find session token")
    conn.close()
    exit(1)

account = os.getenv("SNOWFLAKE_ACCOUNT")
host = f"{account}.snowflakecomputing.com"

with open("cortex/semantic_model.yaml", "r") as f:
    semantic_model = f.read()

url = f"https://{host}/api/v2/cortex/analyst/message"

headers = {
    "Authorization": f'Snowflake Token="{token}"',
    "Content-Type": "application/json",
    "Accept": "application/json",
}

body = {
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who are the top 10 pilots at risk?"}],
        }
    ],
    "semantic_model": semantic_model,
}

print(f"\n=== API Call ===")
print(f"URL: {url}")
print(f"Auth header: {headers['Authorization'][:60]}...")

response = requests.post(url, headers=headers, json=body, timeout=30)
print(f"\nStatus: {response.status_code}")
print(f"Response: {response.text[:2000]}")

conn.close()
