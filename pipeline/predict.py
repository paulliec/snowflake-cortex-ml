"""Run predictions and write churn risk scores back to Silver."""

import os

import pandas as pd

from pipeline.connection import get_connection, use_ml_schema

MODEL_NAME = "ATTRITION_CLASSIFIER"


def predict(conn):
    """Score Gold rows and write risk scores back to Silver."""
    print("  Running predictions (all Gold rows with features)...")
    cur = conn.cursor()
    use_ml_schema(cur)

    db = os.environ.get("SNOWFLAKE_DATABASE", "ATTRITION_ML").strip().strip('"')

    # score every row in Gold (train + test) so the dashboard has broad coverage
    cur.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _PREDICTIONS AS
        SELECT
            EMPLOYEE_ID,
            {MODEL_NAME}!PREDICT(
                INPUT_DATA => OBJECT_CONSTRUCT(
                    'ROLE', ROLE,
                    'REGION', REGION,
                    'TENURE_YEARS', TENURE_YEARS,
                    'SALARY', SALARY,
                    'PERFORMANCE_SCORE', PERFORMANCE_SCORE,
                    'MANAGER_RATING', MANAGER_RATING,
                    'PROMOTIONS_LAST_3_YEARS', PROMOTIONS_LAST_3_YEARS,
                    'OVERTIME_HOURS_MONTHLY', OVERTIME_HOURS_MONTHLY,
                    'DAYS_SINCE_LAST_RAISE', DAYS_SINCE_LAST_RAISE,
                    'TEAM_SIZE', TEAM_SIZE,
                    'REMOTE_ELIGIBLE', REMOTE_ELIGIBLE,
                    'EXIT_SURVEY_SENTIMENT', EXIT_SURVEY_SENTIMENT
                )
            ) AS PREDICTION
        FROM {db}.GOLD.EMPLOYEE_ML_READY
    """)

    cur.execute("SELECT COUNT(*) FROM _PREDICTIONS")
    n_pred = cur.fetchone()[0]
    print(f"  Rows in _PREDICTIONS: {n_pred}")

    # inspect raw PREDICTION variant to get the correct extraction path
    cur.execute("SELECT PREDICTION FROM _PREDICTIONS LIMIT 1")
    sample = cur.fetchone()[0]
    print(f"  Sample PREDICTION variant: {sample}")

    # Cortex ML Classification returns {"class": ..., "probability": {"0": p0, "1": p1}}
    # but key names match the target column's distinct values — for NUMBER(1,0)
    # targets, keys may be integers (0/1) not strings ("0"/"1").
    # Try both paths and pick whichever isn't null.
    cur.execute("""
        SELECT
            PREDICTION:"probability":"1"::FLOAT   AS str_key,
            PREDICTION['probability'][1]::FLOAT    AS int_key
        FROM _PREDICTIONS LIMIT 1
    """)
    str_val, int_val = cur.fetchone()
    print(f"  Probability via string key '\"1\"': {str_val}")
    print(f"  Probability via integer key [1]: {int_val}")

    if str_val is not None and str_val != 0.0 and str_val != 1.0:
        prob_expr = """p.PREDICTION:"probability":"1"::FLOAT"""
        print("  Using string key path")
    elif int_val is not None and int_val != 0.0 and int_val != 1.0:
        prob_expr = """p.PREDICTION['probability'][1]::FLOAT"""
        print("  Using integer key path")
    else:
        # fallback — try the class probabilities object directly
        prob_expr = """p.PREDICTION:"probability":"1"::FLOAT"""
        print("  WARNING: neither path returned a probability — defaulting to string key")

    # write risk scores back to Silver
    cur.execute(f"""
        MERGE INTO {db}.SILVER.EMPLOYEE_CLEANSED s
        USING _PREDICTIONS p
        ON s.EMPLOYEE_ID = p.EMPLOYEE_ID
        WHEN MATCHED THEN UPDATE SET
            s.CHURN_RISK_SCORE = {prob_expr}
    """)

    updated = cur.rowcount
    print(f"  MERGE rows (Snowflake rowcount): {updated}")

    cur.execute(f"""
        SELECT COUNT(*) FROM {db}.SILVER.EMPLOYEE_CLEANSED WHERE CHURN_RISK_SCORE IS NOT NULL
    """)
    n_scored = cur.fetchone()[0]
    print(f"  Silver rows with CHURN_RISK_SCORE set: {n_scored}")

    # verify scores are real probabilities, not binary 0/1
    cur.execute(f"""
        SELECT MIN(CHURN_RISK_SCORE), MAX(CHURN_RISK_SCORE),
               ROUND(AVG(CHURN_RISK_SCORE), 4),
               COUNT(DISTINCT ROUND(CHURN_RISK_SCORE, 2))
        FROM {db}.SILVER.EMPLOYEE_CLEANSED
        WHERE CHURN_RISK_SCORE IS NOT NULL
    """)
    mn, mx, avg, n_distinct = cur.fetchone()
    print(f"\n  Score stats — min: {mn}  max: {mx}  avg: {avg}  distinct values: {n_distinct}")
    if n_distinct <= 2:
        print("  WARNING: only 2 distinct values — scores may still be binary class labels")

    # show top 10 highest risk by role
    cur.execute(f"""
        SELECT EMPLOYEE_ID, EMPLOYEE_NAME, ROLE, REGION,
               ROUND(CHURN_RISK_SCORE, 3) AS RISK_SCORE
        FROM {db}.SILVER.EMPLOYEE_CLEANSED
        WHERE CHURN_RISK_SCORE IS NOT NULL
        ORDER BY CHURN_RISK_SCORE DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    print("\n  Top 10 highest churn risk:")
    print(f"  {'ID':<12} {'Name':<25} {'Role':<22} {'Region':<12} {'Risk':>6}")
    print(f"  {'-'*80}")
    for emp_id, name, role, region, score in rows:
        print(f"  {emp_id:<12} {name:<25} {role:<22} {region:<12} {score:>6}")

    cur.close()


def main():
    conn = get_connection()
    try:
        predict(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
