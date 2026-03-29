"""Run predictions and write churn risk scores back to Silver."""

import pandas as pd

from pipeline.connection import get_connection

MODEL_NAME = "ATTRITION_CLASSIFIER"


def predict(conn):
    """Score test split and write risk scores back to Silver."""
    print("  Running predictions on test split...")
    cur = conn.cursor()

    # predict on test set, get probability of churn (class 1)
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
        FROM GOLD.EMPLOYEE_ML_READY
        WHERE SPLIT = 'TEST'
    """)

    # write risk scores back to silver
    cur.execute("""
        UPDATE SILVER.EMPLOYEE_CLEANSED s
        SET s.CHURN_RISK_SCORE = p.PREDICTION:"probability"::FLOAT
        FROM _PREDICTIONS p
        WHERE s.EMPLOYEE_ID = p.EMPLOYEE_ID
    """)

    updated = cur.rowcount
    print(f"  Updated {updated} risk scores in Silver.")

    # show top 10 highest risk by role
    cur.execute("""
        SELECT EMPLOYEE_ID, EMPLOYEE_NAME, ROLE, REGION,
               ROUND(CHURN_RISK_SCORE, 3) AS RISK_SCORE
        FROM SILVER.EMPLOYEE_CLEANSED
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
