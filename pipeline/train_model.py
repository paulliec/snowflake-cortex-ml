"""Train a Cortex ML Classification model on the Gold feature set."""

from pipeline.connection import get_connection, use_ml_schema

MODEL_NAME = "ATTRITION_CLASSIFIER"
TARGET = "CHURNED"
# categorical features get one-hot encoded by Cortex ML internally
FEATURE_COLS = [
    "ROLE", "REGION", "TENURE_YEARS", "SALARY",
    "PERFORMANCE_SCORE", "MANAGER_RATING", "PROMOTIONS_LAST_3_YEARS",
    "OVERTIME_HOURS_MONTHLY", "DAYS_SINCE_LAST_RAISE", "TEAM_SIZE",
    "REMOTE_ELIGIBLE", "EXIT_SURVEY_SENTIMENT",
]


def train(conn):
    """Train classification model using Snowflake Cortex ML."""
    print("  Training Cortex ML Classification model...")
    cur = conn.cursor()
    use_ml_schema(cur)

    # build model on TRAIN split
    cur.execute(f"""
        CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION {MODEL_NAME}(
            INPUT_DATA => SYSTEM$REFERENCE('TABLE', 'GOLD.EMPLOYEE_ML_READY'),
            TARGET_COLNAME => '{TARGET}',
            CONFIG_OBJECT => {{
                'ON_ERROR': 'SKIP'
            }}
        )
    """)
    print(f"  Model '{MODEL_NAME}' created.")

    # evaluation APIs are CALL ... (), not SELECT (see Snowflake ML classification docs)
    cur.execute(f"CALL {MODEL_NAME}!SHOW_EVALUATION_METRICS()")
    eval_rows = cur.fetchall()
    eval_cols = [c[0] for c in cur.description]
    print("  Evaluation metrics:")
    for row in eval_rows:
        print(f"    {dict(zip(eval_cols, row))}")

    cur.execute(f"CALL {MODEL_NAME}!SHOW_FEATURE_IMPORTANCE()")
    fi_rows = cur.fetchall()
    fi_cols = [c[0] for c in cur.description]
    print("\n  Feature importance:")
    for row in fi_rows:
        print(f"    {dict(zip(fi_cols, row))}")

    cur.close()
    return eval_rows


def main():
    conn = get_connection()
    try:
        train(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
