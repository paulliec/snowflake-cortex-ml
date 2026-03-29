"""Train a Cortex ML Classification model on the Gold feature set."""

from snowflake.ml.modeling.classification import ClassificationExperiment

from pipeline.connection import get_connection

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

    # build model on TRAIN split
    feature_list = ", ".join(FEATURE_COLS)
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

    # evaluate on test split
    cur.execute(f"""
        SELECT {MODEL_NAME}!SHOW_EVALUATION_METRICS()
    """)
    metrics = cur.fetchone()[0]
    print(f"  Evaluation metrics:\n{metrics}")

    # feature importance
    cur.execute(f"""
        SELECT {MODEL_NAME}!SHOW_FEATURE_IMPORTANCE()
    """)
    importance = cur.fetchone()[0]
    print(f"\n  Feature importance:\n{importance}")

    cur.close()
    return metrics


def main():
    conn = get_connection()
    try:
        train(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
