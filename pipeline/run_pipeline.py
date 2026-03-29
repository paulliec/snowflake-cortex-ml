"""Orchestrate the full attrition ML pipeline."""

import time

from pipeline.connection import get_connection
from pipeline.ingest import setup_schema, load_bronze
from pipeline.transform import bronze_to_silver, silver_to_gold
from pipeline.train_model import train
from pipeline.predict import predict


STEPS = [
    ("Schema Setup", setup_schema),
    ("Bronze Ingestion", load_bronze),
    ("Bronze → Silver", bronze_to_silver),
    ("Silver → Gold", silver_to_gold),
    ("Model Training", train),
    ("Prediction", predict),
]


def run():
    conn = get_connection()
    print("=" * 60)
    print("Attrition ML Pipeline")
    print("=" * 60)

    try:
        for i, (name, step_fn) in enumerate(STEPS, 1):
            print(f"\n[{i}/{len(STEPS)}] {name}")
            start = time.time()
            step_fn(conn)
            elapsed = time.time() - start
            print(f"  Done ({elapsed:.1f}s)")
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    run()
