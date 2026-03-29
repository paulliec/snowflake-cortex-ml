# Snowflake Cortex ML — Employee Attrition Prediction

End-to-end ML pipeline demonstrating how to activate structured HR data and unstructured exit survey text through Snowflake Cortex to predict and understand employee attrition in hard-to-staff roles.

## The Business Problem

Replacing specialized employees is expensive — a departing pilot costs 2-3x their annual salary in recruiting, training, and lost productivity. Specialty physicians, critical care nurses, and senior engineers carry similar replacement costs and long ramp-up times.

Most attrition models stop at structured data: tenure, salary, performance scores. This pipeline goes further by incorporating free-text exit survey responses through Cortex LLM functions, giving leadership both a *prediction* of who might leave and an *understanding* of why.

**Target roles:** Pilots, Cardiologists, ICU Nurses, Anesthesiologists, Flight Paramedics, Data Engineers, Security Engineers.

## Architecture

```
  CSV Generator           Snowflake (Medallion Architecture)
 ┌──────────────┐    ┌─────────────────────────────────────────────────┐
 │  Faker +     │    │                                                 │
 │  NumPy       │───▶│  BRONZE          SILVER           GOLD          │
 │  2000 rows   │    │  ┌──────────┐   ┌──────────┐   ┌──────────┐   │
 └──────────────┘    │  │EMPLOYEE_ │──▶│EMPLOYEE_ │──▶│EMPLOYEE_ │   │
                     │  │RAW       │   │CLEANSED  │   │ML_READY  │   │
                     │  │          │   │          │   │          │   │
                     │  │all VARCHAR│   │typed cols │   │features  │   │
                     │  │+LOADED_AT│   │+SENTIMENT│   │+SPLIT    │   │
                     │  └──────────┘   │+RISK_SCORE   │80/20     │   │
                     │                 └──────────┘   └──────────┘   │
                     │                      │               │         │
                     │          Cortex AI_SENTIMENT    Cortex ML      │
                     │          (exit survey text)    Classification  │
                     │                      │               │         │
                     └──────────────────────┼───────────────┼─────────┘
                                            │               │
                                            ▼               ▼
                     ┌─────────────────────────────────────────────────┐
                     │              Streamlit Dashboard                 │
                     │  • Churn rate by role / region                   │
                     │  • Top 25 at-risk employees (filterable)        │
                     │  • Feature importance from Cortex ML            │
                     │  • Exit survey sentiment distribution           │
                     └─────────────────────────────────────────────────┘
```

### Pipeline Flow

```
python pipeline/run_pipeline.py

  [1/6] Schema Setup       — creates ATTRITION_ML db, BRONZE/SILVER/GOLD schemas
  [2/6] Bronze Ingestion   — loads CSV as all-VARCHAR into BRONZE.EMPLOYEE_RAW
  [3/6] Bronze → Silver    — type casts, cleans, runs Cortex AI_SENTIMENT()
  [4/6] Silver → Gold      — builds feature table, 80/20 train/test split
  [5/6] Model Training     — Cortex ML Classification on GOLD.EMPLOYEE_ML_READY
  [6/6] Prediction         — scores test set, writes risk back to Silver
```

## Prerequisites

- Python 3.9+
- Snowflake account with Cortex ML and Cortex LLM functions enabled
- Snowflake role with CREATE DATABASE, CREATE SCHEMA privileges

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd snowflake-cortex-ml
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Configure Snowflake credentials
cp .env.example .env
# edit .env with your Snowflake account details

# 3. Generate synthetic data (2000 employees, ~19% churn)
python generators/generate_attrition_data.py --records 2000

# 4. Run the full pipeline
python -m pipeline.run_pipeline

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
snowflake-cortex-ml/
├── data/                          # generated CSVs (gitignored)
├── generators/
│   └── generate_attrition_data.py # synthetic data with realistic churn patterns
├── pipeline/
│   ├── sql/
│   │   ├── 01_setup_schema.sql    # database + table DDL
│   │   ├── 02_bronze_to_silver.sql # type casting + Cortex sentiment
│   │   └── 03_silver_to_gold.sql  # feature engineering + train/test split
│   ├── connection.py              # shared Snowflake connection factory
│   ├── ingest.py                  # CSV → Bronze loader
│   ├── transform.py               # Silver + Gold transformations
│   ├── train_model.py            # Cortex ML Classification training
│   ├── predict.py                 # score employees, write risk to Silver
│   └── run_pipeline.py           # full pipeline orchestrator
├── dashboard/
│   └── app.py                     # Streamlit dashboard (4 pages)
├── docs/
├── requirements.txt
├── .env.example
└── .gitignore
```

## Snowflake Cortex Functions Used

| Function | Purpose |
|----------|---------|
| `SNOWFLAKE.CORTEX.SENTIMENT()` | Score exit survey text (-1 to 1) |
| `SNOWFLAKE.ML.CLASSIFICATION` | Train binary classifier on structured + sentiment features |
| `model!PREDICT()` | Score active employees for churn risk |
| `model!SHOW_EVALUATION_METRICS()` | Model accuracy, precision, recall |
| `model!SHOW_FEATURE_IMPORTANCE()` | Which features drive attrition |

## Self-Learning Loop (Future State)

The end goal is a pipeline that improves itself:

1. **Predict** — Model scores active employees for churn risk
2. **Intervene** — Dashboard flags high-risk employees for manager action
3. **Observe** — Track whether interventions changed outcomes
4. **Retrain** — New outcomes feed back into the model as training data
5. **Refine** — Cortex LLM summarizes exit survey themes quarterly, surfacing emerging reasons that the structured model might miss

This can be automated with Snowflake Tasks on a schedule:

```sql
-- example: retrain monthly
CREATE TASK ATTRITION_ML.PUBLIC.RETRAIN_MONTHLY
  WAREHOUSE = 'COMPUTE_WH'
  SCHEDULE = 'USING CRON 0 2 1 * * America/Chicago'
AS
  -- trigger pipeline via Snowflake Python UDF or external function
```

## Tech Stack

- **Snowflake** — Warehouse, Cortex ML, Cortex LLM functions
- **Python** — Data generation, pipeline orchestration
- **Streamlit** — Interactive dashboard
- **Faker** — Realistic synthetic data
- **Plotly** — Dashboard visualizations
