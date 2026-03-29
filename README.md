# Snowflake Cortex ML — Employee Attrition Prediction

End-to-end ML pipeline demonstrating how to activate structured HR data and unstructured exit survey text through Snowflake Cortex to predict and understand employee attrition in hard-to-staff roles.

## The Business Problem

Replacing specialized employees is expensive — a departing pilot costs 2-3x their annual salary in recruiting, training, and lost productivity. Specialty physicians, critical care nurses, and senior engineers carry similar replacement costs and long ramp-up times.

Most attrition models stop at structured data: tenure, salary, performance scores. This pipeline goes further by incorporating free-text exit survey responses through Cortex LLM functions, giving leadership both a *prediction* of who might leave and an *understanding* of why.

**Target roles:** Pilots, Cardiologists, ICU Nurses, Anesthesiologists, Flight Paramedics, Data Engineers, Security Engineers.

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────────────────┐
│  Synthetic Data  │     │                  Snowflake                       │
│   Generator      │────▶│                                                  │
│  (Python/Faker)  │     │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
└─────────────────┘     │  │  BRONZE   │─▶│  SILVER   │─▶│   GOLD   │      │
                        │  │ raw load  │  │ cleaned + │  │ features │      │
                        │  │           │  │ sentiment │  │ + scores │      │
                        │  └──────────┘  └──────────┘  └──────────┘      │
                        │       │              │              │            │
                        │       │     Cortex AI_SENTIMENT     │            │
                        │       │     Cortex COMPLETE (LLM)   │            │
                        │       │              │              │            │
                        │       │         Cortex ML           │            │
                        │       │     Classification Model    │            │
                        │       │              │              │            │
                        └───────┼──────────────┼──────────────┼────────────┘
                                │              │              │
                                ▼              ▼              ▼
                        ┌──────────────────────────────────────────┐
                        │           Streamlit Dashboard             │
                        │  • Churn risk scores by role/region       │
                        │  • Exit survey theme analysis             │
                        │  • Feature importance                     │
                        │  • Manager intervention alerts            │
                        └──────────────────────────────────────────┘
```

### Pipeline Stages

1. **Generator** — Synthetic data with realistic attrition patterns across 7 hard-to-staff roles
2. **Bronze** — Raw CSV loaded into Snowflake staging table
3. **Silver** — Cleaned, typed, enriched with Cortex `AI_SENTIMENT()` on exit surveys
4. **Gold** — Feature-engineered table with Cortex ML classification model predictions
5. **Dashboard** — Streamlit app surfacing risk scores, themes, and intervention triggers

## Project Structure

```
snowflake-cortex-ml/
├── data/                  # Generated CSV files (gitignored)
├── generators/            # Synthetic data generation scripts
│   └── generate_attrition_data.py
├── pipeline/              # Snowflake SQL + Python pipeline code
├── dashboard/             # Streamlit app
├── docs/                  # Additional documentation
├── requirements.txt
├── .env.example           # Snowflake credential placeholders
└── .gitignore
```

## Quick Start

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Generate synthetic data
python generators/generate_attrition_data.py --records 2000

# 3. Copy .env.example to .env and fill in Snowflake credentials
cp .env.example .env

# 4. Run pipeline stages (coming soon)
# python pipeline/load_bronze.py
# python pipeline/transform_silver.py
# python pipeline/build_gold.py

# 5. Launch dashboard (coming soon)
# streamlit run dashboard/app.py
```

## Self-Learning Loop (Future State)

The end goal is a pipeline that improves itself:

1. **Predict** — Model scores active employees for churn risk
2. **Intervene** — Dashboard flags high-risk employees for manager action
3. **Observe** — Track whether interventions changed outcomes
4. **Retrain** — New outcomes feed back into the model as training data
5. **Refine** — Cortex LLM summarizes exit survey themes quarterly, surfacing emerging reasons that the structured model might miss

This creates a flywheel: better predictions → better interventions → better outcomes → better training data → better predictions.

## Tech Stack

- **Snowflake** — Warehouse, Cortex ML, Cortex LLM functions
- **Python** — Data generation, pipeline orchestration
- **Streamlit** — Interactive dashboard
- **Faker** — Realistic synthetic data
