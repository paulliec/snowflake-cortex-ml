"""Attrition ML Dashboard — Streamlit app connected to Snowflake Gold/Silver layers."""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import snowflake.connector

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(
    page_title="Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- dark, clean theme overrides
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; color: #a0aec0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database="ATTRITION_ML",
        role=os.environ.get("SNOWFLAKE_ROLE"),
    )


@st.cache_data(ttl=300)
def load_silver():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM SILVER.EMPLOYEE_CLEANSED", conn)


@st.cache_data(ttl=300)
def load_gold():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM GOLD.EMPLOYEE_ML_READY", conn)


@st.cache_data(ttl=600)
def load_feature_importance():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT ATTRITION_CLASSIFIER!SHOW_FEATURE_IMPORTANCE()")
        raw = cur.fetchone()[0]
        cur.close()
        return pd.read_json(raw)
    except Exception:
        return None


# ============================================================
# Sidebar
# ============================================================
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "At-Risk Employees", "Feature Importance", "Sentiment Analysis"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Attrition ML Pipeline v1.0")


# ============================================================
# Page 1: Overview
# ============================================================
if page == "Overview":
    st.title("Attrition Intelligence — Overview")

    silver = load_silver()

    col1, col2, col3, col4 = st.columns(4)
    total = len(silver)
    churned = silver[silver["CHURNED"] == "Y"]
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Churned", f"{len(churned):,}")
    col3.metric("Churn Rate", f"{len(churned)/total:.1%}")
    scored = silver["CHURN_RISK_SCORE"].notna().sum()
    col4.metric("Risk Scored", f"{scored:,}")

    st.markdown("---")
    left, right = st.columns(2)

    # churn rate by role
    with left:
        st.subheader("Churn Rate by Role")
        role_stats = silver.groupby("ROLE").agg(
            total=("CHURNED", "count"),
            churned=("CHURNED", lambda x: (x == "Y").sum()),
        ).reset_index()
        role_stats["rate"] = role_stats["churned"] / role_stats["total"]
        role_stats = role_stats.sort_values("rate", ascending=True)

        fig = px.bar(
            role_stats, x="rate", y="ROLE", orientation="h",
            color="rate", color_continuous_scale="RdYlGn_r",
            labels={"rate": "Churn Rate", "ROLE": ""},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=300,
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # churn rate by region
    with right:
        st.subheader("Churn Rate by Region")
        region_stats = silver.groupby("REGION").agg(
            total=("CHURNED", "count"),
            churned=("CHURNED", lambda x: (x == "Y").sum()),
        ).reset_index()
        region_stats["rate"] = region_stats["churned"] / region_stats["total"]

        fig = px.bar(
            region_stats, x="REGION", y="rate",
            color="rate", color_continuous_scale="RdYlGn_r",
            labels={"rate": "Churn Rate", "REGION": ""},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=300,
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # avg risk score by department
    st.subheader("Average Churn Risk Score by Department")
    scored_df = silver[silver["CHURN_RISK_SCORE"].notna()]
    if len(scored_df) > 0:
        dept_risk = scored_df.groupby("DEPARTMENT")["CHURN_RISK_SCORE"].mean().reset_index()
        dept_risk = dept_risk.sort_values("CHURN_RISK_SCORE", ascending=True)
        fig = px.bar(
            dept_risk, x="CHURN_RISK_SCORE", y="DEPARTMENT", orientation="h",
            color="CHURN_RISK_SCORE", color_continuous_scale="RdYlGn_r",
            labels={"CHURN_RISK_SCORE": "Avg Risk Score", "DEPARTMENT": ""},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=250,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the prediction pipeline to populate risk scores.")


# ============================================================
# Page 2: At-Risk Employees
# ============================================================
elif page == "At-Risk Employees":
    st.title("At-Risk Employees")

    silver = load_silver()
    scored = silver[silver["CHURN_RISK_SCORE"].notna()].copy()

    if len(scored) == 0:
        st.warning("No risk scores yet. Run the prediction pipeline first.")
        st.stop()

    # filters
    col1, col2 = st.columns(2)
    roles = ["All"] + sorted(scored["ROLE"].unique().tolist())
    regions = ["All"] + sorted(scored["REGION"].unique().tolist())
    selected_role = col1.selectbox("Role", roles)
    selected_region = col2.selectbox("Region", regions)

    filtered = scored
    if selected_role != "All":
        filtered = filtered[filtered["ROLE"] == selected_role]
    if selected_region != "All":
        filtered = filtered[filtered["REGION"] == selected_region]

    top25 = filtered.nlargest(25, "CHURN_RISK_SCORE")

    display_cols = [
        "EMPLOYEE_ID", "EMPLOYEE_NAME", "ROLE", "DEPARTMENT", "REGION",
        "TENURE_YEARS", "MANAGER_RATING", "OVERTIME_HOURS_MONTHLY",
        "DAYS_SINCE_LAST_RAISE", "CHURN_RISK_SCORE",
    ]
    st.dataframe(
        top25[display_cols].style.background_gradient(
            subset=["CHURN_RISK_SCORE"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
        height=700,
    )

    st.caption(f"Showing top {len(top25)} of {len(filtered)} scored employees")


# ============================================================
# Page 3: Feature Importance
# ============================================================
elif page == "Feature Importance":
    st.title("Feature Importance — What Drives Attrition")

    fi = load_feature_importance()
    if fi is not None and len(fi) > 0:
        # Cortex returns feature name + importance score
        fi = fi.sort_values(fi.columns[-1], ascending=True)
        fig = px.bar(
            fi, x=fi.columns[-1], y=fi.columns[0], orientation="h",
            color=fi.columns[-1], color_continuous_scale="Viridis",
            labels={fi.columns[-1]: "Importance", fi.columns[0]: "Feature"},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance available after model training.")
        st.markdown("""
        **Expected top drivers based on data generation patterns:**
        - Manager Rating
        - Days Since Last Raise
        - Overtime Hours Monthly
        - Performance Score
        - Role (Pilot, ICU Nurse have higher base rates)
        """)


# ============================================================
# Page 4: Sentiment Analysis
# ============================================================
elif page == "Sentiment Analysis":
    st.title("Exit Survey Sentiment Analysis")

    silver = load_silver()
    churned = silver[silver["CHURNED"] == "Y"].copy()

    if churned["EXIT_SURVEY_SENTIMENT"].isna().all():
        st.warning("Sentiment scores not yet populated. Run Bronze → Silver transformation.")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Score Distribution")
        fig = px.histogram(
            churned, x="EXIT_SURVEY_SENTIMENT", nbins=30,
            color_discrete_sequence=["#636EFA"],
            labels={"EXIT_SURVEY_SENTIMENT": "Sentiment Score"},
        )
        fig.update_layout(
            margin=dict(l=0, r=20, t=10, b=0), height=350,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Sentiment by Role")
        role_sent = churned.groupby("ROLE")["EXIT_SURVEY_SENTIMENT"].mean().reset_index()
        role_sent = role_sent.sort_values("EXIT_SURVEY_SENTIMENT")
        fig = px.bar(
            role_sent, x="EXIT_SURVEY_SENTIMENT", y="ROLE", orientation="h",
            color="EXIT_SURVEY_SENTIMENT", color_continuous_scale="RdYlGn",
            labels={"EXIT_SURVEY_SENTIMENT": "Avg Sentiment", "ROLE": ""},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # keyword frequency in exit surveys
    st.subheader("Common Themes in Exit Surveys")
    keywords = {
        "compensation/pay": ["pay", "salary", "compensation", "comp", "money", "offer"],
        "burnout": ["burnout", "burn out", "exhausted", "empty", "fatigue", "mental health"],
        "work-life balance": ["hours", "schedule", "family", "overtime", "weekend", "work-life"],
        "management": ["manager", "leadership", "micromanag", "feedback", "mentorship", "credit"],
        "opportunity": ["opportunity", "recruited", "career", "growth", "step up", "title"],
        "relocation": ["relocat", "moved", "transfer", "spouse", "partner", "parents"],
    }

    texts = churned["EXIT_SURVEY_TEXT"].dropna().str.lower()
    theme_counts = {}
    for theme, terms in keywords.items():
        theme_counts[theme] = sum(texts.str.contains("|".join(terms), regex=True))

    theme_df = pd.DataFrame(
        sorted(theme_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["Theme", "Mentions"],
    )
    fig = px.bar(
        theme_df, x="Mentions", y="Theme", orientation="h",
        color="Mentions", color_continuous_scale="Blues",
    )
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=10, b=0), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
