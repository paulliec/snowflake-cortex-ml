"""Attrition ML Dashboard — Streamlit app connected to Snowflake Gold/Silver layers."""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.connection import get_connection as _snowflake_connect
from pipeline.connection import use_ml_schema


def _snowflake_db() -> str:
    try:
        return st.secrets["snowflake"].get("database", "ATTRITION_ML")
    except (KeyError, FileNotFoundError):
        return os.environ.get("SNOWFLAKE_DATABASE", "ATTRITION_ML").strip().strip('"')


def _read_sql(conn, sql: str) -> pd.DataFrame:
    df = pd.read_sql(sql, conn)
    df.columns = [str(c).upper() for c in df.columns]
    return df


def _pretty_col(name: str) -> str:
    """DAYS_SINCE_LAST_RAISE → Days Since Last Raise"""
    return name.replace("_", " ").title()


_KEEP_UPPER = {"ICU", "YTD"}


def _title_case_values(s: pd.Series) -> pd.Series:
    """MIDWEST → Midwest, DATA_ENGINEER → Data Engineer, ICU NURSE → ICU Nurse"""
    def _fmt(v):
        if not isinstance(v, str):
            return v
        words = v.replace("_", " ").split()
        return " ".join(w if w in _KEEP_UPPER else w.title() for w in words)
    return s.map(_fmt)


# columns whose cell values should be title-cased for display
_TITLE_CASE_COLS = {"ROLE", "DEPARTMENT", "REGION", "EMPLOYEE_NAME"}


def _prepare_employee_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types so risk scores and churn flags work with Snowflake + pandas."""
    out = df.copy()
    if "CHURN_RISK_SCORE" in out.columns:
        out["CHURN_RISK_SCORE"] = pd.to_numeric(out["CHURN_RISK_SCORE"], errors="coerce")
    if "CHURNED" in out.columns:
        def _churn_flag(v):
            if pd.isna(v):
                return v
            if isinstance(v, (bool, int)) and not isinstance(v, str):
                return "Y" if bool(v) else "N"
            s = str(v).strip().upper()
            if s in ("TRUE", "1", "YES"):
                return "Y"
            if s in ("FALSE", "0", "NO"):
                return "N"
            return s

        out["CHURNED"] = out["CHURNED"].map(_churn_flag)
    if "DEPARTMENT" in out.columns:
        out["DEPARTMENT"] = out["DEPARTMENT"].fillna("(Unknown)")

    for col in _TITLE_CASE_COLS:
        if col in out.columns:
            out[col] = _title_case_values(out[col])

    out.columns = [_pretty_col(c) for c in out.columns]
    return out

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
    return _snowflake_connect()


@st.cache_data(ttl=120)
def load_silver():
    conn = get_connection()
    db = _snowflake_db()
    # qualify with database so we are not dependent on session CURRENT_DATABASE
    df = _read_sql(conn, f"SELECT * FROM {db}.SILVER.EMPLOYEE_CLEANSED")
    return _prepare_employee_df(df)


@st.cache_data(ttl=120)
def load_gold():
    conn = get_connection()
    db = _snowflake_db()
    return _read_sql(conn, f"SELECT * FROM {db}.GOLD.EMPLOYEE_ML_READY")


@st.cache_data(ttl=600)
def load_feature_importance():
    conn = get_connection()
    try:
        cur = conn.cursor()
        use_ml_schema(cur)
        cur.execute("CALL ATTRITION_CLASSIFIER!SHOW_FEATURE_IMPORTANCE()")
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
        cur.close()
        return pd.DataFrame(rows, columns=cols)
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
if st.sidebar.button("Refresh data"):
    load_silver.clear()
    load_gold.clear()
    load_feature_importance.clear()
    st.rerun()

with st.sidebar.expander("Data check"):
    st.caption(f"Database: `{_snowflake_db()}`")
    try:
        _s = load_silver()
        _n = len(_s)
        _k = int(_s["Churn Risk Score"].notna().sum()) if "Churn Risk Score" in _s.columns else 0
        st.caption(f"Silver rows: {_n:,} · with risk score: {_k:,}")
    except Exception as e:
        st.caption(str(e)[:200])

st.sidebar.caption("Attrition ML Pipeline v1.0")


# ============================================================
# Page 1: Overview
# ============================================================
if page == "Overview":
    st.title("Attrition Intelligence — Overview")

    silver = load_silver()

    col1, col2, col3, col4 = st.columns(4)
    total = len(silver)
    churned = silver[silver["Churned"] == "Y"]
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Churned", f"{len(churned):,}")
    col3.metric("Churn Rate", f"{len(churned)/total:.1%}")
    scored = silver["Churn Risk Score"].notna().sum()
    col4.metric("Risk Scored", f"{scored:,}")

    st.markdown("---")
    left, right = st.columns(2)

    # churn rate by role
    with left:
        st.subheader("Churn Rate by Role")
        role_stats = silver.groupby("Role").agg(
            total=("Churned", "count"),
            churned=("Churned", lambda x: (x == "Y").sum()),
        ).reset_index()
        role_stats["rate"] = role_stats["churned"] / role_stats["total"]
        role_stats = role_stats.sort_values("rate", ascending=True)

        fig = px.bar(
            role_stats, x="rate", y="Role", orientation="h",
            color="rate", color_continuous_scale="RdYlGn_r",
            labels={"rate": "Churn Rate", "Role": ""},
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
        region_stats = silver.groupby("Region").agg(
            total=("Churned", "count"),
            churned=("Churned", lambda x: (x == "Y").sum()),
        ).reset_index()
        region_stats["rate"] = region_stats["churned"] / region_stats["total"]

        fig = px.bar(
            region_stats, x="Region", y="rate",
            color="rate", color_continuous_scale="RdYlGn_r",
            labels={"rate": "Churn Rate", "Region": ""},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=300,
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Churn Risk Score by Department")
    scored_df = silver[silver["Churn Risk Score"].notna()]
    if len(scored_df) > 0:
        dept_risk = scored_df.groupby("Department", dropna=False)["Churn Risk Score"].mean().reset_index()
        dept_risk = dept_risk.sort_values("Churn Risk Score", ascending=True)
        fig = px.bar(
            dept_risk, x="Churn Risk Score", y="Department", orientation="h",
            color="Churn Risk Score", color_continuous_scale="RdYlGn_r",
            labels={"Churn Risk Score": "Avg Risk Score", "Department": ""},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=250,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No churn risk scores found. Run `python -m pipeline.run_pipeline` "
            "(prediction step), then **Refresh data** here."
        )


# ============================================================
# Page 2: At-Risk Employees
# ============================================================
elif page == "At-Risk Employees":
    st.title("At-Risk Employees")

    silver = load_silver()
    scored = silver[silver["Churn Risk Score"].notna()].copy()

    if len(scored) == 0:
        st.warning(
            "No risk scores found. Run the full pipeline (includes prediction), "
            "then use **Refresh data** in the sidebar."
        )
        st.stop()

    # filters
    col1, col2 = st.columns(2)
    roles = ["All"] + sorted(scored["Role"].unique().tolist())
    regions = ["All"] + sorted(scored["Region"].unique().tolist())
    selected_role = col1.selectbox("Role", roles)
    selected_region = col2.selectbox("Region", regions)

    filtered = scored
    if selected_role != "All":
        filtered = filtered[filtered["Role"] == selected_role]
    if selected_region != "All":
        filtered = filtered[filtered["Region"] == selected_region]

    top25 = filtered.nlargest(25, "Churn Risk Score")

    display_cols = [
        "Employee Id", "Employee Name", "Role", "Department", "Region",
        "Tenure Years", "Manager Rating", "Overtime Hours Monthly",
        "Days Since Last Raise", "Churn Risk Score",
    ]
    display_cols = [c for c in display_cols if c in top25.columns]
    st.dataframe(
        top25[display_cols],
        use_container_width=True,
        height=700,
        column_config={
            "Churn Risk Score": st.column_config.ProgressColumn(
                "Churn Risk",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
        },
    )

    st.caption(f"Showing top {len(top25)} of {len(filtered)} scored employees")


# ============================================================
# Page 3: Feature Importance
# ============================================================
elif page == "Feature Importance":
    st.title("Feature Importance — What Drives Attrition")

    fi = load_feature_importance()
    if fi is not None and len(fi) > 0:
        fi.columns = [str(c).upper() for c in fi.columns]
        # post-event signal, not actionable for retention
        fi = fi[fi["FEATURE"] != "EXIT_SURVEY_SENTIMENT"]
        fi["FEATURE"] = fi["FEATURE"].map(_pretty_col)
        fi = fi.sort_values("SCORE", ascending=True)
        fig = px.bar(
            fi, x="SCORE", y="FEATURE", orientation="h",
            color="SCORE", color_continuous_scale="Viridis",
            labels={"SCORE": "Importance", "FEATURE": "Feature"},
        )
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=0), height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Exit Survey Sentiment is excluded — it's only available after "
            "separation and cannot inform retention decisions."
        )
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
    churned = silver[silver["Churned"] == "Y"].copy()

    if churned["Exit Survey Sentiment"].isna().all():
        st.warning("Sentiment scores not yet populated. Run Bronze → Silver transformation.")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Score Distribution")
        fig = px.histogram(
            churned, x="Exit Survey Sentiment", nbins=30,
            color_discrete_sequence=["#636EFA"],
            labels={"Exit Survey Sentiment": "Sentiment Score"},
        )
        fig.update_layout(
            margin=dict(l=0, r=20, t=10, b=0), height=350,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Sentiment by Role")
        role_sent = churned.groupby("Role")["Exit Survey Sentiment"].mean().reset_index()
        role_sent = role_sent.sort_values("Exit Survey Sentiment")
        fig = px.bar(
            role_sent, x="Exit Survey Sentiment", y="Role", orientation="h",
            color="Exit Survey Sentiment", color_continuous_scale="RdYlGn",
            labels={"Exit Survey Sentiment": "Avg Sentiment", "Role": ""},
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

    texts = churned["Exit Survey Text"].dropna().str.lower()
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
