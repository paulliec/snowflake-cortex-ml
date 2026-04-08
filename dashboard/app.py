"""Attrition ML Dashboard — Streamlit app connected to Snowflake Gold/Silver layers."""

import json
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


_COL_DISPLAY_NAMES = {
    "DAYS_SINCE_LAST_1ON1": "Days Since Last 1:1",
    "PTO_DAYS_UNUSED": "Unused PTO Days",
    "INTERNAL_TRANSFERS_REQUESTED": "Internal Transfer Requests",
}


def _pretty_col(name: str) -> str:
    """DAYS_SINCE_LAST_RAISE → Days Since Last Raise"""
    if name in _COL_DISPLAY_NAMES:
        return _COL_DISPLAY_NAMES[name]
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


def _classify_risk(score, p85, p65):
    """Assign risk tier based on percentile cutoffs from the score distribution.

    Top 15% (above p85) → Act Now, next 20% (p65-p85) → Early Warning, rest → Stable.
    """
    if pd.isna(score):
        return pd.Series(["—", "—", "—"])
    s = float(score)
    if s >= p85:
        return pd.Series(["🔴 Act Now", "< 90 days", "High Risk — Act Now"])
    if s >= p65:
        return pd.Series(["🟡 Early Warning", "3-6 months", "Early Warning — Monitor"])
    return pd.Series(["🟢 Stable", "Stable", "Stable"])


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

    # percentile-based risk tiers
    if "CHURN_RISK_SCORE" in out.columns:
        scores = out["CHURN_RISK_SCORE"].dropna()
        if len(scores) > 0:
            p85 = float(scores.quantile(0.85))
            p65 = float(scores.quantile(0.65))
        else:
            p85, p65 = 0.70, 0.40  # fallback if no scores
        tier_cols = out["CHURN_RISK_SCORE"].apply(_classify_risk, args=(p85, p65))
        tier_cols.columns = ["RISK_TIER", "TIME_WINDOW", "RISK_LABEL"]
        out = pd.concat([out, tier_cols], axis=1)
        # stash cutoffs for the forecast section
        out.attrs["_p85"] = p85
        out.attrs["_p65"] = p65

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


@st.cache_data(ttl=60)
def load_annotations():
    conn = get_connection()
    db = _snowflake_db()
    try:
        return _read_sql(conn, f"""
            SELECT EMPLOYEE_ID, ANNOTATION_TYPE, ANNOTATION_TEXT,
                   CREATED_BY, CREATED_AT
            FROM {db}.SILVER.HR_ANNOTATIONS
            WHERE ACTIVE = TRUE
            ORDER BY CREATED_AT DESC
        """)
    except Exception:
        return pd.DataFrame()


def _ensure_annotations_table():
    """Create HR_ANNOTATIONS if it doesn't exist yet."""
    conn = get_connection()
    cur = conn.cursor()
    db = _snowflake_db()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {db}.SILVER.HR_ANNOTATIONS (
            EMPLOYEE_ID     VARCHAR,
            ANNOTATION_TYPE VARCHAR,
            ANNOTATION_TEXT VARCHAR,
            CREATED_BY      VARCHAR,
            CREATED_AT      TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
            ACTIVE          BOOLEAN DEFAULT TRUE
        )
    """)
    cur.close()


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
    ["Overview", "Employee Attrition Risk",
     "HR Annotations", "Feature Importance", "Sentiment Analysis",
     "Ask the Data"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
_TIER_OPTIONS = ["All", "🔴 Act Now", "🟡 Early Warning", "🟢 Stable"]
_TIME_OPTIONS = ["All", "< 90 days", "3-6 months", "Stable"]
sidebar_tier = st.sidebar.selectbox("Risk Tier", _TIER_OPTIONS)
sidebar_time = st.sidebar.selectbox("Time Window", _TIME_OPTIONS)

st.sidebar.markdown("---")
if st.sidebar.button("Refresh data"):
    load_silver.clear()
    load_gold.clear()
    load_feature_importance.clear()
    load_annotations.clear()
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


def _apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply risk tier and time window sidebar filters."""
    out = df
    if sidebar_tier != "All" and "Risk Tier" in out.columns:
        out = out[out["Risk Tier"] == sidebar_tier]
    if sidebar_time != "All" and "Time Window" in out.columns:
        out = out[out["Time Window"] == sidebar_time]
    return out


# ============================================================
# Page 1: Overview
# ============================================================
if page == "Overview":
    st.title("Attrition Intelligence — Overview")
    st.caption("Risk tiers: 🔴 Act Now (top 15%) · 🟡 Early Warning (next 20%) · 🟢 Stable (bottom 65%) — ranked by relative risk")

    silver = load_silver()
    if silver is None or len(silver) == 0:
        st.error("Unable to load data from Snowflake")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    total = len(silver)
    churned = silver[silver["Churned"] == "Y"]
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Churned", f"{len(churned):,}")
    col3.metric("Churn Rate", f"{len(churned)/total:.1%}" if total else "—")
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
        role_stats["rate"] = role_stats.apply(
            lambda r: r["churned"] / r["total"] if r["total"] > 0 else 0, axis=1
        )
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
        region_stats["rate"] = region_stats.apply(
            lambda r: r["churned"] / r["total"] if r["total"] > 0 else 0, axis=1
        )

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

    # -- Workforce Impact Forecast --
    st.markdown("---")
    st.subheader("Expected Attrition by Role and Region")
    st.caption("Ranges show scenario spread around each tier threshold — use as a planning signal, not a guarantee")

    _all_silver = load_silver()  # unfiltered for forecast
    _forecast_df = _all_silver[_all_silver["Churn Risk Score"].notna()].copy()
    if len(_forecast_df) > 0 and "Risk Tier" in _forecast_df.columns:
        _p85 = _all_silver.attrs.get("_p85", 0.70)
        _p65 = _all_silver.attrs.get("_p65", 0.40)
        _forecast_rows = []
        # count employees per tier per role+region
        for (role, region), grp in _forecast_df.groupby(["Role", "Region"]):
            act_now = int((grp["Risk Tier"] == "🔴 Act Now").sum())
            early_warn = int((grp["Risk Tier"] == "🟡 Early Warning").sum())
            if act_now > 0:
                _forecast_rows.append({
                    "Tier": "🔴 Act Now",
                    "Role": role,
                    "Region": region,
                    "At-Risk Employees": act_now,
                    "Timeframe": "Next 90 days",
                })
            if early_warn > 0:
                _forecast_rows.append({
                    "Tier": "🟡 Early Warning",
                    "Role": role,
                    "Region": region,
                    "At-Risk Employees": early_warn,
                    "Timeframe": "3-6 months",
                })

        if _forecast_rows:
            forecast_table = pd.DataFrame(_forecast_rows)
            forecast_table = forecast_table.sort_values("At-Risk Employees", ascending=False)
            st.dataframe(
                forecast_table[["Tier", "Role", "Region", "At-Risk Employees", "Timeframe"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Tier": st.column_config.TextColumn(width="small"),
                },
            )
        else:
            st.info("No employees meet current risk thresholds.")
    else:
        st.info("Risk scores required for workforce forecast.")


# ============================================================
# Page 2: At-Risk Employees
# ============================================================
elif page == "Employee Attrition Risk":
    st.title("Employee Attrition Risk")
    st.caption("Employees ranked by 90-Day Attrition Risk Score")
    st.info(
        "Risk scores are ranked by relative attrition risk. Employees in the "
        "top 15% are flagged for immediate attention, the next 20% as early "
        "warning signals. Scores reflect model predictions on available data "
        "and should be used alongside manager judgment and HR context — "
        "not as a definitive assessment of any individual."
    )

    silver = load_silver()
    if silver is None or len(silver) == 0:
        st.error("Unable to load data from Snowflake")
        st.stop()

    scored = silver[silver["Churn Risk Score"].notna()].copy()

    if len(scored) == 0:
        st.warning(
            "No risk scores found. Run the full pipeline (includes prediction), "
            "then use **Refresh data** in the sidebar."
        )
        st.stop()

    # inline filters for role and region
    col1, col2 = st.columns(2)
    roles = ["All"] + sorted(scored["Role"].unique().tolist())
    regions = ["All"] + sorted(scored["Region"].unique().tolist())
    selected_role = col1.selectbox("Role", roles)
    selected_region = col2.selectbox("Region", regions)

    filtered = _apply_sidebar_filters(scored)
    if selected_role != "All":
        filtered = filtered[filtered["Role"] == selected_role]
    if selected_region != "All":
        filtered = filtered[filtered["Region"] == selected_region]

    if len(filtered) == 0:
        st.info("No employees match the selected filters.")
        st.stop()

    top25 = filtered.nlargest(25, "Churn Risk Score")

    # annotation indicator
    annot_df = load_annotations()
    if len(annot_df) > 0:
        # latest annotation type per employee
        annot_latest = annot_df.drop_duplicates(subset=["EMPLOYEE_ID"], keep="first")
        annot_map = dict(zip(annot_latest["EMPLOYEE_ID"], annot_latest["ANNOTATION_TYPE"]))
        top25["Notes"] = top25["Employee Id"].map(
            lambda eid: f"\U0001f4dd {annot_map[eid]}" if eid in annot_map else ""
        )
    else:
        top25["Notes"] = ""

    display_cols = [
        "Employee Id", "Employee Name", "Notes", "Role", "Department", "Region",
        "Risk Tier", "Risk Label", "Time Window",
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
                "90-Day Attrition Risk",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Risk Tier": st.column_config.TextColumn(width="small"),
            "Risk Label": st.column_config.TextColumn("Assessment"),
            "Time Window": st.column_config.TextColumn(width="small"),
            "Notes": st.column_config.TextColumn(width="medium"),
        },
    )

    st.caption(f"Showing top {len(top25)} of {len(filtered)} scored employees")


# ============================================================
# Page 3: HR Annotations
# ============================================================
elif page == "HR Annotations":
    st.title("HR Annotations")
    st.caption("Add qualitative context to at-risk employees — manager observations, known circumstances, retention actions")

    _ensure_annotations_table()

    silver = load_silver()
    if silver is None or len(silver) == 0:
        st.error("Unable to load data from Snowflake")
        st.stop()

    scored = silver[silver["Churn Risk Score"].notna()].copy()
    # show Act Now + Early Warning employees
    at_risk = scored[scored["Risk Tier"].isin(["🔴 Act Now", "🟡 Early Warning"])]
    at_risk = at_risk.nlargest(50, "Churn Risk Score")

    if len(at_risk) == 0:
        st.info("No at-risk employees to annotate.")
        st.stop()

    _ANNOTATION_TYPES = [
        "Manager flagged disengagement",
        "Internal transfer requested",
        "Compensation review overdue",
        "Recent personal hardship",
        "Actively interviewing - rumored",
        "High performer - retention priority",
        "Other",
    ]

    annot_df = load_annotations()

    for _, emp in at_risk.iterrows():
        emp_id = emp["Employee Id"]
        emp_name = emp.get("Employee Name", emp_id)
        tier = emp.get("Risk Tier", "")
        score = emp.get("Churn Risk Score", 0)
        role = emp.get("Role", "")
        region = emp.get("Region", "")

        with st.expander(f"{tier} **{emp_name}** — {role}, {region} (score: {score:.3f})"):
            # existing annotations
            if len(annot_df) > 0:
                emp_annots = annot_df[annot_df["EMPLOYEE_ID"] == emp_id]
                if len(emp_annots) > 0:
                    for _, a in emp_annots.iterrows():
                        ts = a.get("CREATED_AT", "")
                        by = a.get("CREATED_BY", "")
                        st.markdown(
                            f"**{a['ANNOTATION_TYPE']}** — {a.get('ANNOTATION_TEXT', '')}  \n"
                            f"<small>{by} · {ts}</small>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("---")

            # annotation form
            form_key = f"annot_{emp_id}"
            with st.form(form_key):
                atype = st.selectbox("Annotation type", _ANNOTATION_TYPES, key=f"type_{emp_id}")
                atext = st.text_input("Additional notes (optional)", key=f"text_{emp_id}")
                submitted = st.form_submit_button("Save annotation")
                if submitted:
                    conn = get_connection()
                    cur = conn.cursor()
                    db = _snowflake_db()
                    cur.execute(
                        f"INSERT INTO {db}.SILVER.HR_ANNOTATIONS "
                        f"(EMPLOYEE_ID, ANNOTATION_TYPE, ANNOTATION_TEXT, CREATED_BY) "
                        f"VALUES (%s, %s, %s, %s)",
                        (emp_id, atype, atext or None, "dashboard_user"),
                    )
                    cur.close()
                    load_annotations.clear()
                    st.success(f"Annotation saved for {emp_name}")
                    st.rerun()

    st.markdown("---")
    st.caption(
        "Annotations are reviewed at each model retraining cycle and "
        "high-signal patterns may be incorporated as features in future model versions."
    )


# ============================================================
# Page: Feature Importance
# ============================================================
elif page == "Feature Importance":
    st.title("Feature Importance — What Drives Attrition")
    st.caption("Model feature weights — these factors have the strongest signal for 90-day attrition risk")

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
# Page: Sentiment Analysis
# ============================================================
elif page == "Sentiment Analysis":
    st.title("Exit Survey Sentiment Analysis")
    st.caption("Post-separation survey analysis — informs future retention strategy, not current risk scoring")

    silver = load_silver()
    if silver is None or len(silver) == 0:
        st.error("Unable to load data from Snowflake")
        st.stop()

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


# ============================================================
# Page: Ask the Data (Cortex Analyst)
# ============================================================
elif page == "Ask the Data":
    import requests as _requests

    st.title("Ask the Data")
    st.caption(
        "Ask questions about attrition risk in plain English. "
        "Powered by Snowflake Cortex Analyst."
    )

    _SEMANTIC_MODEL_PATH = _ROOT / "cortex" / "semantic_model.yaml"

    _EXAMPLE_QUESTIONS = [
        "Which roles have the highest attrition risk?",
        "How many pilots are at high risk in the next 90 days?",
        "What factors drive attrition for ICU nurses?",
        "Which region has the most at-risk employees?",
        "Show me employees with high overtime and low manager ratings",
    ]

    # keyword → fallback SQL for when Cortex Analyst API is unavailable
    _FALLBACK_QUERIES = {
        "pilot": """
            SELECT UPPER(ROLE) AS ROLE, COUNT(*) AS HIGH_RISK_PILOTS,
                   ROUND(AVG(CHURN_RISK_SCORE), 3) AS AVG_RISK
            FROM ATTRITION_ML.SILVER.EMPLOYEE_CLEANSED
            WHERE UPPER(ROLE) = 'PILOT' AND CHURN_RISK_SCORE > 0.5
            GROUP BY ROLE""",
        "region": """
            SELECT REGION, COUNT(*) AS AT_RISK_COUNT,
                   ROUND(AVG(CHURN_RISK_SCORE), 3) AS AVG_RISK
            FROM ATTRITION_ML.SILVER.EMPLOYEE_CLEANSED
            WHERE CHURN_RISK_SCORE > 0.5
            GROUP BY REGION ORDER BY AT_RISK_COUNT DESC""",
        "icu": """
            SELECT ROUND(AVG(DAYS_SINCE_LAST_RAISE)) AS AVG_DAYS_SINCE_RAISE,
                   ROUND(AVG(MANAGER_RATING), 1) AS AVG_MANAGER_RATING,
                   ROUND(AVG(OVERTIME_HOURS_MONTHLY), 1) AS AVG_OVERTIME,
                   ROUND(AVG(DAYS_SINCE_LAST_1ON1)) AS AVG_DAYS_SINCE_1ON1,
                   ROUND(AVG(PTO_DAYS_UNUSED), 1) AS AVG_UNUSED_PTO
            FROM ATTRITION_ML.SILVER.EMPLOYEE_CLEANSED
            WHERE UPPER(ROLE) = 'ICU NURSE' AND CHURN_RISK_SCORE > 0.5""",
        "nurse": None,  # alias — matched after icu
        "overtime": """
            SELECT EMPLOYEE_NAME, ROLE, REGION,
                   OVERTIME_HOURS_MONTHLY, MANAGER_RATING,
                   ROUND(CHURN_RISK_SCORE, 3) AS CHURN_RISK_SCORE
            FROM ATTRITION_ML.SILVER.EMPLOYEE_CLEANSED
            WHERE OVERTIME_HOURS_MONTHLY > 20 AND MANAGER_RATING < 3
            ORDER BY CHURN_RISK_SCORE DESC LIMIT 25""",
    }
    _FALLBACK_DEFAULT = """
        SELECT ROLE, COUNT(*) AS TOTAL,
               SUM(CASE WHEN CHURN_RISK_SCORE > 0.5 THEN 1 ELSE 0 END) AS HIGH_RISK,
               ROUND(AVG(CHURN_RISK_SCORE), 3) AS AVG_RISK_SCORE
        FROM ATTRITION_ML.SILVER.EMPLOYEE_CLEANSED
        WHERE CHURN_RISK_SCORE IS NOT NULL
        GROUP BY ROLE ORDER BY AVG_RISK_SCORE DESC"""

    def _fallback_query(question: str):
        """Match question keywords to verified SQL and run it."""
        q = question.lower()
        sql = None
        for kw, fallback_sql in _FALLBACK_QUERIES.items():
            if kw in q:
                sql = fallback_sql or _FALLBACK_QUERIES.get("icu", _FALLBACK_DEFAULT)
                break
        if sql is None:
            sql = _FALLBACK_DEFAULT
        conn = get_connection()
        df = pd.read_sql(sql, conn)
        df.columns = [str(c).upper() for c in df.columns]
        return sql.strip(), df

    def _get_analyst_host() -> str:
        """Build the Snowflake account host for REST API calls."""
        try:
            acct = st.secrets["snowflake"]["account"]
        except (KeyError, FileNotFoundError):
            acct = os.environ.get("SNOWFLAKE_ACCOUNT", "")
        # account may already include region (e.g. xy12345.us-east-1)
        acct = acct.strip()
        if acct.endswith(".snowflakecomputing.com"):
            return acct
        return f"{acct}.snowflakecomputing.com"

    def _get_auth_token(conn) -> str | None:
        """Extract session token from an open snowflake connection."""
        # snowflake-connector-python stores the token on the rest adapter
        rest = getattr(conn, "_rest", None) or getattr(conn, "rest", None)
        if rest is None:
            return None
        # try the most common attribute names across connector versions
        for attr in ("token", "_token", "master_token"):
            tok = getattr(rest, attr, None)
            if tok:
                return tok
        return None

    def _call_cortex_analyst_rest(messages: list, semantic_model: str) -> dict | None:
        """POST to the Cortex Analyst REST endpoint with full conversation history."""
        conn = get_connection()
        token = _get_auth_token(conn)
        if not token:
            return None

        host = _get_analyst_host()
        url = f"https://{host}/api/v2/cortex/analyst/message"
        headers = {
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {"messages": messages, "semantic_model": semantic_model}

        try:
            resp = _requests.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            st.toast(f"Cortex Analyst API returned {resp.status_code}")
        except Exception as e:
            st.toast(f"Cortex Analyst REST error: {type(e).__name__}")
        return None

    def _call_cortex_analyst_sql(messages: list, semantic_model: str) -> dict | None:
        """Fallback: call Cortex Analyst via SQL function."""
        request_body = {"messages": messages, "semantic_model": semantic_model}
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE('analyst', PARSE_JSON(%s))",
                (json.dumps(request_body),),
            )
            raw = cur.fetchone()[0]
            return json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass
        try:
            cur.execute(
                "SELECT SNOWFLAKE.CORTEX.ANALYST(PARSE_JSON(%s))",
                (json.dumps(request_body),),
            )
            raw = cur.fetchone()[0]
            return json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return None
        finally:
            cur.close()

    def _call_analyst(messages: list) -> dict | None:
        """Try REST API first, then SQL function, return parsed response or None."""
        semantic_model = _SEMANTIC_MODEL_PATH.read_text()
        resp = _call_cortex_analyst_rest(messages, semantic_model)
        if resp is not None:
            return resp
        resp = _call_cortex_analyst_sql(messages, semantic_model)
        if resp is not None:
            return resp
        st.toast("Cortex Analyst unavailable — using verified queries.")
        return None

    def _extract_content(response: dict) -> list:
        """Pull the content list from whichever response shape we get."""
        msg = response.get("message", response)
        return msg.get("content", [])

    def _render_content(content: list):
        """Render analyst content blocks — text, sql, suggestions."""
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                st.markdown(block["text"])
            elif btype == "sql":
                sql = block.get("statement", block.get("sql", ""))
                st.code(sql, language="sql")
                try:
                    conn = get_connection()
                    df = pd.read_sql(sql, conn)
                    df.columns = [str(c).upper() for c in df.columns]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Query failed: {e}")
            elif btype == "suggestions":
                suggestions = block.get("suggestions", [])
                if suggestions:
                    st.caption("Follow-up questions:")
                    cols = st.columns(min(len(suggestions), 3))
                    for j, s in enumerate(suggestions):
                        if cols[j % len(cols)].button(s, key=f"suggest_{hash(s)}"):
                            _submit_question(s)

    def _submit_question(question: str):
        """Add user question to history, call analyst, store response, rerun."""
        # build API message list from conversation history
        messages = st.session_state.analyst_messages.copy()
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": question}],
        })

        # store for display
        st.session_state.analyst_display.append({"role": "user", "text": question})

        resp = _call_analyst(messages)
        if resp is not None:
            content = _extract_content(resp)
            # add analyst turn to API message history
            st.session_state.analyst_messages = messages
            st.session_state.analyst_messages.append({
                "role": "analyst",
                "content": content,
            })
            st.session_state.analyst_display.append({
                "role": "assistant", "content": content,
            })
        else:
            # keyword fallback — no conversation context but always works
            sql, df = _fallback_query(question)
            st.session_state.analyst_messages = messages  # keep user turn
            st.session_state.analyst_display.append({
                "role": "assistant", "fallback_sql": sql, "fallback_df": df,
            })
        st.rerun()

    def _render_display_msg(msg: dict):
        """Render a single display message."""
        if "content" in msg:
            _render_content(msg["content"])
        elif "fallback_sql" in msg:
            st.code(msg["fallback_sql"], language="sql")
            st.dataframe(msg["fallback_df"], use_container_width=True, hide_index=True)
        else:
            st.markdown(msg.get("text", ""))

    # -- session state: API messages (full history for multi-turn) and display messages
    if "analyst_messages" not in st.session_state:
        st.session_state.analyst_messages = []
    if "analyst_display" not in st.session_state:
        st.session_state.analyst_display = []

    # clear conversation button
    if st.button("Clear conversation"):
        st.session_state.analyst_messages = []
        st.session_state.analyst_display = []
        st.rerun()

    # example question chips (only when conversation is empty)
    if len(st.session_state.analyst_display) == 0:
        chip_cols = st.columns(len(_EXAMPLE_QUESTIONS))
        for i, q in enumerate(_EXAMPLE_QUESTIONS):
            if chip_cols[i].button(q, key=f"chip_{i}", use_container_width=True):
                _submit_question(q)

    # render conversation
    for msg in st.session_state.analyst_display:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["text"])
        else:
            with st.chat_message("assistant"):
                _render_display_msg(msg)

    # chat input
    if user_input := st.chat_input("Ask a question about your workforce data..."):
        _submit_question(user_input)

    st.markdown("---")
    st.caption(
        "Cortex Analyst interprets your question and queries the underlying "
        "data directly. Results reflect the synthetic dataset used in this demo."
    )
