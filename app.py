from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import (
    guess_positive_label, score_customers, suggest_excluded_columns, train_models_from_frame,
)

# NIGGESH brand palette
VIOLET = "#5146D8"
CORAL = "#FF6B6B"
TEAL = "#20BFA9"
YELLOW = "#FFB547"
INK = "#20243A"
MUTED = "#667085"
PALETTE = [VIOLET, TEAL, CORAL, YELLOW, "#8892A6", "#142850"]

st.set_page_config(page_title="NIGGESH Intelligence", page_icon="✨", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Nunito:wght@700;800;900&display=swap');
:root {{ --violet:{VIOLET}; --coral:{CORAL}; --teal:{TEAL}; --yellow:{YELLOW}; --ink:{INK}; --muted:{MUTED}; --line:rgba(81,70,216,.14); --glass:rgba(255,255,255,.72); }}
html,body,[class*="css"] {{font-family:'Inter',sans-serif;}}
.stApp {{background-color:#FCFCFE;color:{INK};}}
section[data-testid="stMain"] {{background-color:#FCFCFE;background-image:repeating-linear-gradient(90deg,transparent 0 88px,rgba(255,255,255,.72) 88px 91px),repeating-linear-gradient(0deg,transparent 0 88px,rgba(255,255,255,.72) 88px 91px),radial-gradient(circle at 84% 30%,rgba(81,70,216,.16) 0 30px,transparent 32px),conic-gradient(from 45deg at 23% 58%,transparent 0 25%,rgba(255,107,107,.16) 25% 50%,transparent 50% 100%),radial-gradient(circle at 43% 74%,rgba(32,191,169,.12) 0 38px,transparent 40px),linear-gradient(112deg,#F8ECF4,#F2F1FA);background-size:100% 280px,100% 280px,135% 330px,130% 330px,125% 320px,100% 280px;background-position:0 48px,0 48px,-12% 48px,2% 48px,-10% 48px,0 48px;background-repeat:no-repeat;animation:niggeshShapesDrift 24s ease-in-out infinite alternate;}}
@keyframes niggeshShapesDrift {{0%{{background-position:0 48px,0 48px,-12% 48px,2% 48px,-10% 48px,0 48px;}}50%{{background-position:0 48px,0 48px,-6% 60px,-4% 36px,-4% 36px,0 48px;}}100%{{background-position:0 48px,0 48px,-15% 48px,2% 62px,-12% 60px,0 48px;}}}}
.block-container {{padding-top:1.15rem!important;padding-bottom:3rem!important;max-width:1500px;}}
h1,h2,h3 {{font-family:'Nunito','Inter',sans-serif!important;color:{INK};letter-spacing:-.025em;}}
h1 {{color:{INK};-webkit-text-fill-color:{INK};}}
.workspace-header {{background:rgba(255,255,255,.96);border:1px solid #E9EAF1;border-radius:15px;padding:.9rem 1.25rem .65rem;margin-bottom:.65rem;box-shadow:0 5px 18px rgba(34,38,72,.045);}}
.dataset-note {{background:rgba(255,255,255,.96);border:1px solid #E9EAF1;border-radius:10px;padding:.65rem .9rem;margin:.4rem 0 1rem;color:#50596B;font-size:.88rem;}}
[data-testid="stSidebar"] {{background:rgba(255,255,255,.91);border-right:1px solid rgba(32,36,58,.08);backdrop-filter:blur(22px);}}
[data-testid="stSidebar"]>div:first-child {{background:transparent;}}
[data-testid="stSidebar"]>div:first-child {{padding-top:1rem;}}
[data-testid="stSidebar"] [data-testid="stImage"] {{display:flex;align-items:center;justify-content:center;min-height:142px;padding:.65rem .45rem;margin:.15rem 0 1.2rem;border-radius:18px;background:linear-gradient(120deg,rgba(81,70,216,.11),rgba(32,191,169,.09),rgba(255,107,107,.10),rgba(255,181,71,.12),rgba(81,70,216,.11));background-size:300% 300%;animation:niggeshBrandShift 18s ease-in-out infinite;}}
[data-testid="stSidebar"] [data-testid="stImage"] img {{display:block;width:98%!important;max-width:460px!important;height:auto;margin-inline:auto;}}
@keyframes niggeshBrandShift {{0%{{background-position:0% 50%;}}50%{{background-position:100% 50%;}}100%{{background-position:0% 50%;}}}}
@media(prefers-reduced-motion:reduce){{.stApp,[data-testid="stMain"],[data-testid="stSidebar"] [data-testid="stImage"]{{animation:none;}}*,*::before,*::after{{transition:none!important;}}}}
[data-testid="stSidebar"] [data-testid="stRadio"]>label {{font-size:.68rem;font-weight:800;letter-spacing:.14em;color:#9298A8;margin:1.1rem .2rem .55rem;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {{gap:.2rem;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {{border:1px solid transparent;border-radius:10px;padding:.48rem .66rem;color:#555D70;transition:background .15s ease,border-color .15s ease,transform .15s ease;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {{background:#F7F6FF;border-color:#ECEAFF;color:{VIOLET};transform:translateX(2px);}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{background:#F2F0FF;border-color:#E7E3FF;box-shadow:inset 3px 0 0 {VIOLET};}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {{color:{VIOLET};font-weight:700;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {{font-size:.89rem;line-height:1.25;}}
div[data-testid="stMetric"] {{background:#FFFFFF;border:1px solid #E9EAF1;outline:none;padding:1rem 1.1rem;border-radius:14px;box-shadow:0 5px 18px rgba(34,38,72,.045);backdrop-filter:blur(10px);transition:transform .18s ease,box-shadow .18s ease;}}
div[data-testid="stMetric"]:hover {{transform:translateY(-2px);box-shadow:0 18px 42px rgba(45,48,100,.12),inset 0 1px 0 rgba(255,255,255,.98);}}
div[data-testid="stMetric"] label {{color:#777F91;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{color:{INK};font-family:'Nunito',sans-serif;font-weight:900;font-size:1.9rem;}}
.nig-card {{background:#FFFFFF;border:1px solid #E9EAF1;outline:none;padding:1.1rem 1.25rem;border-radius:15px;box-shadow:0 6px 20px rgba(45,48,100,.05);margin-bottom:.8rem;}}
div[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {{background:#FFFFFF;border:1px solid #E8EAF2;border-radius:16px;padding:.35rem .45rem;box-shadow:0 7px 22px rgba(45,48,100,.045);}}
[data-testid="stVerticalBlockBorderWrapper"] {{background:#FFFFFF;border-color:#E9EAF1!important;border-radius:15px!important;box-shadow:0 6px 20px rgba(45,48,100,.045);}}
[data-testid="stDataFrame"] {{border:1px solid rgba(81,70,216,.12);border-radius:15px;overflow:hidden;box-shadow:0 10px 26px rgba(45,48,100,.06);}}
.stButton>button {{background:{VIOLET};color:white;border:1px solid {VIOLET};border-radius:10px;font-weight:700;box-shadow:0 5px 14px rgba(81,70,216,.16);transition:transform .16s ease,box-shadow .16s ease,filter .16s ease;}}
.stButton>button:hover {{color:white;transform:translateY(-1px);filter:saturate(1.12);box-shadow:0 12px 26px rgba(81,70,216,.27);}}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,div[data-baseweb="textarea"]>div {{background:rgba(255,255,255,.78);border-color:rgba(81,70,216,.17);border-radius:12px;}}
[data-testid="stFileUploaderDropzone"] {{background:rgba(255,255,255,.58);border:1px dashed rgba(81,70,216,.30);border-radius:16px;}}
@media(max-width:800px){{.block-container{{padding-left:1rem!important;padding-right:1rem!important;}}}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(BytesIO(data))
    else:
        frame = pd.read_csv(BytesIO(data))
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError("This file has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique. Please rename duplicate columns and upload again.")
    return frame


@st.cache_data(show_spinner=False)
def get_prototype_frame() -> pd.DataFrame:
    """Create a neutral sample dataset for the initial product prototype."""
    rng = np.random.default_rng(27)
    rows = 900
    tenure = np.clip(rng.gamma(shape=2.2, scale=12, size=rows), 1, 72).round().astype(int)
    spend = np.clip(rng.normal(72, 24, size=rows), 18, 160).round(2)
    engagement = np.clip(rng.normal(64, 21, size=rows), 0, 100).round(1)
    support_issues = rng.poisson(.35, size=rows)
    plan_type = rng.choice(["Monthly", "Annual", "Two-year"], size=rows, p=[.56, .29, .15])
    churn_logit = (
        -2.0 + (tenure <= 6) * .8 + (engagement < 35) * 1.15
        + (spend > 110) * .35 + (support_issues >= 2) * .5
        + (plan_type == "Monthly") * .75
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    return pd.DataFrame({
        "Customer ID": [f"SAMPLE-{index:04d}" for index in range(1, rows + 1)],
        "Tenure Months": tenure,
        "Monthly Spend": spend,
        "Engagement Score": engagement,
        "Support Issues": support_issues,
        "Plan Type": plan_type,
        "Churn": rng.binomial(1, churn_probability),
    })


@st.cache_resource(show_spinner="Loading the NIGGESH sample prototype…")
def get_demo_bundle():
    frame = get_prototype_frame()
    frame.attrs["source_name"] = "NIGGESH sample prototype"
    return train_models_from_frame(frame, "Churn", "1", suggest_excluded_columns(frame, "Churn"))


@st.cache_data(show_spinner=False)
def get_demo_scored():
    return score_customers(get_prototype_frame(), get_demo_bundle())


def apply_chart_theme(fig, *, show_legend=False):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=13), title=dict(font=dict(color=INK, size=17), x=.02),
        colorway=PALETTE, margin=dict(l=24, r=24, t=62, b=26),
        hoverlabel=dict(bgcolor=INK, font_color="white", bordercolor=VIOLET),
        showlegend=show_legend,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#D8DCE7", tickfont=dict(color="#343B4A", size=12), title_font=dict(color="#343B4A", size=13))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(32,36,58,.09)", zeroline=False, tickfont=dict(color="#343B4A", size=12), title_font=dict(color="#343B4A", size=13))
    if show_legend:
        fig.update_layout(margin=dict(l=24, r=24, t=62, b=84), legend=dict(orientation="h", yanchor="top", y=-.18, xanchor="left", x=0, title_text="", font=dict(color=INK, size=12)))
    return fig


def render_header(title: str, subtitle: str):
    st.markdown(f"<div class='workspace-header'><h1 style='font-size:2.3rem;margin-top:0;margin-bottom:.12rem'>{title}</h1><p style='color:{MUTED};font-size:1rem;margin-top:0;margin-bottom:.2rem'>{subtitle}</p></div>", unsafe_allow_html=True)


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    """Match a column name without relying on the model module's helper."""
    normalized_columns = {normalized(column): column for column in frame.columns}
    for alias in aliases:
        key = normalized(alias)
        if key in normalized_columns:
            return normalized_columns[key]
    for column in frame.columns:
        name = normalized(column)
        if any(normalized(alias) in name for alias in aliases):
            return column
    return None


def best_categorical_column(frame: pd.DataFrame, bundle: dict, aliases: list[str] | None = None, exclude: list[str] | None = None) -> str | None:
    excluded = set(exclude or []) | {bundle["target_column"]}
    valid = [column for column in bundle["categorical_columns"] if column in frame.columns and column not in excluded and 2 <= frame[column].nunique(dropna=True) <= 14]
    if aliases:
        for alias in aliases:
            direct = find_column(frame[valid], [alias]) if valid else None
            if direct:
                return direct
        # Some useful business groups arrive as numbers (for example a gym
        # contract duration of 1, 6, or 12 months). Treat low-cardinality
        # numeric fields as categories for churn-rate comparisons.
        numeric_groups = [column for column in bundle["numeric_columns"] if column in frame.columns and column not in excluded and 2 <= frame[column].nunique(dropna=True) <= 14]
        for alias in aliases:
            direct = find_column(frame[numeric_groups], [alias]) if numeric_groups else None
            if direct:
                return direct
    return valid[0] if valid else None


def best_numeric_column(frame: pd.DataFrame, bundle: dict, aliases: list[str] | None = None, exclude: list[str] | None = None) -> str | None:
    excluded = set(exclude or []) | {bundle["target_column"]}
    valid = [column for column in bundle["numeric_columns"] if column in frame.columns and column not in excluded and frame[column].nunique(dropna=True) > 1]
    if aliases:
        for alias in aliases:
            direct = find_column(frame[valid], [alias]) if valid else None
            if direct:
                return direct
    return valid[0] if valid else None


def categorical_churn_chart(frame: pd.DataFrame, column: str, title: str):
    data = frame[[column, "Observed Churn"]].copy()
    data[column] = data[column].fillna("Unknown").astype(str)
    grouped = data.groupby(column, dropna=False).agg(churn_rate=("Observed Churn", "mean"), customers=("Observed Churn", "size")).reset_index()
    grouped = grouped.sort_values("churn_rate", ascending=False).head(14)
    fig = px.bar(grouped, x=column, y="churn_rate", color=column, text=grouped["churn_rate"].map(lambda value: f"{value:.0%}"), title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=".0%", title="Observed churn rate")
    fig.update_xaxes(title=column, tickangle=-18)
    return apply_chart_theme(fig)


def numeric_churn_chart(frame: pd.DataFrame, column: str, title: str):
    data = frame[[column, "Observed Churn"]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[column])
    if data.empty or data[column].nunique() < 2:
        return None
    try:
        data["Range"] = pd.qcut(data[column], q=min(5, data[column].nunique()), duplicates="drop").astype(str)
    except ValueError:
        return None
    grouped = data.groupby("Range", observed=True).agg(churn_rate=("Observed Churn", "mean")).reset_index()
    fig = px.line(grouped, x="Range", y="churn_rate", markers=True, title=title, color_discrete_sequence=[CORAL])
    fig.update_yaxes(tickformat=".0%", title="Observed churn rate")
    fig.update_xaxes(title=column)
    return apply_chart_theme(fig)


def recorded_issue(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if not column:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).gt(0)
    text = values.astype("string").str.strip().str.lower().fillna("")
    benign = {"", "no", "none", "false", "0", "na", "n/a", "nan", "unknown", "not available", "resolved", "closed", "paid", "current", "ok"}
    return ~text.isin(benign)


def build_retention_plan(scored: pd.DataFrame) -> pd.DataFrame:
    """Suggest business-specific next steps from available customer signals."""
    plan = scored.copy()
    risk = plan["Risk Level"]
    high = risk.eq("High")
    moderate = risk.eq("Moderate")

    activity_col = find_column(plan, [
        "visits per month", "monthly visits", "weekly visits", "gym visits", "class attendance",
        "sessions attended", "sessions per month", "class bookings", "check ins per month", "weekly check in frequency",
        "check ins", "attendance rate", "attendance", "activity score", "engagement score", "usage count", "usage",
        "avg class frequency current month", "avg class frequency total", "class frequency",
    ])
    inactivity_col = find_column(plan, ["days since last visit", "days since last activity", "days inactive", "inactive days", "days absent"])
    tenure_col = find_column(plan, ["tenure months", "membership tenure", "member tenure", "membership length", "months active", "tenure", "lifetime"])
    value_col = find_column(plan, ["cltv", "customer lifetime value", "monthly charges", "monthly fee", "membership fee", "monthly revenue", "revenue", "value", "avg additional charges total", "additional charges total"])
    service_col = find_column(plan, ["complaint", "support ticket", "service issue", "service outage", "open issue", "support case", "problem reported"])
    payment_col = find_column(plan, ["late payment", "payment issue", "payment status", "failed payment", "overdue balance", "past due"])
    satisfaction_col = find_column(plan, ["satisfaction score", "customer satisfaction", "member satisfaction", "csat", "rating"])
    renewal_col = find_column(plan, ["days to renewal", "days until renewal", "renewal due in", "days to expiry", "days until expiry", "month to end contract", "months to contract end"])

    low_activity = pd.Series(False, index=plan.index)
    is_new = pd.Series(False, index=plan.index)
    high_value = pd.Series(False, index=plan.index)
    low_satisfaction = pd.Series(False, index=plan.index)
    if activity_col:
        activity = pd.to_numeric(plan[activity_col], errors="coerce")
        if activity.notna().any():
            low_activity = activity.le(activity.quantile(.25))
    if inactivity_col:
        inactivity = pd.to_numeric(plan[inactivity_col], errors="coerce")
        if inactivity.notna().any():
            low_activity = inactivity.ge(inactivity.quantile(.75))
    if tenure_col:
        tenure = pd.to_numeric(plan[tenure_col], errors="coerce")
        if tenure.notna().any():
            is_new = tenure.le(min(3, tenure.quantile(.25)))
    if value_col:
        value = pd.to_numeric(plan[value_col], errors="coerce")
        if value.notna().any():
            high_value = value.ge(value.quantile(.75))
    if satisfaction_col:
        satisfaction = pd.to_numeric(plan[satisfaction_col], errors="coerce")
        if satisfaction.notna().any():
            low_satisfaction = satisfaction.le(satisfaction.quantile(.25))
        else:
            satisfaction_text = plan[satisfaction_col].astype("string").str.lower()
            low_satisfaction = satisfaction_text.str.contains("poor|low|bad|unhappy|dissatisf|1 star|2 star", na=False)
    renewal_due = pd.Series(False, index=plan.index)
    if renewal_col:
        days_to_renewal = pd.to_numeric(plan[renewal_col], errors="coerce")
        if days_to_renewal.notna().any():
            is_month_measure = "month" in normalized(renewal_col)
            renewal_due = days_to_renewal.between(0, 1 if is_month_measure else 30)

    service_issue = recorded_issue(plan, service_col) | low_satisfaction
    payment_issue = recorded_issue(plan, payment_col)
    activity_signal_col = activity_col or inactivity_col
    gym_activity = bool(activity_signal_col and any(token in normalized(activity_signal_col) for token in ["visit", "attendance", "class", "session", "gym", "check in"]))

    action_conditions = [
        service_issue,
        payment_issue,
        renewal_due,
        high & low_activity,
        high & is_new,
        high & high_value,
        high,
        moderate & low_activity,
        moderate,
        high_value,
    ]
    action_choices = [
        "Resolve the service issue, then follow up",
        "Remove payment friction",
        "Contact them before renewal",
        "Personal attendance check-in" if gym_activity else "Personal engagement check-in",
        "Early-tenure onboarding check-in",
        "Personal save conversation for a high-value customer",
        "Personal retention conversation",
        "Rebuild activity with a personal check-in" if gym_activity else "Rebuild engagement with a personal check-in",
        "Proactive needs check-in",
        "Recognize and retain a high-value customer",
    ]
    plan["Recommended action"] = np.select(action_conditions, action_choices, default="Maintain engagement and recognize loyalty")

    why_conditions = [service_issue, payment_issue, renewal_due, high & low_activity, high & is_new, high & high_value, high, moderate & low_activity, moderate, high_value]
    why_choices = [
        "A service concern or low satisfaction appears in the available data.",
        "A payment or billing issue appears in the available data.",
        "Renewal is due within the next 30 days.",
        "High predicted churn risk and activity is in the lowest quarter.",
        "High predicted churn risk during the earliest tenure group.",
        "High predicted churn risk and customer value is in the top quarter.",
        "The model assigned this customer a high churn risk.",
        "Activity is in the lowest quarter and churn risk is moderate.",
        "The model assigned this customer a moderate churn risk.",
        "Customer value is in the top quarter; prioritize preserving the relationship.",
    ]
    plan["Why this action"] = np.select(why_conditions, why_choices, default="Low predicted risk; maintain a positive customer experience.")

    offer_conditions = [service_issue, payment_issue, renewal_due, high & low_activity, high & is_new, high & high_value, high, moderate, high_value]
    offer_choices = [
        "Fix the underlying issue first; consider a service credit only if the resolution was delayed.",
        "Offer a payment-date adjustment or temporary pause if available; avoid an automatic discount.",
        "Discuss a flexible renewal term or useful add-on before offering a price reduction.",
        "Invite them to a suitable class or a complimentary fitness check-in." if gym_activity else "Offer a guided setup or use-case session before considering a perk.",
        "Offer a welcome session or onboarding support; use a small perk only if it fits their needs.",
        "Consider a tailored renewal benefit or added-value bundle after learning their concern.",
        "Ask what would make them stay, then offer one relevant plan or service option.",
        "Offer a relevant class, feature, or service reminder; hold discounts until needed.",
        "Use recognition, early access, or a referral benefit instead of a blanket discount.",
    ]
    plan["Targeted offer idea"] = np.select(offer_conditions, offer_choices, default="Use a low-cost loyalty benefit or personal thank-you; no discount needed by default.")
    plan["Action priority"] = np.select(
        [high | service_issue | payment_issue, moderate | low_activity | renewal_due, high_value],
        ["Priority", "Follow-up", "Relationship"],
        default="Routine",
    )
    return plan


# Sidebar: select the sample data or load a business's own file.
with st.sidebar:
    logo_path = Path(__file__).parent / "assets" / "niggesh-logo.svg"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown(f"<h1 style='color:{VIOLET}'>NIGGESH</h1>", unsafe_allow_html=True)
    st.markdown("### Your data")
    uploaded_files = st.file_uploader("Upload up to 2 CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True, help="Upload Telco and Gym together, then switch between their dashboards. Files stay in this local app session.")

custom_bundle = None
custom_scored = None
custom_name = None
custom_ready = False
active_dataset = "prototype"
upload_items = {}
for file in uploaded_files:
    file_bytes = file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    upload_items[file_hash] = (file, file_bytes)

if len(upload_items) > 2:
    st.sidebar.error("Please upload no more than two datasets at once.")
    st.stop()

upload_hashes = list(upload_items)
upload_signature = "|".join(upload_hashes)
previous_upload_signature = st.session_state.get("_upload_signature")
if previous_upload_signature != upload_signature:
    previous_active = st.session_state.get("_active_dataset_hash", "prototype")
    st.session_state["_upload_signature"] = upload_signature
    if not upload_hashes:
        st.session_state["_active_dataset_hash"] = "prototype"
    elif previous_active not in upload_hashes:
        st.session_state["_active_dataset_hash"] = upload_hashes[0]

dataset_labels = {"prototype": "Prototype · Sample data"}
for file_hash, (file, _) in upload_items.items():
    dataset_labels[file_hash] = file.name
active_dataset = st.sidebar.selectbox(
    "View dashboard for",
    options=["prototype", *upload_hashes],
    format_func=lambda key: dataset_labels[key],
    key="_active_dataset_hash",
)

if active_dataset != "prototype":
    uploaded, upload_bytes = upload_items[active_dataset]
    try:
        upload_frame = read_upload(upload_bytes, uploaded.name)
    except Exception as exc:
        st.sidebar.error(f"Could not read that file: {exc}")
        st.stop()
    upload_hash = hashlib.sha256(upload_bytes).hexdigest()[:12]
    st.sidebar.caption(f"Loaded: {uploaded.name} · {len(upload_frame):,} rows · {len(upload_frame.columns)} columns")
    preferred_targets = ["churn value", "churn", "churned", "churn label", "attrition", "membership status", "left"]
    target_guess = next(
        (column for preferred in preferred_targets for column in upload_frame.columns if normalized(column) == preferred),
        next((column for column in upload_frame.columns if any(word in normalized(column) for word in ["churn", "cancel", "attrition", "renewal", "membership status", "left"])), upload_frame.columns[-1]),
    )
    target_key = f"target_{upload_hash}_v2"
    target_options = list(upload_frame.columns)
    default_target_index = target_options.index(target_guess)
    target_column = st.sidebar.selectbox("Outcome column", target_options, index=default_target_index, key=target_key, help="Choose the column that records whether each customer left or stayed.")
    outcome_values = upload_frame[target_column].dropna().unique().tolist()
    outcome_labels = [str(value) for value in outcome_values]
    if len(outcome_labels) != 2:
        st.sidebar.error(f"Choose a binary outcome column. This column has {len(outcome_labels)} distinct values; the model needs exactly two, such as Left/Stayed or 1/0.")
        st.stop()
    positive_guess = guess_positive_label(outcome_values)
    positive_key = f"positive_{upload_hash}_{normalized(target_column).replace(' ', '_')}"
    positive_label = st.sidebar.selectbox("Which value means churn / leaving?", outcome_labels, index=outcome_labels.index(positive_guess), key=positive_key)
    suggested_exclusions = suggest_excluded_columns(upload_frame, target_column)
    exclude_key = f"exclude_{upload_hash}_{normalized(target_column).replace(' ', '_')}_v2"
    excluded = st.sidebar.multiselect("Exclude ID / post-churn columns", [column for column in upload_frame.columns if column != target_column], default=suggested_exclusions, key=exclude_key, help="IDs and information only known after cancellation should not be model inputs.")
    config_key = f"{upload_hash}|{target_column}|{positive_label}|{'|'.join(sorted(map(str, excluded)))}"
    trained_uploads = st.session_state.setdefault("trained_uploads", {})
    if st.sidebar.button("Train & update dashboard", type="primary", use_container_width=True):
        try:
            upload_frame.attrs["source_name"] = uploaded.name
            with st.spinner("Training the Decision Tree and Neural Network on your business data…"):
                custom_bundle = train_models_from_frame(upload_frame, target_column, positive_label, excluded)
                custom_scored = score_customers(upload_frame, custom_bundle)
            trained_uploads[config_key] = {
                "bundle": custom_bundle,
                "scored": custom_scored,
                "name": uploaded.name,
            }
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Training could not finish: {exc}")
    if config_key in trained_uploads:
        custom_bundle = trained_uploads[config_key]["bundle"]
        custom_scored = trained_uploads[config_key]["scored"]
        custom_name = uploaded.name
        custom_ready = custom_bundle is not None and custom_scored is not None
    if not custom_ready:
        st.sidebar.caption("Choose the outcome and click **Train & update dashboard**. Your uploaded data stays in this session.")
        st.title("Finish setting up your uploaded dataset")
        st.write(f"NIGGESH read **{len(upload_frame):,} customer rows** from **{uploaded.name}**. Choose the outcome column and the value that means churn in the sidebar, then click **Train & update dashboard**. Uploading alone does not retrain the models.")
        st.info("For training, each row needs a known outcome: churned/left or stayed. Features should describe the customer before that outcome happened.")
        st.dataframe(upload_frame.head(5), use_container_width=True, hide_index=True)
        st.stop()
    bundle = custom_bundle
    df = custom_scored
    source_label = custom_name
    source_kind = "Uploaded dataset"
else:
    try:
        bundle = get_demo_bundle()
        df = get_demo_scored()
        source_label = bundle.get("dataset_name", "NIGGESH sample prototype")
        source_kind = "Prototype dataset"
    except Exception as exc:
        st.error(f"Could not load the sample prototype: {exc}")
        st.stop()

with st.sidebar:
    st.caption(f"{source_kind} · {source_label} · {len(df):,} customers")
    if active_dataset == "prototype" and st.button("Retrain models", use_container_width=True):
        with st.spinner("Retraining the sample models…"):
            get_demo_bundle.clear()
            get_demo_scored.clear()
            get_demo_bundle()
        st.rerun()
    page_icons = {
        "Executive Dashboard": "📊",
        "Customer Risk Analysis": "🎯",
        "Customer Segmentation": "🧩",
        "Churn Drivers": "🔎",
        "Retention Recommendations": "💡",
        "Model Performance": "⚙️",
    }
    page = st.radio("NAVIGATION", [
        "Executive Dashboard", "Customer Risk Analysis", "Customer Segmentation",
        "Churn Drivers", "Retention Recommendations", "Model Performance",
    ], format_func=lambda name: f"{page_icons[name]}   {name}")

# Useful columns vary by business; find familiar equivalents when possible.
category_col = best_categorical_column(df, bundle, ["contract", "membership type", "plan type", "subscription type", "plan"])
service_col = best_categorical_column(df, bundle, ["internet service", "service type", "service", "location type", "group visits", "partner"] , exclude=[category_col] if category_col else [])
tenure_col = best_numeric_column(df, bundle, ["tenure months", "membership tenure", "member tenure", "membership length", "months active", "tenure", "lifetime"])
value_col = best_numeric_column(df, bundle, ["cltv", "customer lifetime value", "monthly charges", "monthly fee", "membership fee", "monthly revenue", "revenue", "spend", "price", "avg additional charges total", "additional charges total"])
customer_id_col = find_column(df, ["customer id", "member id", "account id", "client id", "subscriber id", "customer number", "member number"])
if customer_id_col is None:
    df["Customer"] = [f"Record {index + 1}" for index in range(len(df))]
    customer_id_col = "Customer"


def observed_churn_rate(frame: pd.DataFrame) -> float:
    return float(frame["Observed Churn"].mean()) if len(frame) else 0.0


if page == "Executive Dashboard":
    render_header("Executive Dashboard", "Customer churn risk and retention opportunities from the selected dataset")
    st.markdown(f"<div class='dataset-note'>{source_kind}: {source_label}. Predicted risk uses the Decision Tree; observed churn uses the selected outcome column.</div>", unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total customers",f"{len(df):,}")
    c2.metric("Observed churn rate",f"{observed_churn_rate(df):.1%}")
    c3.metric("High-risk customers",f"{(df['Risk Level']=='High').sum():,}")
    if df["Estimated Value at Risk"].notna().any():
        c4.metric("Estimated value at risk",f"${df['Estimated Value at Risk'].sum():,.0f}")
    else:
        c4.metric("Estimated value at risk","N/A",help="Upload a numeric customer value, fee, or revenue field to estimate value at risk.")

    left,right=st.columns(2)
    with left:
        if category_col:
            st.plotly_chart(categorical_churn_chart(df,category_col,f"Observed churn by {category_col}"),use_container_width=True)
        elif tenure_col:
            chart=numeric_churn_chart(df,tenure_col,f"Observed churn by {tenure_col}")
            if chart: st.plotly_chart(chart,use_container_width=True)
        else:
            st.info("Add a useful categorical field such as plan or membership type to see churn by group.")
    with right:
        numeric_for_trend=tenure_col or value_col
        if numeric_for_trend:
            chart=numeric_churn_chart(df,numeric_for_trend,f"Observed churn by {numeric_for_trend}")
            if chart: st.plotly_chart(chart,use_container_width=True)
        else:
            st.info("Add a numeric tenure or customer value field to compare churn across ranges.")
    left,right=st.columns(2)
    with left:
        if service_col and service_col != category_col:
            st.plotly_chart(categorical_churn_chart(df,service_col,f"Observed churn by {service_col}"),use_container_width=True)
        elif value_col and value_col != numeric_for_trend:
            chart=numeric_churn_chart(df,value_col,f"Observed churn by {value_col}")
            if chart: st.plotly_chart(chart,use_container_width=True)
        else:
            st.markdown("<div class='nig-card'><b>Outcome definition</b><br>Churn is the positive value selected in the data setup panel.</div>",unsafe_allow_html=True)
    with right:
        fig=px.histogram(df,x="Risk Score",nbins=20,color_discrete_sequence=[VIOLET],title="Predicted risk score distribution")
        fig.update_layout(xaxis_title="Predicted risk (0–100)",yaxis_title="Customers")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)

elif page == "Customer Risk Analysis":
    render_header("Customer Risk Analysis", "Filter and prioritize customers using their predicted churn risk")
    first,second,third=st.columns([1.3,1,1])
    search=first.text_input("Search customer / member",placeholder="Type an ID or name")
    threshold=second.slider("Minimum risk score",0,100,60)
    risk_filter=third.multiselect("Risk level",["High","Moderate","Low"],default=["High","Moderate","Low"])
    filtered=df[(df["Risk Score"]>=threshold)&df["Risk Level"].isin(risk_filter)].copy()
    if search:
        filtered=filtered[filtered[customer_id_col].astype(str).str.contains(search,case=False,na=False)]
    filtered=filtered.sort_values("Risk Score",ascending=False)
    info_cols=[column for column in bundle["feature_columns"] if column in filtered.columns][:3]
    display_cols=list(dict.fromkeys([customer_id_col,*info_cols,"Risk Score","Risk Level","Churn Probability","Segment"]))
    view=filtered[display_cols].copy()
    view["Churn Probability"]=view["Churn Probability"].map(lambda value:f"{value:.1%}")
    view["Risk Score"]=view["Risk Score"].map(lambda value:f"{value:.1f}")
    st.caption(f"{len(view):,} matching customers, sorted by predicted risk.")
    st.dataframe(view.head(200),use_container_width=True,hide_index=True)

elif page == "Customer Segmentation":
    render_header("Customer Segmentation", "Groups formed from churn risk and the customer fields available in this dataset")
    segment_counts=df["Segment"].value_counts()
    cols=st.columns(min(4,max(1,len(segment_counts))))
    colors={"At-Risk":CORAL,"Premium":VIOLET,"Loyal":TEAL,"New":YELLOW,"Standard":"#8892A6","Watchlist":YELLOW,"Stable":TEAL}
    icons={"At-Risk":"🔥","Premium":"💎","Loyal":"🌿","New":"✨","Standard":"👥","Watchlist":"👀","Stable":"✅"}
    for col,(segment,count) in zip(cols,segment_counts.items()):
        color=colors.get(segment,VIOLET)
        col.markdown(f"<div class='nig-card' style='text-align:center;border-top:4px solid {color}'><div style='font-size:1.6rem'>{icons.get(segment,'•')}</div><div style='color:{MUTED};font-weight:700'>{segment}</div><div style='font:900 1.9rem Nunito;color:{color}'>{count:,}</div></div>",unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        seg_counts=segment_counts.rename_axis("Segment").reset_index(name="Customers")
        fig=px.pie(seg_counts,values="Customers",names="Segment",hole=.58,title="Customer mix",color="Segment",color_discrete_map=colors)
        fig.update_traces(textposition="inside",textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
    with right:
        x_col=tenure_col or value_col
        y_col=value_col if x_col != value_col else next((column for column in bundle["numeric_columns"] if column != x_col and column in df.columns),None)
        if x_col and y_col:
            fig=px.scatter(df,x=x_col,y=y_col,color="Segment",title=f"{x_col} and {y_col} by segment",opacity=.72,color_discrete_map=colors)
            st.plotly_chart(apply_chart_theme(fig,show_legend=True),use_container_width=True)
        else:
            st.info("Add two numeric fields to explore customer value and tenure by segment.")
    stats=df.groupby("Segment",observed=True).agg(Customers=(customer_id_col,"count"),Observed_Churn=("Observed Churn","mean"),Average_Risk=("Risk Score","mean")).reset_index()
    stats=stats.rename(columns={"Observed_Churn":"Observed churn rate","Average_Risk":"Average risk score"})
    stats["Observed churn rate"]=stats["Observed churn rate"].map(lambda value:f"{value:.1%}")
    stats["Average risk score"]=stats["Average risk score"].map(lambda value:f"{value:.1f}")
    st.dataframe(stats,use_container_width=True,hide_index=True)

elif page == "Churn Drivers":
    render_header("Churn Drivers", "Customer attributes the Decision Tree relied on most")
    st.info("Feature importance shows which fields influenced the model across the dataset. It does not prove that a field causes churn.")
    drivers=bundle.get("feature_importance",[])[:15]
    if drivers:
        impact=pd.DataFrame(drivers).sort_values("importance")
        fig=px.bar(impact,x="importance",y="feature",orientation="h",title="Top model drivers",color="importance",color_continuous_scale=[YELLOW,VIOLET])
        fig.update_layout(coloraxis_showscale=False,yaxis_title="",xaxis_title="Relative importance")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
    else:
        st.info("The fitted Decision Tree did not produce feature importance values for this file.")
    left,right=st.columns(2)
    with left:
        if category_col:
            st.plotly_chart(categorical_churn_chart(df,category_col,f"Observed churn by {category_col}"),use_container_width=True)
    with right:
        if tenure_col:
            chart=numeric_churn_chart(df,tenure_col,f"Observed churn by {tenure_col}")
            if chart: st.plotly_chart(chart,use_container_width=True)
        elif value_col:
            chart=numeric_churn_chart(df,value_col,f"Observed churn by {value_col}")
            if chart: st.plotly_chart(chart,use_container_width=True)

elif page == "Retention Recommendations":
    render_header("Retention Recommendations", "Practical next steps tailored to the fields in your customer data")
    st.caption("Rule-based suggestions adapt to the fields available in your data. Resolve service or payment friction first; use targeted offers only when they fit.")
    recommendations=build_retention_plan(df)
    summary=(recommendations.groupby(["Action priority","Recommended action"], dropna=False)
             .agg(Customers=(customer_id_col,"count"),Average_Risk=("Risk Score","mean"),Offer=("Targeted offer idea","first"))
             .reset_index())
    priority_order={"Priority":0,"Follow-up":1,"Relationship":2,"Routine":3}
    summary["_order"]=summary["Action priority"].map(priority_order)
    summary=summary.sort_values(["_order","Average_Risk"],ascending=[True,False]).drop(columns="_order")
    st.subheader("Recommended plays")
    for _,row in summary.iterrows():
        with st.container(border=True):
            a,b=st.columns([3,1])
            a.markdown(f"**{row['Action priority']} · {row['Recommended action']}**")
            a.caption(row["Offer"])
            b.metric("Customers",f"{int(row['Customers']):,}")
    st.subheader("Customers to contact first")
    high=recommendations.assign(_order=recommendations["Action priority"].map(priority_order)).sort_values(["_order","Risk Score"],ascending=[True,False]).head(50)
    cols=[customer_id_col,*([value_col] if value_col and value_col != customer_id_col else []),"Action priority","Risk Level","Risk Score","Churn Probability","Recommended action","Why this action","Targeted offer idea"]
    high_view=high[list(dict.fromkeys(column for column in cols if column in high.columns))].copy()
    high_view["Churn Probability"]=high_view["Churn Probability"].map(lambda value:f"{value:.1%}")
    st.dataframe(high_view,use_container_width=True,hide_index=True)

elif page == "Model Performance":
    render_header("Model Performance", "Compare both models on customers held out from training")
    st.markdown(f"<div class='nig-card'>Dataset: <b>{source_label}</b> · <b>{bundle['dataset_rows']:,}</b> customers. Models were trained on 80% and evaluated on the remaining <b>{bundle['test_size']:,}</b>; those rows were held out during fitting.</div>",unsafe_allow_html=True)
    metrics=bundle["metrics"]
    names=list(metrics.keys())
    selected=st.selectbox("Choose a model",names,key=f"model_select_{hashlib.sha1(source_label.encode()).hexdigest()[:8]}")
    st.caption("The metric cards and confusion matrix both follow this selection.")
    st.subheader(selected)
    cards=st.columns(4)
    for card,label,key in zip(cards,["Accuracy","Precision","Recall","F1 score"],["accuracy","precision","recall","f1"]):
        card.metric(label,f"{metrics[selected][key]:.1%}")
    matrix=np.array(metrics[selected]["confusion_matrix"])
    fig=px.imshow(matrix,text_auto=True,color_continuous_scale=[[0,"#F4F3FF"],[1,VIOLET]],labels=dict(x="Predicted",y="Actual",color="Customers"),x=["Stayed / retained","Left / churned"],y=["Stayed / retained","Left / churned"],title=f"{selected}: held-out predictions")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
    st.caption("Accuracy = overall correct predictions. Precision = how many predicted leavers did leave. Recall = how many actual leavers the model caught. F1 balances precision and recall.")
    if bundle.get("mlp_warning"):
        st.warning("The neural network reached its training iteration limit. Its displayed metrics are still measured on the held-out data.")
