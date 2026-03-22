# app.py
# Safety & Sustainability Intelligence for Permit-to-Work (PTW)
# Streamlit dashboard + Neo4j graph persistence + What-If simulation
#
# Install:
#   python -m pip install streamlit pandas plotly openpyxl neo4j
#
# Run:
#   streamlit run app.py
#
# Notes:
# - Works without Neo4j (graph persistence is optional).
# - CSV encoding: tries UTF-8 then CP1252.
# - Uses a canonical PTW "contract" of columns but will safely default missing ones.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional Neo4j integration
NEO4J_AVAILABLE = True
try:
    from neo4j_store import Neo4jStore
except Exception:
    NEO4J_AVAILABLE = False

# -----------------------------
# Defaults / Factors
# -----------------------------
DEFAULT_DIESEL_KG_PER_L = 2.68
DEFAULT_ELEC_KG_PER_KWH = 0.42

DEFAULT_PROCESS_FACTORS = {
    "steam_reforming": 1800.0,
    "ethylene_cracking": 1400.0,
    "cement_kiln": 900.0,
}

DEFAULT_CARBON_BUDGET_KG = 500.0
DEFAULT_HIGH_CARBON_ALERT_KG = 300.0

DEFAULT_HIGH_RISK_THRESHOLD = 70
DEFAULT_MED_RISK_THRESHOLD = 40

SCENARIOS = {
    "All Permits": "all",
    "Live Now": "live",
    "High Risk": "high_risk",
    "Level II Isolations": "level2",
    "Over Budget (Carbon)": "over_budget",
    "High Priority (Live + High + LevelII/OverBudget)": "high_priority",
    "Conflicts / Clashes": "clashes",
}

DISPLAY_COLUMNS = [
    "permit_no", "permit_type", "status",
    "risk_band", "risk_score",
    "level2_detected", "over_budget", "has_clash",
    "ptw_kg_co2e", "CIS",
    "combustion_kg", "electricity_kg", "process_kg",
    "site", "main_area", "sub_area", "location",
    "primary_contractor_company", "tra_number", "isolation_nos",
    "start_datetime", "end_datetime",
]

CANONICAL_DEFAULTS = [
    ("permit_no", ""),
    ("permit_type", ""),
    ("status", "Requested"),
    ("reference_no", ""),
    ("work_category", ""),
    ("permit_title", ""),
    ("created_by", ""),
    ("contact_number", ""),
    ("work_to_be_done", ""),
    ("accepted_by", ""),
    ("signature_present", False),
    ("start_datetime", None),
    ("end_datetime", None),
    ("site", ""),
    ("main_area", ""),
    ("sub_area", ""),
    ("location", ""),
    ("primary_contractor_company", ""),
    ("tra_number", ""),
    ("isolation_nos", ""),
    ("clash_permit_no", ""),
    ("fuel_qty", 0.0),
    ("electricity_kwh", 0.0),
    ("process_activity_qty", 0.0),
    ("process_type", ""),
]

def ensure_col(df: pd.DataFrame, name: str, default=None) -> None:
    if name not in df.columns:
        df[name] = default

def safe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}

def norm_status(x) -> str:
    s = (str(x) if x is not None else "").strip()
    if not s:
        return "Requested"
    s_title = s.title()
    mapping = {
        "In Progress": "Live",
        "Open": "Live",
        "Live": "Live",
        "Closed": "Closed",
        "On Hold": "On Hold",
        "Requested": "Requested",
        "Cancelled": "Cancelled",
    }
    return mapping.get(s_title, s_title)

def parse_dt(v):
    return pd.to_datetime(v, errors="coerce")

def read_upload(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="cp1252")
    return pd.read_excel(uploaded)

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        ensure_col(df, c, 0)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

def split_csv_list(v: str) -> List[str]:
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def compute_cis(df: pd.DataFrame, col="ptw_kg_co2e") -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    mn = float(df[col].min())
    mx = float(df[col].max())
    return (df[col] - mn) / (mx - mn + 1e-6)

@dataclass
class Thresholds:
    diesel_kg_per_l: float = DEFAULT_DIESEL_KG_PER_L
    electricity_kg_per_kwh: float = DEFAULT_ELEC_KG_PER_KWH
    carbon_budget_kg: float = DEFAULT_CARBON_BUDGET_KG
    high_carbon_alert_kg: float = DEFAULT_HIGH_CARBON_ALERT_KG
    high_risk_threshold: int = DEFAULT_HIGH_RISK_THRESHOLD
    med_risk_threshold: int = DEFAULT_MED_RISK_THRESHOLD

def detect_level2(df: pd.DataFrame) -> pd.Series:
    if "level2_detected" in df.columns:
        return df["level2_detected"].apply(safe_bool)
    for c, d in [("work_category", ""), ("work_to_be_done", ""), ("isolation_nos", "")]:
        ensure_col(df, c, d)
    pats = ("lvl ii", "level ii", "l2", "level2", "level 2")

    def infer(row) -> bool:
        wc = str(row.get("work_category") or "").lower()
        wtd = str(row.get("work_to_be_done") or "").lower()
        iso = str(row.get("isolation_nos") or "").lower()
        if any(p in wc for p in pats) or any(p in wtd for p in pats) or any(p in iso for p in pats):
            return True
        nums = pd.Series(split_csv_list(iso))
        nums = pd.to_numeric(nums.str.extract(r"(\\d+)")[0], errors="coerce")
        return bool(len(nums.dropna())) and float(nums.max()) >= 1110

    return df.apply(infer, axis=1)

def compute_carbon(df: pd.DataFrame, t: Thresholds, process_factors: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    coerce_numeric(df, ["fuel_qty", "electricity_kwh", "process_activity_qty"])
    ensure_col(df, "process_type", "")
    df["process_type"] = df["process_type"].fillna("").astype(str)

    df["combustion_kg"] = df["fuel_qty"] * float(t.diesel_kg_per_l)
    df["electricity_kg"] = df["electricity_kwh"] * float(t.electricity_kg_per_kwh)

    def proc_kg(row):
        ptype = str(row.get("process_type") or "").strip()
        qty = float(row.get("process_activity_qty") or 0.0)
        return qty * float(process_factors.get(ptype, 0.0))

    df["process_kg"] = df.apply(proc_kg, axis=1)

    if "ptw_kg_co2e" not in df.columns:
        df["ptw_kg_co2e"] = df["combustion_kg"] + df["electricity_kg"] + df["process_kg"]
    else:
        df["ptw_kg_co2e"] = pd.to_numeric(df["ptw_kg_co2e"], errors="coerce").fillna(
            df["combustion_kg"] + df["electricity_kg"] + df["process_kg"]
        )

    if "over_budget" in df.columns:
        df["over_budget"] = df["over_budget"].apply(safe_bool)
    else:
        df["over_budget"] = df["ptw_kg_co2e"] > float(t.carbon_budget_kg)

    if "CIS" in df.columns:
        df["CIS"] = pd.to_numeric(df["CIS"], errors="coerce").fillna(compute_cis(df))
    else:
        df["CIS"] = compute_cis(df)

    df["high_carbon"] = df["ptw_kg_co2e"] >= float(t.high_carbon_alert_kg)
    return df

def compute_risk(df: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    df = df.copy()
    if "risk_score" in df.columns and "risk_band" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).astype(int)
        df["risk_band"] = df["risk_band"].fillna("Low").astype(str)
        return df

    ensure_col(df, "permit_type", "")
    ensure_col(df, "work_to_be_done", "")
    ensure_col(df, "status", "Requested")

    base = {"hot work": 30, "confined space": 35, "electrical": 28, "maintenance": 15, "coldwork": 15, "general": 10}
    keywords = {
        "rope access": 10, "bunker": 10, "hv": 10, "energized": 10, "welding": 10, "grinding": 8,
        "vessel": 10, "confined": 10, "excavation": 8, "lifting": 8,
    }

    def score(row) -> int:
        pt = str(row.get("permit_type") or "").lower()
        wtd = str(row.get("work_to_be_done") or "").lower()
        stt = norm_status(row.get("status"))

        s = int(base.get(pt, base["general"]))
        if stt == "Live":
            s += 10
        if safe_bool(row.get("level2_detected")):
            s += 15
        if safe_bool(row.get("over_budget")):
            s += 8
        if safe_bool(row.get("high_carbon")):
            s += 5
        for k, w in keywords.items():
            if k in wtd:
                s += int(w)
        return max(0, min(100, int(s)))

    df["risk_score"] = df.apply(score, axis=1)
    df["risk_band"] = df["risk_score"].apply(
        lambda x: "High" if x >= int(t.high_risk_threshold)
        else ("Medium" if x >= int(t.med_risk_threshold) else "Low")
    )
    return df

def _dedupe_pairs(pairs: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for p in pairs:
        a, b = p.get("a"), p.get("b")
        if not a or not b or a == b:
            continue
        key = (a, b, p.get("reason", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def infer_clashes(df: pd.DataFrame, overlap_hours: int = 6) -> List[dict]:
    pairs: List[dict] = []
    if "clash_permit_no" in df.columns:
        for _, r in df.iterrows():
            a = str(r.get("permit_no") or "").strip()
            raw = str(r.get("clash_permit_no") or "").strip()
            if not a or not raw:
                continue
            for b in split_csv_list(raw):
                if b and b != a:
                    aa, bb = sorted([a, b])
                    pairs.append({"a": aa, "b": bb, "reason": "declared"})

    m = df.copy()
    m["start_dt"] = parse_dt(m.get("start_datetime"))

    loc_key = None
    for k in ["location", "sub_area", "main_area"]:
        if k in m.columns:
            loc_key = k
            break
    if loc_key is None:
        return _dedupe_pairs(pairs)

    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            a = m.iloc[i]
            b = m.iloc[j]
            la = str(a.get(loc_key) or "").strip()
            lb = str(b.get(loc_key) or "").strip()
            if not la or not lb or la != lb:
                continue
            sa = a.get("start_dt")
            sb = b.get("start_dt")
            if pd.isna(sa) or pd.isna(sb):
                continue
            if abs((sa - sb).total_seconds()) <= overlap_hours * 3600:
                aa, bb = sorted([str(a.get("permit_no")), str(b.get("permit_no"))])
                pairs.append({"a": aa, "b": bb, "reason": f"time_overlap_{overlap_hours}h"})
    return _dedupe_pairs(pairs)

def add_has_clash_flag(df: pd.DataFrame, clash_pairs: List[dict]) -> pd.DataFrame:
    df = df.copy()
    s = set()
    for p in clash_pairs:
        s.add(p["a"]); s.add(p["b"])
    df["has_clash"] = df["permit_no"].astype(str).isin(s)
    return df

def apply_scenario(df: pd.DataFrame, scenario_key: str) -> pd.DataFrame:
    d = df.copy()
    if scenario_key == "all":
        return d
    if scenario_key == "live":
        return d[d["status"].eq("Live")]
    if scenario_key == "high_risk":
        return d[d["risk_band"].eq("High")]
    if scenario_key == "level2":
        return d[d["level2_detected"].eq(True)]
    if scenario_key == "over_budget":
        return d[d["over_budget"].eq(True)]
    if scenario_key == "high_priority":
        return d[(d["status"].eq("Live")) & (d["risk_band"].eq("High")) & ((d["level2_detected"]) | (d["over_budget"]))]
    if scenario_key == "clashes":
        return d[d["has_clash"].eq(True)]
    return d

def build_clash_network(df: pd.DataFrame, clash_pairs: List[dict], max_nodes: int = 80):
    if not clash_pairs:
        return None
    permits = sorted(set([p["a"] for p in clash_pairs] + [p["b"] for p in clash_pairs]))
    if not permits:
        return None
    permits = permits[:max_nodes]

    import math
    n = len(permits)
    pos = {}
    for i, pid in enumerate(permits):
        angle = 2 * math.pi * i / max(n, 1)
        pos[pid] = (0.9 * math.cos(angle), 0.9 * math.sin(angle))

    df_idx = df.set_index(df["permit_no"].astype(str))
    node_x, node_y, texts, sizes, colors = [], [], [], [], []
    for pid in permits:
        x, y = pos[pid]
        node_x.append(x); node_y.append(y)
        r = df_idx.loc[pid] if pid in df_idx.index else None
        risk_band = str(r.get("risk_band")) if r is not None else "Low"
        status = str(r.get("status")) if r is not None else "Requested"
        over_budget = bool(r.get("over_budget")) if r is not None else False
        level2 = bool(r.get("level2_detected")) if r is not None else False
        texts.append(f"{pid}<br>Status: {status}<br>Risk: {risk_band}<br>OverBudget: {over_budget}<br>LevelII: {level2}")
        sizes.append(18 if risk_band == "High" else (14 if risk_band == "Medium" else 10))
        colors.append("red" if risk_band == "High" else ("orange" if risk_band == "Medium" else "green"))

    edge_x, edge_y = [], []
    for p in clash_pairs:
        a, b = p["a"], p["b"]
        if a not in pos or b not in pos:
            continue
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1)),
        text=texts, hoverinfo="text"
    ))
    fig.update_layout(
        title="Clash Relationship Network (permits with overlaps/declared clashes)",
        showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False), height=520
    )
    return fig

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Safety & Sustainability Intelligence (PTW)", layout="wide")
st.title("Safety & Sustainability Intelligence for Permit-to-Work")

st.sidebar.header("Data Import")
uploaded = st.sidebar.file_uploader("Upload PTW CSV/XLSX", type=["csv", "xlsx"])
st.sidebar.caption("If you don't have data ready, download the sample CSV below and upload it.")

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "ptw_sample_all_scenarios.csv")
if uploaded is None:
    try:
        with open(SAMPLE_PATH, "rb") as f:
            st.sidebar.download_button("Download sample PTW CSV", f, file_name="ptw_sample_all_scenarios.csv", mime="text/csv")
    except Exception:
        pass

st.sidebar.header("Adjustable Thresholds")
with st.sidebar.expander("Carbon / Energy factors", expanded=False):
    diesel_factor = st.slider("Diesel factor (kg CO₂e per litre)", 0.5, 6.0, float(DEFAULT_DIESEL_KG_PER_L), 0.01)
    elec_factor = st.slider("Electricity factor (kg CO₂e per kWh)", 0.05, 1.5, float(DEFAULT_ELEC_KG_PER_KWH), 0.01)
    carbon_budget = st.slider("Per-permit carbon budget (kg CO₂e)", 50.0, 5000.0, float(DEFAULT_CARBON_BUDGET_KG), 10.0)
    high_carbon_alert = st.slider("High-carbon alert threshold (kg CO₂e)", 50.0, 5000.0, float(DEFAULT_HIGH_CARBON_ALERT_KG), 10.0)

with st.sidebar.expander("Risk thresholds", expanded=False):
    high_risk_thr = st.slider("High risk threshold", 50, 95, int(DEFAULT_HIGH_RISK_THRESHOLD), 1)
    med_risk_thr = st.slider("Medium risk threshold", 10, 70, int(DEFAULT_MED_RISK_THRESHOLD), 1)

with st.sidebar.expander("Clash overlap window", expanded=False):
    overlap_hours = st.slider("Time overlap window (hours)", 1, 24, 6, 1)

thresholds = Thresholds(
    diesel_kg_per_l=diesel_factor,
    electricity_kg_per_kwh=elec_factor,
    carbon_budget_kg=carbon_budget,
    high_carbon_alert_kg=high_carbon_alert,
    high_risk_threshold=high_risk_thr,
    med_risk_threshold=med_risk_thr,
)

st.sidebar.header("Process Emissions")
with st.sidebar.expander("Process factors (optional)", expanded=False):
    pf = DEFAULT_PROCESS_FACTORS.copy()
    for k in list(pf.keys()):
        pf[k] = st.number_input(f"{k} (kg CO₂e per unit)", min_value=0.0, max_value=10000.0, value=float(pf[k]), step=10.0)
process_factors = pf

st.sidebar.header("Scenario Presets")
scenario_name = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
scenario_key = SCENARIOS[scenario_name]

st.sidebar.header("Neo4j")
persist_to_graph = st.sidebar.checkbox("Enable push to Neo4j", value=False, disabled=not NEO4J_AVAILABLE)
if not NEO4J_AVAILABLE:
    st.sidebar.info("Neo4j integration not available (ensure neo4j_store.py is present and neo4j driver is installed).")

with st.sidebar.expander("Neo4j connection", expanded=False):
    neo_uri = st.text_input("NEO4J_URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo_user = st.text_input("NEO4J_USER", value=os.getenv("NEO4J_USER", "neo4j"))
    neo_pass = st.text_input("NEO4J_PASSWORD", value=os.getenv("NEO4J_PASSWORD", "neo4j"), type="password")
    neo_db = st.text_input("Database", value=os.getenv("NEO4J_DATABASE", "neo4j"))

df = read_upload(uploaded)
if df.empty:
    st.info("Upload a PTW file to begin. Use the sample CSV from the sidebar.")
    st.stop()

df = df.copy()
df.columns = [c.strip() for c in df.columns]

for col, default in CANONICAL_DEFAULTS:
    ensure_col(df, col, default)

df["status"] = df["status"].apply(norm_status)
df["signature_present"] = df["signature_present"].apply(safe_bool)
df["start_datetime"] = parse_dt(df["start_datetime"])
df["end_datetime"] = parse_dt(df["end_datetime"])

df["level2_detected"] = detect_level2(df)
df = compute_carbon(df, thresholds, process_factors)
df = compute_risk(df, thresholds)

clash_pairs = infer_clashes(df, overlap_hours=overlap_hours)
df = add_has_clash_flag(df, clash_pairs)

st.sidebar.header("🔍 Filters")
status_values = sorted(df["status"].dropna().astype(str).unique())
risk_values = ["Low", "Medium", "High"]
location_values = sorted(df["location"].dropna().astype(str).unique())
contractor_values = sorted(df["primary_contractor_company"].dropna().astype(str).unique())

status_filter = st.sidebar.multiselect("Status", status_values, default=status_values)
risk_filter = st.sidebar.multiselect("Risk band", risk_values, default=risk_values)
level2_only = st.sidebar.selectbox("Level II", ["All", "Yes", "No"], index=0)
over_budget_only = st.sidebar.selectbox("Over budget", ["All", "Yes", "No"], index=0)
high_carbon_only = st.sidebar.selectbox("High-carbon", ["All", "Yes", "No"], index=0)

loc_filter = st.sidebar.multiselect("Location", location_values, default=location_values)
con_filter = st.sidebar.multiselect("Contractor", contractor_values, default=contractor_values)

min_dt = df["start_datetime"].min()
max_dt = df["start_datetime"].max()
date_default = (min_dt.date(), max_dt.date()) if pd.notna(min_dt) and pd.notna(max_dt) else None
date_range = st.sidebar.date_input("Start date range", value=date_default)

filtered = df.copy()
if status_filter:
    filtered = filtered[filtered["status"].isin(status_filter)]
if risk_filter:
    filtered = filtered[filtered["risk_band"].isin(risk_filter)]
if level2_only != "All":
    filtered = filtered[filtered["level2_detected"].eq(level2_only == "Yes")]
if over_budget_only != "All":
    filtered = filtered[filtered["over_budget"].eq(over_budget_only == "Yes")]
if high_carbon_only != "All":
    filtered = filtered[filtered["high_carbon"].eq(high_carbon_only == "Yes")]
if loc_filter:
    filtered = filtered[filtered["location"].astype(str).isin(loc_filter)]
if con_filter:
    filtered = filtered[filtered["primary_contractor_company"].astype(str).isin(con_filter)]
if isinstance(date_range, tuple) and len(date_range) == 2 and date_range[0] and date_range[1]:
    start_d, end_d = date_range
    filtered = filtered[(filtered["start_datetime"].dt.date >= start_d) & (filtered["start_datetime"].dt.date <= end_d)]

scenario_df = apply_scenario(filtered, scenario_key)

live_cnt = int((scenario_df["status"] == "Live").sum())
high_cnt = int((scenario_df["risk_band"] == "High").sum())
lvl2_cnt = int(scenario_df["level2_detected"].sum())
ob_cnt = int(scenario_df["over_budget"].sum())
clash_cnt = int(scenario_df["has_clash"].sum())
total_carbon = float(scenario_df["ptw_kg_co2e"].sum()) if not scenario_df.empty else 0.0
avg_cis = float(scenario_df["CIS"].mean()) if not scenario_df.empty else 0.0

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("Permits", len(scenario_df))
k2.metric("Live", live_cnt)
k3.metric("High risk", high_cnt)
k4.metric("Level II", lvl2_cnt)
k5.metric("Over budget", ob_cnt)
k6.metric("Clashes", clash_cnt)
k7.metric("Total CO₂e", f"{total_carbon:,.0f} kg")

st.caption(
    f"Scenario: **{scenario_name}** | Average CIS: **{avg_cis:.2f}** | "
    "Use filters + scenario presets to drill into live ops, risk, carbon, and clashes."
)

tab_overview, tab_carbon, tab_relationships, tab_whatif, tab_data = st.tabs(
    ["Overview", "Carbon", "Relationships", "What‑If", "Data"]
)

with tab_overview:
    c1, c2 = st.columns(2)
    risk_dist = scenario_df.groupby("risk_band", as_index=False).size()
    if not risk_dist.empty:
        c1.plotly_chart(px.bar(risk_dist, x="risk_band", y="size", title="Risk distribution"), use_container_width=True)
    else:
        c1.info("No data for risk distribution.")

    st_dist = scenario_df.groupby("status", as_index=False).size()
    if not st_dist.empty:
        c2.plotly_chart(px.pie(st_dist, names="status", values="size", title="Status distribution"), use_container_width=True)
    else:
        c2.info("No data for status distribution.")

    c3, c4 = st.columns(2)
    if not scenario_df.empty:
        loc_risk = (
            scenario_df.groupby("location", as_index=False)
            .agg(avg_risk=("risk_score", "mean"), permits=("permit_no", "count"))
            .sort_values(["avg_risk", "permits"], ascending=False)
            .head(10)
        )
        if not loc_risk.empty:
            c3.plotly_chart(px.bar(loc_risk, x="location", y="avg_risk", title="Top locations by avg risk (Top 10)"), use_container_width=True)

        con_risk = (
            scenario_df.groupby("primary_contractor_company", as_index=False)
            .agg(avg_risk=("risk_score", "mean"), permits=("permit_no", "count"))
            .sort_values(["avg_risk", "permits"], ascending=False)
            .head(10)
        )
        if not con_risk.empty:
            c4.plotly_chart(px.bar(con_risk, x="primary_contractor_company", y="avg_risk", title="Top contractors by avg risk (Top 10)"), use_container_width=True)

with tab_carbon:
    if scenario_df.empty:
        st.info("No data in this filter/scenario.")
    else:
        st.subheader("Daily carbon trend")
        rolling_days = st.slider("Rolling window (days)", 1, 30, 7, 1, help="Smooths the daily trend using a rolling average.")
        d = scenario_df.copy()
        d["date"] = d["start_datetime"].dt.date
        trend = d.groupby("date", as_index=False).agg(ptw_kg_co2e=("ptw_kg_co2e", "sum")).sort_values("date")
        trend["rolling_avg"] = trend["ptw_kg_co2e"].rolling(rolling_days, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["date"], y=trend["ptw_kg_co2e"], mode="lines+markers", name="Daily total"))
        fig.add_trace(go.Scatter(x=trend["date"], y=trend["rolling_avg"], mode="lines", name=f"{rolling_days}d rolling avg"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        by_loc = (
            d.groupby("location", as_index=False)
            .agg(ptw_kg_co2e=("ptw_kg_co2e", "sum"))
            .sort_values("ptw_kg_co2e", ascending=False)
            .head(10)
        )
        if not by_loc.empty:
            c1.plotly_chart(px.bar(by_loc, x="location", y="ptw_kg_co2e", title="Carbon by location (Top 10)"), use_container_width=True)

        by_con = (
            d.groupby("primary_contractor_company", as_index=False)
            .agg(ptw_kg_co2e=("ptw_kg_co2e", "sum"))
            .sort_values("ptw_kg_co2e", ascending=False)
            .head(10)
        )
        if not by_con.empty:
            c2.plotly_chart(px.bar(by_con, x="primary_contractor_company", y="ptw_kg_co2e", title="Carbon by contractor (Top 10)"), use_container_width=True)

with tab_relationships:
    st.subheader("Clashes / overlaps")
    if not clash_pairs:
        st.info("No clashes inferred/declared in the current dataset.")
    else:
        st.caption("Clashes are derived from declared references or inferred overlaps (same location within time window).")
        fig = build_clash_network(df, clash_pairs)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(clash_pairs), use_container_width=True, height=240)
        st.code(
            "MATCH (a:Permit)-[r:CLASHES_WITH]->(b:Permit)\\nRETURN a.permit_no AS a, r.reason AS reason, b.permit_no AS b\\nLIMIT 200;",
            language="cypher"
        )

with tab_whatif:
    st.subheader("What‑If: simulate changes to one permit")
    st.caption("Adjust energy quantities or thresholds and see impact on CO₂e and budget status.")
    if scenario_df.empty:
        st.info("No permits available to simulate in current filters.")
    else:
        permit_options = scenario_df["permit_no"].astype(str).tolist()
        sel = st.selectbox("Select permit to simulate", permit_options, index=0)
        base_row = scenario_df[scenario_df["permit_no"].astype(str) == str(sel)].iloc[0].copy()

        c1, c2, c3 = st.columns(3)
        fuel_new = c1.number_input("Fuel qty (litres)", min_value=0.0, value=float(base_row.get("fuel_qty", 0.0)), step=1.0)
        elec_new = c2.number_input("Electricity (kWh)", min_value=0.0, value=float(base_row.get("electricity_kwh", 0.0)), step=1.0)
        proc_new = c3.number_input("Process activity qty", min_value=0.0, value=float(base_row.get("process_activity_qty", 0.0)), step=0.1)

        proc_type = st.text_input("Process type (optional)", value=str(base_row.get("process_type", "") or ""))

        sim = base_row.to_dict()
        sim["fuel_qty"] = fuel_new
        sim["electricity_kwh"] = elec_new
        sim["process_activity_qty"] = proc_new
        sim["process_type"] = proc_type

        sim_df = pd.DataFrame([sim])
        sim_df = compute_carbon(sim_df, thresholds, process_factors)

        sim_total = float(sim_df["ptw_kg_co2e"].iloc[0])
        delta = sim_total - float(base_row["ptw_kg_co2e"])
        sim_over_budget = bool(sim_df["over_budget"].iloc[0])

        r1, r2, r3 = st.columns(3)
        r1.metric("Simulated CO₂e (kg)", f"{sim_total:,.1f}", f"{delta:+,.1f}")
        r2.metric("Over budget?", "Yes" if sim_over_budget else "No")
        r3.metric("High carbon alert?", "Yes" if sim_total >= thresholds.high_carbon_alert_kg else "No")

        comp = pd.DataFrame([{
            "component": "combustion_kg", "kg_co2e": float(sim_df["combustion_kg"].iloc[0])
        }, {
            "component": "electricity_kg", "kg_co2e": float(sim_df["electricity_kg"].iloc[0])
        }, {
            "component": "process_kg", "kg_co2e": float(sim_df["process_kg"].iloc[0])
        }])
        st.plotly_chart(px.bar(comp, x="component", y="kg_co2e", title="Simulated carbon components"), use_container_width=True)

with tab_data:
    st.subheader("Permit table (sortable)")
    cols = [c for c in DISPLAY_COLUMNS if c in scenario_df.columns]
    table_df = scenario_df[cols].copy() if cols else scenario_df.copy()
    if "risk_score" in table_df.columns:
        table_df = table_df.sort_values("risk_score", ascending=False)
    st.dataframe(table_df, use_container_width=True, height=420)

# -----------------------------
# Push to Neo4j (optional)
# -----------------------------
if persist_to_graph and NEO4J_AVAILABLE:
    st.sidebar.markdown("---")
    if st.sidebar.button("⬆️ Push full dataset to Neo4j"):
        try:
            store = Neo4jStore(neo_uri, neo_user, neo_pass, database=neo_db)
            store.bootstrap()
            store.upsert(df, clash_pairs)
            count = store.count_permits()
            store.close()
            st.sidebar.success(f"Loaded graph successfully. Permits in DB: {count}")
        except Exception as e:
            st.sidebar.error(f"Neo4j load failed: {e}")
