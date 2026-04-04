"""
UEBA Insider Threat Detection - SOC Dashboard
==============================================
Run from project root:
    streamlit run dashboard.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from datetime import datetime

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="UEBA SOC Dashboard",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Custom CSS -------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #080c14;
    color: #c8d6e5;
}

.stApp {
    background: radial-gradient(ellipse at top left, #0d1b2a 0%, #080c14 60%);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Title bar */
.dashboard-header {
    background: linear-gradient(90deg, #0a1628 0%, #0d2240 50%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 4px;
    padding: 18px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 0 40px rgba(0, 120, 255, 0.08);
}

.dashboard-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 22px;
    color: #4a9eff;
    letter-spacing: 2px;
    margin: 0;
}

.dashboard-subtitle {
    font-size: 13px;
    color: #5a7a9a;
    margin: 4px 0 0 0;
    letter-spacing: 1px;
}

.dashboard-time {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #2a5a8a;
    text-align: right;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1e30 0%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 4px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

.metric-card-total::before  { background: #4a9eff; }
.metric-card-high::before   { background: #ff4444; }
.metric-card-medium::before { background: #ffa500; }
.metric-card-accuracy::before { background: #00cc66; }

.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 36px;
    font-weight: bold;
    margin: 0;
    line-height: 1;
}

.metric-value-total    { color: #4a9eff; }
.metric-value-high     { color: #ff4444; }
.metric-value-medium   { color: #ffa500; }
.metric-value-accuracy { color: #00cc66; }

.metric-label {
    font-size: 12px;
    color: #5a7a9a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 8px 0 0 0;
}

.metric-sublabel {
    font-size: 11px;
    color: #2a4a6a;
    margin: 4px 0 0 0;
}

/* Section headers */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #4a9eff;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-left: 3px solid #4a9eff;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628;
    border-bottom: 1px solid #1a3a5c;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    color: #3a6a9a;
    padding: 12px 24px;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: #4a9eff !important;
    border-bottom: 2px solid #4a9eff !important;
    background: transparent !important;
}

/* Risk badges */
.badge-high   { color: #ff4444; font-weight: 700; letter-spacing: 1px; }
.badge-medium { color: #ffa500; font-weight: 700; letter-spacing: 1px; }
.badge-low    { color: #00cc66; font-weight: 700; letter-spacing: 1px; }

/* User detail card */
.detail-card {
    background: #0d1e30;
    border: 1px solid #1a3a5c;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.detail-card-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #4a9eff;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

/* Explanation items */
.explanation-item {
    background: rgba(255, 68, 68, 0.05);
    border-left: 3px solid #ff4444;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 0 4px 4px 0;
    font-size: 13px;
    color: #c8d6e5;
}

.explanation-item-medium {
    border-left-color: #ffa500;
    background: rgba(255, 165, 0, 0.05);
}

.explanation-item-low {
    border-left-color: #00cc66;
    background: rgba(0, 204, 102, 0.05);
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1a3a5c !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0d1e30;
    border-color: #1a3a5c;
    color: #c8d6e5;
}

/* Search input */
.stTextInput > div > div > input {
    background: #0d1e30;
    border-color: #1a3a5c;
    color: #c8d6e5;
    font-family: 'Share Tech Mono', monospace;
}

/* Scanline effect */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)


# -- Data loading -----------------------------------------------------------
@st.cache_data
def load_data():
    processed = os.path.join(BASE_DIR, "dataset", "processed")
    data = {}

    files = {
        "risk"     : "final_risk_scores.csv",
        "ueba"     : "ueba_scores.csv",
        "accuracy" : "accuracy_results.csv",
        "rules"    : "rule_scores.csv",
        "features" : "user_behavior_features.csv",
        "graph"    : "graph_scores.csv",
    }

    for key, fname in files.items():
        path = os.path.join(processed, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = None

    graph_img_path = os.path.join(processed, "network_graph.png")
    data["graph_img"] = graph_img_path if os.path.exists(graph_img_path) else None

    return data

data = load_data()

# -- Check minimum required data --------------------------------------------
if data["risk"] is None:
    st.error("final_risk_scores.csv not found. Please run:  python main.py")
    st.stop()

risk_df = data["risk"]
high_df   = risk_df[risk_df["risk_level"] == "HIGH"]
medium_df = risk_df[risk_df["risk_level"] == "MEDIUM"]
low_df    = risk_df[risk_df["risk_level"] == "LOW"]

# Accuracy metrics
acc_df = data["accuracy"]
if acc_df is not None:
    y_true = acc_df["ueba_anomaly"].values
    y_pred = acc_df["lstm_anomaly"].values
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy_val  = accuracy_score(y_true, y_pred) * 100
    precision_val = precision_score(y_true, y_pred, zero_division=0) * 100
    recall_val    = recall_score(y_true, y_pred, zero_division=0) * 100
    f1_val        = f1_score(y_true, y_pred, zero_division=0) * 100
else:
    accuracy_val = precision_val = recall_val = f1_val = 0


# -- Header -----------------------------------------------------------------
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <p class="dashboard-title">// UEBA INSIDER THREAT DETECTION SYSTEM</p>
        <p class="dashboard-subtitle">SECURITY OPERATIONS CENTER &nbsp;|&nbsp; HYBRID DETECTION ENGINE</p>
    </div>
    <div class="dashboard-time">
        SYSTEM TIME<br>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <span style="color:#1a5a8a">STATUS: OPERATIONAL</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -- Summary Cards ----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card metric-card-total">
        <p class="metric-value metric-value-total">{len(risk_df):,}</p>
        <p class="metric-label">Total Users</p>
        <p class="metric-sublabel">Analyzed in this run</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card metric-card-high">
        <p class="metric-value metric-value-high">{len(high_df)}</p>
        <p class="metric-label">High Risk</p>
        <p class="metric-sublabel">Immediate investigation needed</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card metric-card-medium">
        <p class="metric-value metric-value-medium">{len(medium_df)}</p>
        <p class="metric-label">Medium Risk</p>
        <p class="metric-sublabel">Monitor closely</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card metric-card-accuracy">
        <p class="metric-value metric-value-accuracy">{accuracy_val:.1f}%</p>
        <p class="metric-label">Model Accuracy</p>
        <p class="metric-sublabel">LSTM + UEBA + Rules</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -- Tabs -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "  OVERVIEW  ",
    "  USER RISK TABLE  ",
    "  USER INVESTIGATION  ",
    "  SYSTEM ANALYTICS  "
])


# ==========================================================================
# TAB 1 — OVERVIEW
# ==========================================================================
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)

        fig_pie = go.Figure(go.Pie(
            labels=["HIGH", "MEDIUM", "LOW"],
            values=[len(high_df), len(medium_df), len(low_df)],
            hole=0.6,
            marker=dict(colors=["#ff4444", "#ffa500", "#00cc66"],
                        line=dict(color="#080c14", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Share Tech Mono", size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Users: %{value}<br>%{percent}<extra></extra>"
        ))
        fig_pie.add_annotation(
            text=f"<b style='font-size:24px'>{len(high_df) + len(medium_df)}</b><br>Threats",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Share Tech Mono", size=14, color="#4a9eff")
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d6e5"),
            legend=dict(font=dict(family="Share Tech Mono", color="#c8d6e5"),
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Detection Method Coverage</p>', unsafe_allow_html=True)

        lstm_caught = int(risk_df["lstm_anomaly"].sum())
        ueba_caught = int(risk_df["ueba_anomaly"].sum())
        rule_caught = int(risk_df["rule_anomaly"].sum())
        all_three   = int(((risk_df["lstm_anomaly"]==1) & (risk_df["ueba_anomaly"]==1) & (risk_df["rule_anomaly"]==1)).sum())

        fig_bar = go.Figure()
        methods = ["LSTM Autoencoder", "UEBA Z-Score", "Rule Engine", "All Three"]
        values  = [lstm_caught, ueba_caught, rule_caught, all_three]
        colors  = ["#4a9eff", "#a855f7", "#ffa500", "#ff4444"]

        fig_bar.add_trace(go.Bar(
            x=methods, y=values,
            marker=dict(color=colors, line=dict(color="#080c14", width=1)),
            text=values, textposition="outside",
            textfont=dict(family="Share Tech Mono", color="white", size=14),
            hovertemplate="<b>%{x}</b><br>Users flagged: %{y}<extra></extra>"
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d6e5", family="Share Tech Mono"),
            xaxis=dict(showgrid=False, color="#3a6a9a"),
            yaxis=dict(showgrid=True, gridcolor="#0d2240", color="#3a6a9a"),
            margin=dict(t=40, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Top 15 risk users bar chart
    st.markdown('<p class="section-header">Top 15 Highest Risk Users</p>', unsafe_allow_html=True)

    top15 = risk_df.head(15)
    color_map = {"HIGH": "#ff4444", "MEDIUM": "#ffa500", "LOW": "#00cc66"}
    bar_colors = [color_map[r] for r in top15["risk_level"]]

    fig_top = go.Figure(go.Bar(
        x=top15["user"],
        y=top15["final_risk_score"],
        marker=dict(color=bar_colors, line=dict(color="#080c14", width=1)),
        text=[f"{s:.3f}" for s in top15["final_risk_score"]],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", color="white", size=11),
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.4f}<extra></extra>"
    ))
    fig_top.add_hline(y=0.60, line_dash="dot", line_color="#ff4444",
                      annotation_text="HIGH threshold", annotation_font_color="#ff4444")
    fig_top.add_hline(y=0.30, line_dash="dot", line_color="#ffa500",
                      annotation_text="MEDIUM threshold", annotation_font_color="#ffa500")
    fig_top.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d6e5", family="Share Tech Mono"),
        xaxis=dict(showgrid=False, color="#3a6a9a", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#0d2240", color="#3a6a9a", range=[0, 1.1]),
        margin=dict(t=40, b=80, l=20, r=20),
        height=380
    )
    st.plotly_chart(fig_top, use_container_width=True)


# ==========================================================================
# TAB 2 — USER RISK TABLE
# ==========================================================================
with tab2:
    st.markdown('<p class="section-header">All Users — Risk Assessment</p>', unsafe_allow_html=True)

    col_search, col_filter, col_sort = st.columns([2, 1, 1])
    with col_search:
        search = st.text_input("Search user", placeholder="e.g. DTAA/IRC0991")
    with col_filter:
        level_filter = st.selectbox("Risk Level", ["ALL", "HIGH", "MEDIUM", "LOW"])
    with col_sort:
        sort_by = st.selectbox("Sort by", ["Risk Score", "UEBA Score", "Rule Score"])

    display_df = risk_df.copy()

    if search:
        display_df = display_df[display_df["user"].str.contains(search, case=False)]

    if level_filter != "ALL":
        display_df = display_df[display_df["risk_level"] == level_filter]

    sort_col = {
        "Risk Score" : "final_risk_score",
        "UEBA Score" : "ueba_score_weighted",
        "Rule Score" : "rule_score"
    }[sort_by]
    display_df = display_df.sort_values(sort_col, ascending=False)

    # Build display table
    table_df = display_df[[
        "user", "risk_level", "final_risk_score",
        "lstm_anomaly", "ueba_anomaly", "rule_anomaly", "rules_violated"
    ]].copy()

    table_df.columns = [
        "User", "Risk Level", "Final Score",
        "LSTM", "UEBA", "Rules", "Violations"
    ]
    table_df["LSTM"]  = table_df["LSTM"].map({1: "YES", 0: "no"})
    table_df["UEBA"]  = table_df["UEBA"].map({1: "YES", 0: "no"})
    table_df["Rules"] = table_df["Rules"].map({1: "YES", 0: "no"})
    table_df["Final Score"] = table_df["Final Score"].round(4)

    st.dataframe(
        table_df,
        use_container_width=True,
        height=600,
        column_config={
            "Risk Level": st.column_config.TextColumn(width="small"),
            "Final Score": st.column_config.NumberColumn(format="%.4f"),
            "LSTM": st.column_config.TextColumn(width="small"),
            "UEBA": st.column_config.TextColumn(width="small"),
            "Rules": st.column_config.TextColumn(width="small"),
            "Violations": st.column_config.NumberColumn(width="small"),
        },
        hide_index=True
    )
    st.caption(f"Showing {len(display_df)} of {len(risk_df)} users")


# ==========================================================================
# TAB 3 — USER INVESTIGATION
# ==========================================================================
with tab3:
    st.markdown('<p class="section-header">User Investigation Panel</p>', unsafe_allow_html=True)

    all_users = risk_df["user"].tolist()
    selected_user = st.selectbox(
        "Select user to investigate",
        all_users,
        format_func=lambda u: f"{u}  —  {risk_df[risk_df['user']==u]['risk_level'].values[0]}  "
                              f"({risk_df[risk_df['user']==u]['final_risk_score'].values[0]:.4f})"
    )

    if selected_user:
        row = risk_df[risk_df["user"] == selected_user].iloc[0]
        level = row["risk_level"]
        level_color = {"HIGH": "#ff4444", "MEDIUM": "#ffa500", "LOW": "#00cc66"}[level]

        # User header
        st.markdown(f"""
        <div style="background:#0d1e30; border:1px solid {level_color};
                    border-radius:4px; padding:20px 24px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <p style="font-family:'Share Tech Mono',monospace; font-size:20px;
                               color:{level_color}; margin:0;">{selected_user}</p>
                    <p style="color:#5a7a9a; margin:4px 0 0 0; font-size:13px; letter-spacing:1px;">
                        INSIDER THREAT INVESTIGATION REPORT
                    </p>
                </div>
                <div style="text-align:right;">
                    <p style="font-family:'Share Tech Mono',monospace; font-size:32px;
                               color:{level_color}; margin:0;">{row['final_risk_score']:.4f}</p>
                    <p style="color:{level_color}; margin:0; font-size:14px;
                               letter-spacing:3px; font-weight:700;">{level} RISK</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score breakdown
        col_a, col_b, col_c = st.columns(3)

        def score_gauge(value, title, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number=dict(font=dict(family="Share Tech Mono", color=color, size=28)),
                title=dict(text=title, font=dict(family="Share Tech Mono",
                                                  color="#5a7a9a", size=12)),
                gauge=dict(
                    axis=dict(range=[0, 1], tickcolor="#1a3a5c",
                              tickfont=dict(color="#3a6a9a", size=10)),
                    bar=dict(color=color),
                    bgcolor="#0a1628",
                    bordercolor="#1a3a5c",
                    steps=[
                        dict(range=[0, 0.3], color="#0a1e0a"),
                        dict(range=[0.3, 0.6], color="#1a1a0a"),
                        dict(range=[0.6, 1], color="#1a0a0a"),
                    ],
                    threshold=dict(line=dict(color=color, width=2), value=value)
                )
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin=dict(t=40, b=10, l=30, r=30)
            )
            return fig

        with col_a:
            st.plotly_chart(score_gauge(row["lstm_score_norm"], "LSTM SCORE", "#4a9eff"),
                           use_container_width=True)
            flagged = "FLAGGED" if row["lstm_anomaly"] else "CLEAR"
            color   = "#ff4444" if row["lstm_anomaly"] else "#00cc66"
            st.markdown(f"<p style='text-align:center; color:{color}; font-family:Share Tech Mono; "
                       f"letter-spacing:2px; font-size:12px;'>{flagged}</p>", unsafe_allow_html=True)

        with col_b:
            st.plotly_chart(score_gauge(row["ueba_score_norm"], "UEBA SCORE", "#a855f7"),
                           use_container_width=True)
            flagged = "FLAGGED" if row["ueba_anomaly"] else "CLEAR"
            color   = "#ff4444" if row["ueba_anomaly"] else "#00cc66"
            st.markdown(f"<p style='text-align:center; color:{color}; font-family:Share Tech Mono; "
                       f"letter-spacing:2px; font-size:12px;'>{flagged}</p>", unsafe_allow_html=True)

        with col_c:
            st.plotly_chart(score_gauge(row["rule_score_norm"], "RULE SCORE", "#ffa500"),
                           use_container_width=True)
            flagged = "FLAGGED" if row["rule_anomaly"] else "CLEAR"
            color   = "#ff4444" if row["rule_anomaly"] else "#00cc66"
            st.markdown(f"<p style='text-align:center; color:{color}; font-family:Share Tech Mono; "
                       f"letter-spacing:2px; font-size:12px;'>{flagged}</p>", unsafe_allow_html=True)

        # Activity profile
        st.markdown('<p class="section-header">Activity Profile</p>', unsafe_allow_html=True)

        if data["features"] is not None:
            feat = data["features"][data["features"]["user"] == selected_user]
            if not feat.empty:
                feat_row = feat.iloc[0]
                f1, f2, f3, f4 = st.columns(4)
                metrics = [
                    (f1, "LOGONS",         feat_row.get("logon_count", 0),              "#4a9eff"),
                    (f2, "AFTER-HRS LOGIN", feat_row.get("after_hours_logon_count", 0), "#ffa500"),
                    (f3, "DEVICE CONNECTS", feat_row.get("device_connect_count", 0),    "#a855f7"),
                    (f4, "HTTP VISITS",     feat_row.get("http_count", 0),              "#00cc66"),
                ]
                for col, label, val, color in metrics:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center; padding:14px;">
                            <p style="font-family:Share Tech Mono; font-size:26px;
                                       color:{color}; margin:0;">{int(val):,}</p>
                            <p style="font-size:11px; color:#5a7a9a; letter-spacing:2px;
                                       margin:6px 0 0 0; text-transform:uppercase;">{label}</p>
                        </div>""", unsafe_allow_html=True)

        # Rule violations
        st.markdown('<p class="section-header">Rule Violation Details</p>', unsafe_allow_html=True)

        explanation = str(row.get("explanation", "No rules violated"))
        if explanation != "No rules violated" and explanation != "nan":
            for item in explanation.split(" || "):
                sev_class = "explanation-item"
                if "HIGH" in item:
                    sev_class = "explanation-item"
                elif "MEDIUM" in item:
                    sev_class = "explanation-item explanation-item-medium"
                else:
                    sev_class = "explanation-item explanation-item-low"
                st.markdown(f'<div class="{sev_class}">{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="explanation-item explanation-item-low">'
                       'No rule violations detected for this user.</div>',
                       unsafe_allow_html=True)


# ==========================================================================
# TAB 4 — SYSTEM ANALYTICS
# ==========================================================================
with tab4:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-header">Model Performance Metrics</p>',
                   unsafe_allow_html=True)

        metrics_data = {
            "Accuracy" : accuracy_val,
            "Precision": precision_val,
            "Recall"   : recall_val,
            "F1 Score" : f1_val
        }
        metric_colors = ["#00cc66", "#4a9eff", "#ffa500", "#a855f7"]

        fig_metrics = go.Figure()
        for i, (name, val) in enumerate(metrics_data.items()):
            fig_metrics.add_trace(go.Bar(
                name=name, x=[name], y=[val],
                marker_color=metric_colors[i],
                text=[f"{val:.1f}%"], textposition="outside",
                textfont=dict(family="Share Tech Mono", color="white", size=14),
                showlegend=False
            ))
        fig_metrics.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d6e5", family="Share Tech Mono"),
            xaxis=dict(showgrid=False, color="#3a6a9a"),
            yaxis=dict(showgrid=True, gridcolor="#0d2240", color="#3a6a9a",
                      range=[0, 115]),
            margin=dict(t=40, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        # Confusion matrix
        if acc_df is not None:
            st.markdown('<p class="section-header">Confusion Matrix</p>',
                       unsafe_allow_html=True)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted Normal", "Predicted Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                colorscale=[[0, "#0a1628"], [1, "#4a9eff"]],
                text=cm, texttemplate="%{text}",
                textfont=dict(family="Share Tech Mono", size=20, color="white"),
                showscale=False,
                hovertemplate="<b>%{y} / %{x}</b><br>Count: %{z}<extra></extra>"
            ))
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8d6e5", family="Share Tech Mono", size=12),
                xaxis=dict(color="#3a6a9a"),
                yaxis=dict(color="#3a6a9a"),
                margin=dict(t=20, b=20, l=20, r=20),
                height=280
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Network Behavior Graph</p>',
                   unsafe_allow_html=True)
        if data["graph_img"]:
            img = Image.open(data["graph_img"])
            st.image(img, use_container_width=True,
                    caption="Red=Suspicious  Blue=Normal Users  Gray=PCs  Green=Websites")
        else:
            st.info("Network graph not found. Run the pipeline to generate it.")

        # Top central nodes
        if data["graph"] is not None:
            st.markdown('<p class="section-header">Top 10 Central Network Nodes</p>',
                       unsafe_allow_html=True)
            graph_df = data["graph"].head(10)[["node", "centrality"]].copy()
            graph_df["centrality"] = graph_df["centrality"].round(4)
            st.dataframe(graph_df, use_container_width=True,
                        hide_index=True, height=240)