"""
UEBA Insider Threat Detection - SOC Dashboard
==============================================
Includes:
  Tab 1: Overview
  Tab 2: User Risk Table
  Tab 3: User Investigation
  Tab 4: System Analytics
  Tab 5: Evidence Exporter    <- NEW
  Tab 6: Live Demo Mode       <- NEW

Run from project root:
    streamlit run dashboard.py
"""

import os
import sys
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# -- Custom CSS -------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #080c14; color: #c8d6e5; }
.stApp { background: radial-gradient(ellipse at top left, #0d1b2a 0%, #080c14 60%); }
#MainMenu, footer, header { visibility: hidden; }
.dashboard-header { background: linear-gradient(90deg,#0a1628 0%,#0d2240 50%,#0a1628 100%); border:1px solid #1a3a5c; border-radius:4px; padding:18px 28px; margin-bottom:24px; display:flex; align-items:center; justify-content:space-between; box-shadow:0 0 40px rgba(0,120,255,0.08); }
.dashboard-title { font-family:'Share Tech Mono',monospace; font-size:22px; color:#4a9eff; letter-spacing:2px; margin:0; }
.dashboard-subtitle { font-size:13px; color:#5a7a9a; margin:4px 0 0 0; letter-spacing:1px; }
.dashboard-time { font-family:'Share Tech Mono',monospace; font-size:12px; color:#2a5a8a; text-align:right; }
.metric-card { background:linear-gradient(135deg,#0d1e30 0%,#0a1628 100%); border:1px solid #1a3a5c; border-radius:4px; padding:20px 24px; position:relative; overflow:hidden; }
.metric-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; }
.metric-card-total::before { background:#4a9eff; }
.metric-card-high::before { background:#ff4444; }
.metric-card-medium::before { background:#ffa500; }
.metric-card-accuracy::before { background:#00cc66; }
.metric-value { font-family:'Share Tech Mono',monospace; font-size:36px; font-weight:bold; margin:0; line-height:1; }
.metric-value-total { color:#4a9eff; }
.metric-value-high { color:#ff4444; }
.metric-value-medium { color:#ffa500; }
.metric-value-accuracy { color:#00cc66; }
.metric-label { font-size:12px; color:#5a7a9a; letter-spacing:2px; text-transform:uppercase; margin:8px 0 0 0; }
.metric-sublabel { font-size:11px; color:#2a4a6a; margin:4px 0 0 0; }
.section-header { font-family:'Share Tech Mono',monospace; font-size:13px; color:#4a9eff; letter-spacing:3px; text-transform:uppercase; border-left:3px solid #4a9eff; padding-left:12px; margin:24px 0 16px 0; }
.stTabs [data-baseweb="tab-list"] { background:#0a1628; border-bottom:1px solid #1a3a5c; gap:0; }
.stTabs [data-baseweb="tab"] { font-family:'Share Tech Mono',monospace; font-size:12px; letter-spacing:2px; color:#3a6a9a; padding:12px 20px; border-bottom:2px solid transparent; }
.stTabs [aria-selected="true"] { color:#4a9eff !important; border-bottom:2px solid #4a9eff !important; background:transparent !important; }
.detail-card { background:#0d1e30; border:1px solid #1a3a5c; border-radius:4px; padding:16px 20px; margin-bottom:12px; }
.explanation-item { background:rgba(255,68,68,0.05); border-left:3px solid #ff4444; padding:8px 12px; margin:6px 0; border-radius:0 4px 4px 0; font-size:13px; color:#c8d6e5; }
.demo-card-high { background:rgba(255,68,68,0.08); border:1px solid #ff4444; border-radius:4px; padding:16px 20px; margin-bottom:12px; }
.demo-card-medium { background:rgba(255,165,0,0.08); border:1px solid #ffa500; border-radius:4px; padding:16px 20px; margin-bottom:12px; }
.demo-card-low { background:rgba(0,204,102,0.08); border:1px solid #00cc66; border-radius:4px; padding:16px 20px; margin-bottom:12px; }
.demo-injected { font-family:'Share Tech Mono',monospace; font-size:11px; color:#ff4444; letter-spacing:2px; }
</style>
""", unsafe_allow_html=True)


# -- Data loading -----------------------------------------------------------
@st.cache_data(ttl=5)
def load_data():
    processed = os.path.join(BASE_DIR, "dataset", "processed")
    data = {}
    files = {
        "risk"    : "final_risk_scores.csv",
        "ueba"    : "ueba_scores.csv",
        "accuracy": "accuracy_results.csv",
        "rules"   : "rule_scores.csv",
        "features": "user_behavior_features.csv",
        "graph"   : "graph_scores.csv",
    }
    for key, fname in files.items():
        path = os.path.join(processed, fname)
        data[key] = pd.read_csv(path) if os.path.exists(path) else None
    graph_img = os.path.join(processed, "network_graph.png")
    data["graph_img"] = graph_img if os.path.exists(graph_img) else None
    return data

data = load_data()

if data["risk"] is None:
    st.error("final_risk_scores.csv not found. Please run:  python main.py")
    st.stop()

risk_df   = data["risk"]
high_df   = risk_df[risk_df["risk_level"] == "HIGH"]
medium_df = risk_df[risk_df["risk_level"] == "MEDIUM"]
low_df    = risk_df[risk_df["risk_level"] == "LOW"]

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
    st.markdown(f"""<div class="metric-card metric-card-total">
        <p class="metric-value metric-value-total">{len(risk_df):,}</p>
        <p class="metric-label">Total Users</p>
        <p class="metric-sublabel">Analyzed in this run</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card metric-card-high">
        <p class="metric-value metric-value-high">{len(high_df)}</p>
        <p class="metric-label">High Risk</p>
        <p class="metric-sublabel">Immediate investigation needed</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card metric-card-medium">
        <p class="metric-value metric-value-medium">{len(medium_df)}</p>
        <p class="metric-label">Medium Risk</p>
        <p class="metric-sublabel">Monitor closely</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card metric-card-accuracy">
        <p class="metric-value metric-value-accuracy">{accuracy_val:.1f}%</p>
        <p class="metric-label">Model Accuracy</p>
        <p class="metric-sublabel">LSTM + UEBA + Rules</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -- Tabs -------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  OVERVIEW  ",
    "  USER RISK TABLE  ",
    "  USER INVESTIGATION  ",
    "  SYSTEM ANALYTICS  ",
    "  EVIDENCE EXPORTER  ",
    "  LIVE DEMO MODE  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["HIGH","MEDIUM","LOW"],
            values=[len(high_df), len(medium_df), len(low_df)],
            hole=0.6,
            marker=dict(colors=["#ff4444","#ffa500","#00cc66"], line=dict(color="#080c14", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Share Tech Mono", size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Users: %{value}<br>%{percent}<extra></extra>"
        ))
        fig_pie.add_annotation(text=f"<b style='font-size:24px'>{len(high_df)+len(medium_df)}</b><br>Threats",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Share Tech Mono", size=14, color="#4a9eff"))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d6e5"), legend=dict(font=dict(family="Share Tech Mono",color="#c8d6e5"),bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20,b=20,l=20,r=20), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Detection Method Coverage</p>', unsafe_allow_html=True)
        lstm_caught = int(risk_df["lstm_anomaly"].sum())
        ueba_caught = int(risk_df["ueba_anomaly"].sum())
        rule_caught = int(risk_df["rule_anomaly"].sum())
        all_three   = int(((risk_df["lstm_anomaly"]==1)&(risk_df["ueba_anomaly"]==1)&(risk_df["rule_anomaly"]==1)).sum())
        fig_bar = go.Figure()
        for name, val, col in [("LSTM Autoencoder",lstm_caught,"#4a9eff"),("UEBA Z-Score",ueba_caught,"#a855f7"),("Rule Engine",rule_caught,"#ffa500"),("All Three",all_three,"#ff4444")]:
            fig_bar.add_trace(go.Bar(x=[name], y=[val], marker_color=col, text=[val], textposition="outside",
                textfont=dict(family="Share Tech Mono",color="white",size=14), showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Users: {val}<extra></extra>"))
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d6e5",family="Share Tech Mono"),
            xaxis=dict(showgrid=False,color="#3a6a9a"), yaxis=dict(showgrid=True,gridcolor="#0d2240",color="#3a6a9a"),
            margin=dict(t=40,b=20,l=20,r=20), height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<p class="section-header">Top 15 Highest Risk Users</p>', unsafe_allow_html=True)
    top15 = risk_df.head(15)
    color_map = {"HIGH":"#ff4444","MEDIUM":"#ffa500","LOW":"#00cc66"}
    bar_colors = [color_map[r] for r in top15["risk_level"]]
    fig_top = go.Figure(go.Bar(
        x=top15["user"], y=top15["final_risk_score"],
        marker=dict(color=bar_colors, line=dict(color="#080c14",width=1)),
        text=[f"{s:.3f}" for s in top15["final_risk_score"]], textposition="outside",
        textfont=dict(family="Share Tech Mono",color="white",size=11),
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.4f}<extra></extra>"
    ))
    fig_top.add_hline(y=0.60, line_dash="dot", line_color="#ff4444", annotation_text="HIGH threshold", annotation_font_color="#ff4444")
    fig_top.add_hline(y=0.30, line_dash="dot", line_color="#ffa500", annotation_text="MEDIUM threshold", annotation_font_color="#ffa500")
    fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d6e5",family="Share Tech Mono"),
        xaxis=dict(showgrid=False,color="#3a6a9a",tickangle=-30,tickfont=dict(size=10)),
        yaxis=dict(showgrid=True,gridcolor="#0d2240",color="#3a6a9a",range=[0,1.1]),
        margin=dict(t=40,b=80,l=20,r=20), height=380)
    st.plotly_chart(fig_top, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — USER RISK TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">All Users — Risk Assessment</p>', unsafe_allow_html=True)
    col_s, col_f, col_sort = st.columns([2,1,1])
    with col_s:
        search = st.text_input("Search user", placeholder="e.g. DTAA/IRC0991 or DEMO/")
    with col_f:
        level_filter = st.selectbox("Risk Level", ["ALL","HIGH","MEDIUM","LOW"])
    with col_sort:
        sort_by = st.selectbox("Sort by", ["Risk Score","UEBA Score","Rule Score"])

    display_df = risk_df.copy()
    if search:
        display_df = display_df[display_df["user"].str.contains(search, case=False)]
    if level_filter != "ALL":
        display_df = display_df[display_df["risk_level"] == level_filter]
    sort_col = {"Risk Score":"final_risk_score","UEBA Score":"ueba_score_weighted","Rule Score":"rule_score"}[sort_by]
    display_df = display_df.sort_values(sort_col, ascending=False)

    table_df = display_df[["user","risk_level","final_risk_score","lstm_anomaly","ueba_anomaly","rule_anomaly","rules_violated"]].copy()
    table_df.columns = ["User","Risk Level","Final Score","LSTM","UEBA","Rules","Violations"]
    table_df["LSTM"]  = table_df["LSTM"].map({1:"YES",0:"no"})
    table_df["UEBA"]  = table_df["UEBA"].map({1:"YES",0:"no"})
    table_df["Rules"] = table_df["Rules"].map({1:"YES",0:"no"})
    table_df["Final Score"] = table_df["Final Score"].round(4)

    st.dataframe(table_df, use_container_width=True, height=600,
        column_config={
            "Risk Level" : st.column_config.TextColumn(width="small"),
            "Final Score": st.column_config.NumberColumn(format="%.4f"),
            "LSTM"       : st.column_config.TextColumn(width="small"),
            "UEBA"       : st.column_config.TextColumn(width="small"),
            "Rules"      : st.column_config.TextColumn(width="small"),
            "Violations" : st.column_config.NumberColumn(width="small"),
        }, hide_index=True)
    st.caption(f"Showing {len(display_df)} of {len(risk_df)} users")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — USER INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">User Investigation Panel</p>', unsafe_allow_html=True)
    all_users    = risk_df["user"].tolist()
    selected_user = st.selectbox("Select user to investigate", all_users,
        format_func=lambda u: f"{u}  —  {risk_df[risk_df['user']==u]['risk_level'].values[0]}  ({risk_df[risk_df['user']==u]['final_risk_score'].values[0]:.4f})")

    if selected_user:
        row   = risk_df[risk_df["user"] == selected_user].iloc[0]
        level = row["risk_level"]
        level_color = {"HIGH":"#ff4444","MEDIUM":"#ffa500","LOW":"#00cc66"}[level]
        is_demo = str(selected_user).startswith("DEMO/")

        demo_badge = '<span class="demo-injected">  [DEMO USER — INJECTED]</span>' if is_demo else ""
        st.markdown(f"""
        <div style="background:#0d1e30; border:1px solid {level_color}; border-radius:4px; padding:20px 24px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <p style="font-family:'Share Tech Mono',monospace; font-size:20px; color:{level_color}; margin:0;">
                        {selected_user}{demo_badge}</p>
                    <p style="color:#5a7a9a; margin:4px 0 0 0; font-size:13px; letter-spacing:1px;">INSIDER THREAT INVESTIGATION REPORT</p>
                </div>
                <div style="text-align:right;">
                    <p style="font-family:'Share Tech Mono',monospace; font-size:32px; color:{level_color}; margin:0;">{row['final_risk_score']:.4f}</p>
                    <p style="color:{level_color}; margin:0; font-size:14px; letter-spacing:3px; font-weight:700;">{level} RISK</p>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        def score_gauge(value, title, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=value,
                number=dict(font=dict(family="Share Tech Mono",color=color,size=28)),
                title=dict(text=title, font=dict(family="Share Tech Mono",color="#5a7a9a",size=12)),
                gauge=dict(axis=dict(range=[0,1],tickcolor="#1a3a5c",tickfont=dict(color="#3a6a9a",size=10)),
                    bar=dict(color=color), bgcolor="#0a1628", bordercolor="#1a3a5c",
                    steps=[dict(range=[0,0.3],color="#0a1e0a"),dict(range=[0.3,0.6],color="#1a1a0a"),dict(range=[0.6,1],color="#1a0a0a")],
                    threshold=dict(line=dict(color=color,width=2),value=value))
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(t=40,b=10,l=30,r=30))
            return fig

        with col_a:
            st.plotly_chart(score_gauge(row["lstm_score_norm"],"LSTM SCORE","#4a9eff"), use_container_width=True)
            flagged = "FLAGGED" if row["lstm_anomaly"] else "CLEAR"
            col = "#ff4444" if row["lstm_anomaly"] else "#00cc66"
            st.markdown(f"<p style='text-align:center;color:{col};font-family:Share Tech Mono;letter-spacing:2px;font-size:12px;'>{flagged}</p>", unsafe_allow_html=True)
        with col_b:
            st.plotly_chart(score_gauge(row["ueba_score_norm"],"UEBA SCORE","#a855f7"), use_container_width=True)
            flagged = "FLAGGED" if row["ueba_anomaly"] else "CLEAR"
            col = "#ff4444" if row["ueba_anomaly"] else "#00cc66"
            st.markdown(f"<p style='text-align:center;color:{col};font-family:Share Tech Mono;letter-spacing:2px;font-size:12px;'>{flagged}</p>", unsafe_allow_html=True)
        with col_c:
            st.plotly_chart(score_gauge(row["rule_score_norm"],"RULE SCORE","#ffa500"), use_container_width=True)
            flagged = "FLAGGED" if row["rule_anomaly"] else "CLEAR"
            col = "#ff4444" if row["rule_anomaly"] else "#00cc66"
            st.markdown(f"<p style='text-align:center;color:{col};font-family:Share Tech Mono;letter-spacing:2px;font-size:12px;'>{flagged}</p>", unsafe_allow_html=True)

        st.markdown('<p class="section-header">Activity Profile</p>', unsafe_allow_html=True)
        if data["features"] is not None:
            feat = data["features"][data["features"]["user"] == selected_user]
            if not feat.empty:
                feat_row = feat.iloc[0]
                f1, f2, f3, f4 = st.columns(4)
                for col_w, label, val, color in [
                    (f1,"LOGONS",feat_row.get("logon_count",0),"#4a9eff"),
                    (f2,"AFTER-HRS LOGONS",feat_row.get("after_hours_logon_count",0),"#ffa500"),
                    (f3,"DEVICE CONNECTS",feat_row.get("device_connect_count",0),"#a855f7"),
                    (f4,"HTTP VISITS",feat_row.get("http_count",0),"#00cc66")]:
                    with col_w:
                        st.markdown(f"""<div class="metric-card" style="text-align:center;padding:14px;">
                            <p style="font-family:Share Tech Mono;font-size:26px;color:{color};margin:0;">{int(val):,}</p>
                            <p style="font-size:11px;color:#5a7a9a;letter-spacing:2px;margin:6px 0 0 0;text-transform:uppercase;">{label}</p>
                        </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">Rule Violation Details</p>', unsafe_allow_html=True)
        explanation = str(row.get("explanation","No rules violated"))
        if explanation != "No rules violated" and explanation != "nan":
            for item in explanation.split(" || "):
                sev_class = "explanation-item"
                st.markdown(f'<div class="{sev_class}">{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="explanation-item" style="border-left-color:#00cc66;background:rgba(0,204,102,0.05);">No rule violations detected for this user.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SYSTEM ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_left, col_right = st.columns([1,1])

    with col_left:
        st.markdown('<p class="section-header">Model Performance Metrics</p>', unsafe_allow_html=True)
        metrics_data  = {"Accuracy":accuracy_val,"Precision":precision_val,"Recall":recall_val,"F1 Score":f1_val}
        metric_colors = ["#00cc66","#4a9eff","#ffa500","#a855f7"]
        fig_metrics   = go.Figure()
        for i,(name,val) in enumerate(metrics_data.items()):
            fig_metrics.add_trace(go.Bar(name=name, x=[name], y=[val], marker_color=metric_colors[i],
                text=[f"{val:.1f}%"], textposition="outside",
                textfont=dict(family="Share Tech Mono",color="white",size=14), showlegend=False))
        fig_metrics.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d6e5",family="Share Tech Mono"),
            xaxis=dict(showgrid=False,color="#3a6a9a"), yaxis=dict(showgrid=True,gridcolor="#0d2240",color="#3a6a9a",range=[0,115]),
            margin=dict(t=40,b=20,l=20,r=20), height=300)
        st.plotly_chart(fig_metrics, use_container_width=True)

        if acc_df is not None:
            st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Predicted Normal","Predicted Anomaly"], y=["Actual Normal","Actual Anomaly"],
                colorscale=[[0,"#0a1628"],[1,"#4a9eff"]], text=cm, texttemplate="%{text}",
                textfont=dict(family="Share Tech Mono",size=20,color="white"), showscale=False))
            fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8d6e5",family="Share Tech Mono",size=12),
                xaxis=dict(color="#3a6a9a"), yaxis=dict(color="#3a6a9a"),
                margin=dict(t=20,b=20,l=20,r=20), height=280)
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Network Behavior Graph</p>', unsafe_allow_html=True)

        graph_mode = st.radio(
            "Graph view",
            ["Simplified — Top Suspicious Users Only", "Full Interactive Network"],
            horizontal=True,
            key="graph_mode_radio"
        )

        # ── Load raw data for graph building ──────────────────────────────
        @st.cache_data
        def build_graph_data():
            import os as _os
            logon_path  = _os.path.join(BASE_DIR, "dataset", "logon.csv")
            device_path = _os.path.join(BASE_DIR, "dataset", "device.csv")
            http_path   = _os.path.join(BASE_DIR, "dataset", "http.csv")
            ueba_path   = _os.path.join(BASE_DIR, "dataset", "processed", "ueba_scores.csv")

            if not all(_os.path.exists(p) for p in [logon_path, device_path, http_path, ueba_path]):
                return None

            logon  = pd.read_csv(logon_path).head(5000)
            device = pd.read_csv(device_path).head(5000)
            http   = pd.read_csv(http_path, header=None,
                                  names=["id","date","user","pc","url"]).head(5000)
            ueba   = pd.read_csv(ueba_path)[["user","ueba_score_weighted","ueba_threshold"]]

            threshold        = ueba["ueba_threshold"].iloc[0]
            suspicious_users = set(ueba[ueba["ueba_score_weighted"] > threshold]["user"])

            edges = []
            for _, r in logon.iterrows():
                edges.append({"src": r["user"], "dst": r["pc"],
                               "type": "logon", "src_user": True})
            for _, r in device.iterrows():
                edges.append({"src": r["user"], "dst": r["pc"],
                               "type": "device", "src_user": True})
            for _, r in http.iterrows():
                edges.append({"src": r["user"], "dst": r["url"],
                               "type": "http", "src_user": True})

            return {
                "edges"           : edges,
                "suspicious_users": suspicious_users,
                "ueba"            : ueba,
                "threshold"       : threshold,
            }

        gdata = build_graph_data()

        if gdata is None:
            st.info("Raw dataset files not found. Run the pipeline first.")

        elif "Simplified" in graph_mode:
            # ── OPTION 2: Simplified ego network ──────────────────────────
            susp_users = list(gdata["suspicious_users"])[:12]
            edges      = gdata["edges"]

            # Build ego network: suspicious users + their direct connections
            ego_edges = [e for e in edges if e["src"] in susp_users]

            # Collect all nodes
            nodes      = {}
            for u in susp_users:
                ueba_row = gdata["ueba"][gdata["ueba"]["user"] == u]
                score    = float(ueba_row["ueba_score_weighted"].values[0]) if not ueba_row.empty else 0
                nodes[u] = {"type": "suspicious_user", "score": score}

            dst_counts = {}
            for e in ego_edges:
                dst_counts[e["dst"]] = dst_counts.get(e["dst"], 0) + 1

            # Only keep destinations connected to 2+ suspicious users (shared resources)
            shared_dsts = {dst for dst, cnt in dst_counts.items() if cnt >= 2}
            # Also keep top 5 per user
            per_user_dsts = {}
            for e in ego_edges:
                per_user_dsts.setdefault(e["src"], []).append(e["dst"])

            included_dsts = set(shared_dsts)
            for u, dsts in per_user_dsts.items():
                included_dsts.update(dsts[:4])

            for e in ego_edges:
                dst = e["dst"]
                if dst not in included_dsts:
                    continue
                if dst not in nodes:
                    if "PC" in str(dst):
                        nodes[dst] = {"type": "pc"}
                    else:
                        nodes[dst] = {"type": "website"}

            # Layout: circular for suspicious users, radial for connections
            import math
            node_list = list(nodes.keys())
            pos = {}
            susp_list = [n for n in node_list if nodes[n]["type"] == "suspicious_user"]
            other_list = [n for n in node_list if nodes[n]["type"] != "suspicious_user"]

            for i, n in enumerate(susp_list):
                angle = 2 * math.pi * i / max(len(susp_list), 1)
                pos[n] = (math.cos(angle) * 1.5, math.sin(angle) * 1.5)

            for i, n in enumerate(other_list):
                angle  = 2 * math.pi * i / max(len(other_list), 1)
                radius = 3.0 + (0.5 if "PC" in str(n) else 0)
                pos[n] = (math.cos(angle) * radius, math.sin(angle) * radius)

            # Build plotly figure
            edge_x, edge_y = [], []
            for e in ego_edges:
                src, dst = e["src"], e["dst"]
                if src in pos and dst in pos and dst in included_dsts:
                    edge_x += [pos[src][0], pos[dst][0], None]
                    edge_y += [pos[src][1], pos[dst][1], None]

            fig_g = go.Figure()

            # Edges
            fig_g.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.8, color="rgba(74,158,255,0.25)"),
                hoverinfo="none", showlegend=False
            ))

            # Node groups
            type_cfg = {
                "suspicious_user": {"color": "#ff4444", "size": 22, "symbol": "circle",
                                     "name": "Suspicious User"},
                "pc"             : {"color": "#888888", "size": 12, "symbol": "square",
                                     "name": "PC / Machine"},
                "website"        : {"color": "#00cc66", "size": 10, "symbol": "diamond",
                                     "name": "Website"},
            }

            for ntype, cfg in type_cfg.items():
                type_nodes = [n for n in node_list if nodes[n]["type"] == ntype]
                if not type_nodes:
                    continue
                nx_ = [pos[n][0] for n in type_nodes]
                ny_ = [pos[n][1] for n in type_nodes]
                labels = []
                for n in type_nodes:
                    if ntype == "suspicious_user":
                        sc = nodes[n].get("score", 0)
                        labels.append(f"<b>{n}</b><br>UEBA Score: {sc:.2f}<br>Type: Suspicious User")
                    elif ntype == "pc":
                        labels.append(f"<b>{n}</b><br>Type: PC / Machine<br>Accessed by {dst_counts.get(n,0)} suspicious user(s)")
                    else:
                        short = str(n)[:40]
                        labels.append(f"<b>{short}</b><br>Type: Website<br>Visited by {dst_counts.get(n,0)} suspicious user(s)")

                text_labels = [n if nodes[n]["type"] == "suspicious_user" else "" for n in type_nodes]

                fig_g.add_trace(go.Scatter(
                    x=nx_, y=ny_, mode="markers+text",
                    marker=dict(size=cfg["size"], color=cfg["color"],
                                symbol=cfg["symbol"],
                                line=dict(width=1.5, color="rgba(255,255,255,0.3)")),
                    text=text_labels,
                    textposition="top center",
                    textfont=dict(size=8, color="#c8d6e5", family="Share Tech Mono"),
                    hovertext=labels,
                    hoverinfo="text",
                    name=cfg["name"]
                ))

            fig_g.update_layout(
                paper_bgcolor="#0a1628",
                plot_bgcolor="#0a1628",
                showlegend=True,
                legend=dict(font=dict(color="#c8d6e5", family="Share Tech Mono", size=10),
                            bgcolor="rgba(13,30,48,0.8)", bordercolor="#1a3a5c", borderwidth=1),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(t=20, b=20, l=10, r=10),
                height=500,
                hovermode="closest",
                annotations=[dict(
                    text="Hover over any node for details. Red = Suspicious Users, Gray = PCs, Green = Websites",
                    xref="paper", yref="paper", x=0.5, y=-0.02,
                    showarrow=False, font=dict(size=9, color="#5a7a9a", family="Share Tech Mono"),
                    align="center"
                )]
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Legend explanation
            st.markdown("""
            <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;padding:12px 16px;font-size:12px;">
                <b style="color:#4a9eff;font-family:Share Tech Mono;letter-spacing:1px;">HOW TO READ THIS GRAPH</b><br><br>
                <span style="color:#ff4444;">&#9679;</span> <b>Red circles</b> = Suspicious users flagged by UEBA analysis<br>
                <span style="color:#888888;">&#9632;</span> <b>Gray squares</b> = PCs / machines these users accessed<br>
                <span style="color:#00cc66;">&#9670;</span> <b>Green diamonds</b> = Websites these users visited<br><br>
                <span style="color:#c8d6e5;">Lines connecting nodes show access relationships.
                When multiple suspicious users connect to the same PC (shared gray node),
                it may indicate lateral movement or coordinated activity.</span>
            </div>""", unsafe_allow_html=True)

        else:
            # ── OPTION 1: Full interactive Plotly network ──────────────────
            edges      = gdata["edges"]
            susp_users = gdata["suspicious_users"]
            ueba_df_g  = gdata["ueba"]

            # Collect unique nodes
            all_nodes = {}
            for e in edges:
                all_nodes[e["src"]] = "suspicious_user" if e["src"] in susp_users else "normal_user"
                dst = e["dst"]
                if dst not in all_nodes:
                    all_nodes[dst] = "pc" if "PC" in str(dst) else "website"

            node_list = list(all_nodes.keys())

            # Simple force-like layout using degree
            import math, random
            random.seed(42)
            degree = {}
            for e in edges:
                degree[e["src"]] = degree.get(e["src"], 0) + 1
                degree[e["dst"]] = degree.get(e["dst"], 0) + 1

            pos = {}
            for i, n in enumerate(node_list):
                angle  = 2 * math.pi * i / len(node_list)
                r_base = 1.0 / (degree.get(n, 1) ** 0.1)
                noise  = random.uniform(0.8, 1.2)
                pos[n] = (math.cos(angle) * r_base * noise * 10,
                           math.sin(angle) * r_base * noise * 10)

            edge_x, edge_y = [], []
            for e in edges[:3000]:
                src, dst = e["src"], e["dst"]
                if src in pos and dst in pos:
                    edge_x += [pos[src][0], pos[dst][0], None]
                    edge_y += [pos[src][1], pos[dst][1], None]

            fig_full = go.Figure()
            fig_full.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.4, color="rgba(100,130,160,0.2)"),
                hoverinfo="none", showlegend=False
            ))

            type_cfg_full = {
                "suspicious_user": {"color":"#ff4444","size":14,"name":"Suspicious User"},
                "normal_user"    : {"color":"#4a9eff","size":6, "name":"Normal User"},
                "pc"             : {"color":"#888888","size":5, "name":"PC / Machine"},
                "website"        : {"color":"#00cc66","size":4, "name":"Website"},
            }

            for ntype, cfg in type_cfg_full.items():
                type_nodes = [n for n in node_list if all_nodes[n] == ntype]
                if not type_nodes:
                    continue
                nx_ = [pos[n][0] for n in type_nodes]
                ny_ = [pos[n][1] for n in type_nodes]
                labels = []
                for n in type_nodes:
                    deg  = degree.get(n, 0)
                    urow = ueba_df_g[ueba_df_g["user"]==n] if ntype in ("suspicious_user","normal_user") else pd.DataFrame()
                    sc   = float(urow["ueba_score_weighted"].values[0]) if not urow.empty else 0
                    if ntype in ("suspicious_user","normal_user"):
                        labels.append(f"<b>{n}</b><br>Type: {ntype.replace('_',' ').title()}<br>UEBA Score: {sc:.2f}<br>Connections: {deg}")
                    else:
                        labels.append(f"<b>{str(n)[:50]}</b><br>Type: {ntype.replace('_',' ').title()}<br>Connections: {deg}")

                fig_full.add_trace(go.Scatter(
                    x=nx_, y=ny_, mode="markers",
                    marker=dict(size=cfg["size"], color=cfg["color"],
                                line=dict(width=0.5 if ntype!="suspicious_user" else 1.5,
                                          color="white" if ntype=="suspicious_user" else "rgba(0,0,0,0)")),
                    hovertext=labels, hoverinfo="text",
                    name=cfg["name"]
                ))

            fig_full.update_layout(
                paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
                showlegend=True,
                legend=dict(font=dict(color="#c8d6e5",family="Share Tech Mono",size=10),
                            bgcolor="rgba(13,30,48,0.8)", bordercolor="#1a3a5c", borderwidth=1),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(t=20,b=30,l=10,r=10),
                height=520,
                hovermode="closest",
                annotations=[dict(
                    text="Zoom and pan to explore. Hover over any node for details.",
                    xref="paper", yref="paper", x=0.5, y=-0.04,
                    showarrow=False, font=dict(size=9,color="#5a7a9a",family="Share Tech Mono"),
                    align="center"
                )]
            )
            st.plotly_chart(fig_full, use_container_width=True)

        # ── THREE FOCUSED SECURITY TABLES ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        @st.cache_data
        def build_security_tables():
            import os as _os
            logon_path  = _os.path.join(BASE_DIR, "dataset", "logon.csv")
            device_path = _os.path.join(BASE_DIR, "dataset", "device.csv")
            http_path   = _os.path.join(BASE_DIR, "dataset", "http.csv")
            risk_path   = _os.path.join(BASE_DIR, "dataset", "processed", "final_risk_scores.csv")
            ueba_path   = _os.path.join(BASE_DIR, "dataset", "processed", "ueba_scores.csv")
            graph_path  = _os.path.join(BASE_DIR, "dataset", "processed", "graph_scores.csv")

            for p in [logon_path, risk_path, ueba_path]:
                if not _os.path.exists(p):
                    return None, None, None

            risk_df_t = pd.read_csv(risk_path)
            ueba_df_t = pd.read_csv(ueba_path)[["user","ueba_score_weighted","ueba_threshold"]]
            threshold = ueba_df_t["ueba_threshold"].iloc[0]
            susp_users = set(ueba_df_t[ueba_df_t["ueba_score_weighted"] > threshold]["user"])

            # ── Table 1: Most connected suspicious users ──────────────────
            graph_df_t = None
            if _os.path.exists(graph_path):
                gdf       = pd.read_csv(graph_path)
                # Keep only user nodes (contain DTAA or DEMO)
                user_nodes = gdf[gdf["node"].str.contains("DTAA|DEMO", na=False)].copy()
                susp_nodes = user_nodes[user_nodes["node"].isin(susp_users)].copy()
                susp_nodes = susp_nodes.merge(
                    risk_df_t[["user","risk_level","final_risk_score"]],
                    left_on="node", right_on="user", how="left"
                )
                susp_nodes = susp_nodes.sort_values("centrality", ascending=False).head(10)
                susp_nodes["centrality"]       = susp_nodes["centrality"].round(4)
                susp_nodes["final_risk_score"] = susp_nodes["final_risk_score"].round(4)
                graph_df_t = susp_nodes[["node","centrality","risk_level","final_risk_score"]].copy()
                graph_df_t.columns = ["User","Network Centrality","Risk Level","Unified Score"]

            # ── Table 2: Shared PCs (lateral movement) ───────────────────
            shared_pc_df = None
            if _os.path.exists(logon_path):
                logon  = pd.read_csv(logon_path)
                logon  = logon[logon["user"].isin(susp_users)]
                if not logon.empty:
                    pc_users = logon.groupby("pc")["user"].apply(
                        lambda x: list(x.unique())
                    ).reset_index()
                    pc_users["suspicious_user_count"] = pc_users["user"].apply(len)
                    pc_users = pc_users[pc_users["suspicious_user_count"] >= 2]
                    pc_users = pc_users.sort_values("suspicious_user_count", ascending=False).head(10)
                    pc_users["users_list"] = pc_users["user"].apply(
                        lambda x: ", ".join(x[:3]) + (f"  +{len(x)-3} more" if len(x) > 3 else "")
                    )
                    shared_pc_df = pc_users[["pc","suspicious_user_count","users_list"]].copy()
                    shared_pc_df.columns = ["PC / Machine","Suspicious Users","Who Accessed It"]

            # ── Table 3: Unusual websites visited by suspicious users ─────
            unusual_url_df = None
            common_domains = {
                "google.com","facebook.com","yahoo.com","youtube.com","wikipedia.org",
                "amazon.com","twitter.com","linkedin.com","microsoft.com","apple.com",
                "instagram.com","reddit.com","netflix.com","espn.go.com","craigslist.org",
                "blogspot.com","wordpress.com","tumblr.com","ebay.com","paypal.com",
                "bing.com","msn.com","outlook.com","gmail.com","hotmail.com",
                "cnn.com","bbc.com","nytimes.com","foxnews.com","weather.com",
                "imdb.com","pinterest.com","snapchat.com","tiktok.com","twitch.tv",
                "comcast.net","verizon.com","att.com","target.com","walmart.com",
                "whitepages.com","ticketmaster.com","espn.com","mlb.com","nfl.com",
            }

            if _os.path.exists(http_path):
                http = pd.read_csv(http_path, header=None,
                                    names=["id","date","user","pc","url"])
                http_susp = http[http["user"].isin(susp_users)].copy()
                if not http_susp.empty:
                    def extract_domain(url):
                        url = str(url).replace("http://","").replace("https://","")
                        return url.split("/")[0].lower()

                    http_susp["domain"] = http_susp["url"].apply(extract_domain)
                    unusual = http_susp[~http_susp["domain"].isin(common_domains)]
                    if not unusual.empty:
                        url_counts = unusual.groupby("domain").agg(
                            visit_count=("url","count"),
                            unique_users=("user","nunique")
                        ).reset_index()
                        url_counts = url_counts[url_counts["unique_users"] >= 2]
                        url_counts = url_counts.sort_values("unique_users", ascending=False).head(10)
                        url_counts.columns = ["Domain","Total Visits","Suspicious Users Visiting"]
                        unusual_url_df = url_counts

            return graph_df_t, shared_pc_df, unusual_url_df

        t1, t2, t3 = build_security_tables()

        # Table 1
        st.markdown('<p class="section-header">Most Connected Suspicious Users</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;
                    padding:10px 14px;margin-bottom:8px;font-size:12px;color:#7a8a9a;">
        Network centrality measures how connected a user is across the entire network.
        A high centrality suspicious user has accessed an unusually large number of
        machines and websites — a strong indicator of lateral movement or data collection.
        </div>""", unsafe_allow_html=True)
        if t1 is not None and not t1.empty:
            st.dataframe(t1, use_container_width=True, hide_index=True, height=280,
                column_config={
                    "Network Centrality": st.column_config.NumberColumn(format="%.4f"),
                    "Unified Score"     : st.column_config.NumberColumn(format="%.4f"),
                })
        else:
            st.info("No graph centrality data available. Run the pipeline first.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Table 2
        st.markdown('<p class="section-header">Shared Machines — Lateral Movement Indicator</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;
                    padding:10px 14px;margin-bottom:8px;font-size:12px;color:#7a8a9a;">
        These machines were accessed by multiple suspicious users. When several flagged
        users connect to the same machine, it may be a target system or a pivot point
        used for lateral movement within the network.
        </div>""", unsafe_allow_html=True)
        if t2 is not None and not t2.empty:
            st.dataframe(t2, use_container_width=True, hide_index=True, height=280)
        else:
            st.info("No shared PC data found. This may mean suspicious users accessed distinct machines.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Table 3
        st.markdown('<p class="section-header">Unusual Websites — Visited by Multiple Suspicious Users</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;
                    padding:10px 14px;margin-bottom:8px;font-size:12px;color:#7a8a9a;">
        Common sites like Google and Facebook are filtered out. These are non-standard
        domains visited by two or more suspicious users — potential exfiltration
        destinations, cloud storage, or external communication channels worth investigating.
        </div>""", unsafe_allow_html=True)
        if t3 is not None and not t3.empty:
            st.dataframe(t3, use_container_width=True, hide_index=True, height=280)
        else:
            st.info("No unusual websites found among suspicious users.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EVIDENCE EXPORTER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Evidence Package Exporter</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1e30; border:1px solid #1a3a5c; border-radius:4px; padding:16px 20px; margin-bottom:20px;">
        <p style="color:#c8d6e5; margin:0; font-size:14px;">
        Generate a complete forensic evidence package for any flagged user.
        The ZIP file contains a forensic PDF report, suspicious event timeline,
        all relevant CSVs, and a plain-text summary ready to hand to HR or Legal.
        </p>
    </div>""", unsafe_allow_html=True)

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        export_user = st.selectbox(
            "Select user to export",
            risk_df["user"].tolist(),
            format_func=lambda u: (
                f"{u}  [{risk_df[risk_df['user']==u]['risk_level'].values[0]}]"
                f"  score: {risk_df[risk_df['user']==u]['final_risk_score'].values[0]:.4f}"
            ),
            key="export_user_select"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Package", type="primary", use_container_width=True)

    if generate_btn and export_user:
        with st.spinner(f"Generating evidence package for {export_user}..."):
            try:
                from evidence_exporter import generate_package
                zip_path = generate_package(BASE_DIR, export_user)

                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()

                safe_name = export_user.replace("/", "_")
                fname     = f"evidence_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

                st.success(f"Package generated successfully for {export_user}")
                st.download_button(
                    label="Download Evidence Package (ZIP)",
                    data=zip_bytes,
                    file_name=fname,
                    mime="application/zip",
                    use_container_width=True
                )

                # Preview the package contents
                st.markdown('<p class="section-header">Package Contents Preview</p>', unsafe_allow_html=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    col_f1, col_f2 = st.columns(2)
                    for i, name in enumerate(zf.namelist()):
                        info = zf.getinfo(name)
                        size_kb = info.file_size / 1024
                        icon = "pdf" if name.endswith(".pdf") else ("csv" if name.endswith(".csv") else "txt")
                        with (col_f1 if i % 2 == 0 else col_f2):
                            st.markdown(f"""
                            <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;
                                        padding:10px 14px;margin-bottom:8px;">
                                <span style="color:#4a9eff;font-family:Share Tech Mono;font-size:12px;">[{icon.upper()}]</span>
                                <span style="color:#c8d6e5;font-size:13px;margin-left:8px;">{name}</span>
                                <span style="color:#5a7a9a;font-size:11px;float:right;">{size_kb:.1f} KB</span>
                            </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Failed to generate package: {e}")

    # Show existing packages
    pkg_dir = os.path.join(BASE_DIR, "evidence_packages")
    if os.path.exists(pkg_dir):
        existing = sorted(
            [f for f in os.listdir(pkg_dir) if f.endswith(".zip")],
            reverse=True
        )
        if existing:
            st.markdown('<p class="section-header">Previously Generated Packages</p>', unsafe_allow_html=True)
            for pkg in existing[:5]:
                size_mb = os.path.getsize(os.path.join(pkg_dir, pkg)) / (1024*1024)
                mtime   = datetime.fromtimestamp(os.path.getmtime(os.path.join(pkg_dir, pkg)))
                col_n, col_s, col_t, col_d = st.columns([4,1,2,1])
                with col_n:
                    st.markdown(f'<span style="color:#c8d6e5;font-family:Share Tech Mono;font-size:11px;">{pkg}</span>', unsafe_allow_html=True)
                with col_s:
                    st.markdown(f'<span style="color:#5a7a9a;font-size:11px;">{size_mb:.2f} MB</span>', unsafe_allow_html=True)
                with col_t:
                    st.markdown(f'<span style="color:#5a7a9a;font-size:11px;">{mtime.strftime("%Y-%m-%d %H:%M")}</span>', unsafe_allow_html=True)
                with col_d:
                    with open(os.path.join(pkg_dir, pkg), "rb") as f:
                        st.download_button("Download", f.read(), file_name=pkg,
                            mime="application/zip", key=f"dl_{pkg}", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LIVE DEMO MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-header">Live Demo Mode — Threat Scenario Injection</p>', unsafe_allow_html=True)

    # Check if demo users currently injected
    demo_users_present = any(
        u in risk_df["user"].values for u in ["DEMO/EVL0001","DEMO/EVL0002","DEMO/EVL0003"]
    )

    if demo_users_present:
        st.markdown("""
        <div style="background:rgba(255,68,68,0.1);border:1px solid #ff4444;border-radius:4px;padding:12px 16px;margin-bottom:16px;">
            <span style="color:#ff4444;font-family:Share Tech Mono;font-size:12px;letter-spacing:2px;">
            DEMO USERS ACTIVE — System is running with injected threat scenarios
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;padding:16px 20px;margin-bottom:20px;">
            <p style="color:#c8d6e5;margin:0;font-size:14px;">
            Inject synthetic malicious users into the live system to demonstrate
            end-to-end threat detection during your presentation. The dashboard
            updates in real time and the risk scores reflect the injected users.
            Choose a scenario, inject it, and watch the system respond.
            </p>
        </div>""", unsafe_allow_html=True)

    # Scenario cards
    st.markdown('<p class="section-header">Available Scenarios</p>', unsafe_allow_html=True)

    try:
        from demo_mode import get_scenario_info, inject_scenario, clear_demo_users
        scenarios = get_scenario_info()
    except Exception as e:
        st.error(f"Could not load demo_mode module: {e}")
        scenarios = {}

    if scenarios:
        sc_cols = st.columns(3)
        for i, (key, sc) in enumerate(scenarios.items()):
            with sc_cols[i]:
                demo_user = sc["user"]
                is_active = demo_user in risk_df["user"].values
                if is_active:
                    user_row   = risk_df[risk_df["user"] == demo_user].iloc[0]
                    level      = user_row["risk_level"]
                    score      = user_row["final_risk_score"]
                    card_class = f"demo-card-{level.lower()}"
                    badge = f'<span style="color:{"#ff4444" if level=="HIGH" else "#ffa500"};font-family:Share Tech Mono;font-size:11px;letter-spacing:1px;">ACTIVE — {level} ({score:.4f})</span>'
                else:
                    card_class = "detail-card"
                    badge = '<span style="color:#5a7a9a;font-size:11px;">Not injected</span>'

                st.markdown(f"""
                <div class="{card_class}">
                    <p style="font-family:Share Tech Mono;color:#4a9eff;font-size:13px;margin:0 0 4px 0;">
                        SCENARIO {key}</p>
                    <p style="color:#c8d6e5;font-size:15px;font-weight:700;margin:0 0 8px 0;">
                        {sc['label']}</p>
                    <p style="color:#7a8a9a;font-size:12px;margin:0 0 8px 0;">
                        {sc['description']}</p>
                    <p style="color:#4a9eff;font-size:11px;margin:0 0 8px 0;">
                        User: {sc['user']}</p>
                    {badge}
                </div>""", unsafe_allow_html=True)

                if is_active:
                    st.markdown(f"""
                    <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;padding:8px 12px;margin-top:8px;">
                        <p style="font-family:Share Tech Mono;font-size:10px;color:#5a7a9a;margin:0 0 4px 0;">DETECTION FLAGS</p>
                        <p style="font-size:12px;margin:0;">
                            LSTM: <b style="color:{'#ff4444' if user_row['lstm_anomaly'] else '#00cc66'}">{'FLAGGED' if user_row['lstm_anomaly'] else 'clear'}</b>
                            &nbsp;|&nbsp;
                            UEBA: <b style="color:{'#ff4444' if user_row['ueba_anomaly'] else '#00cc66'}">{'FLAGGED' if user_row['ueba_anomaly'] else 'clear'}</b>
                            &nbsp;|&nbsp;
                            Rules: <b style="color:{'#ff4444' if user_row['rule_anomaly'] else '#00cc66'}">{'FLAGGED' if user_row['rule_anomaly'] else 'clear'}</b>
                        </p>
                    </div>""", unsafe_allow_html=True)

        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Inject / Clear</p>', unsafe_allow_html=True)

        col_sc, col_inj, col_clr = st.columns([2,1,1])
        with col_sc:
            chosen = st.selectbox("Choose scenario to inject",
                ["ALL — Inject all three scenarios", "A — Data Exfiltration",
                 "B — Credential Theft", "C — Negligent Insider"],
                key="demo_scenario_select")
        with col_inj:
            st.markdown("<br>", unsafe_allow_html=True)
            inject_btn = st.button("Inject Scenario", type="primary", use_container_width=True)
        with col_clr:
            st.markdown("<br>", unsafe_allow_html=True)
            clear_btn  = st.button("Clear Demo Users", use_container_width=True)

        if inject_btn:
            scenario_key = chosen.split(" ")[0]
            with st.spinner(f"Injecting scenario {scenario_key}..."):
                try:
                    results = inject_scenario(BASE_DIR, scenario_key)
                    st.cache_data.clear()
                    st.success(f"Injected {len(results)} demo user(s). Refresh the page to see updated dashboard.")
                    for k, r in results.items():
                        level_col = "#ff4444" if r["risk_level"]=="HIGH" else ("#ffa500" if r["risk_level"]=="MEDIUM" else "#00cc66")
                        st.markdown(f"""
                        <div style="background:#0d1e30;border:1px solid {level_col};border-radius:4px;padding:12px 16px;margin-top:8px;">
                            <b style="color:{level_col};font-family:Share Tech Mono;">{r['user']} — {r['risk_level']} ({r['final_risk_score']:.4f})</b><br>
                            <span style="font-size:12px;color:#c8d6e5;">
                                LSTM: {'FLAGGED' if r['lstm_anomaly'] else 'clear'} &nbsp;|&nbsp;
                                UEBA: {'FLAGGED' if r['ueba_anomaly'] else 'clear'} &nbsp;|&nbsp;
                                Rules: {'FLAGGED' if r['rule_anomaly'] else 'clear'} ({r['rules_violated']} violations)
                            </span><br>
                            <span style="font-size:11px;color:#5a7a9a;">{r['expected']}</span>
                        </div>""", unsafe_allow_html=True)
                    st.info("Go to the Overview or User Risk Table tab to see the injected users highlighted.")
                except Exception as e:
                    st.error(f"Injection failed: {e}")

        if clear_btn:
            with st.spinner("Restoring original data..."):
                try:
                    from demo_mode import clear_demo_users
                    n = clear_demo_users(BASE_DIR)
                    st.cache_data.clear()
                    st.success(f"Demo users removed. {n} files restored to original state.")
                    st.info("Refresh the page to see the restored dashboard.")
                except Exception as e:
                    st.error(f"Failed to clear demo users: {e}")

        # Presentation tips
        st.markdown('<p class="section-header">Presentation Guide</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1e30;border:1px solid #1a3a5c;border-radius:4px;padding:16px 20px;">
            <p style="color:#4a9eff;font-family:Share Tech Mono;font-size:12px;letter-spacing:2px;margin:0 0 10px 0;">60-SECOND LIVE DEMO SCRIPT</p>
            <p style="color:#c8d6e5;font-size:13px;margin:0 0 6px 0;"><b style="color:#4a9eff;">Step 1</b> — Open this tab and explain the three scenario types to your examiner</p>
            <p style="color:#c8d6e5;font-size:13px;margin:0 0 6px 0;"><b style="color:#4a9eff;">Step 2</b> — Select "ALL" and click Inject Scenario</p>
            <p style="color:#c8d6e5;font-size:13px;margin:0 0 6px 0;"><b style="color:#4a9eff;">Step 3</b> — Switch to Overview tab and show the risk counts have increased</p>
            <p style="color:#c8d6e5;font-size:13px;margin:0 0 6px 0;"><b style="color:#4a9eff;">Step 4</b> — Go to User Risk Table, search "DEMO" to show all three injected users</p>
            <p style="color:#c8d6e5;font-size:13px;margin:0 0 6px 0;"><b style="color:#4a9eff;">Step 5</b> — Go to User Investigation, select DEMO/EVL0001, show the gauge meters and rule explanations</p>
            <p style="color:#c8d6e5;font-size:13px;margin:0 0 12px 0;"><b style="color:#4a9eff;">Step 6</b> — Go to Evidence Exporter and generate a package for DEMO/EVL0001 live</p>
            <p style="color:#c8d6e5;font-size:13px;margin:0;"><b style="color:#4a9eff;">Step 7</b> — Come back here and click Clear Demo Users to restore the system</p>
        </div>""", unsafe_allow_html=True)