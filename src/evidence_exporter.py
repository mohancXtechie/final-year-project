"""
Evidence Package Exporter
==========================
Generates a forensic evidence package for a flagged user containing:
  - forensic_report.pdf  : full investigation report with timeline
  - user_risk_profile.csv: complete risk scores and scores breakdown
  - suspicious_events.csv: all suspicious events from raw logs
  - detection_summary.txt: plain text summary for HR/Legal

Everything is packaged into a ZIP file ready to hand to HR or Legal.

Usage from dashboard or standalone:
    from src.evidence_exporter import generate_package
    zip_path = generate_package(base_dir, username)
"""

import os
import csv
import zipfile
import io
from datetime import datetime

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# ── Colour palette ──────────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#0d1e30")
C_BLUE   = colors.HexColor("#1a3a5c")
C_ACCENT = colors.HexColor("#2E75B6")
C_RED    = colors.HexColor("#c0392b")
C_ORANGE = colors.HexColor("#e67e22")
C_GREEN  = colors.HexColor("#27ae60")
C_GREY   = colors.HexColor("#7f8c8d")
C_LIGHT  = colors.HexColor("#ecf0f1")
C_WHITE  = colors.white
C_BLACK  = colors.black


# ── Styles ───────────────────────────────────────────────────────────────────
def _styles():
    return {
        "title": ParagraphStyle("title",
            fontName="Helvetica-Bold", fontSize=18,
            textColor=C_WHITE, alignment=TA_LEFT,
            leading=22, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle",
            fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#aaaaaa"), alignment=TA_LEFT,
            leading=14, spaceAfter=2),
        "section": ParagraphStyle("section",
            fontName="Helvetica-Bold", fontSize=12,
            textColor=C_ACCENT, alignment=TA_LEFT,
            leading=16, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=9,
            textColor=C_BLACK, alignment=TA_LEFT,
            leading=13, spaceAfter=4),
        "mono": ParagraphStyle("mono",
            fontName="Courier", fontSize=8,
            textColor=colors.HexColor("#333333"), alignment=TA_LEFT,
            leading=12, spaceAfter=2),
        "caption": ParagraphStyle("caption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=C_GREY, alignment=TA_CENTER,
            leading=10, spaceAfter=6),
        "label": ParagraphStyle("label",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=C_DARK, alignment=TA_LEFT, leading=12),
        "value": ParagraphStyle("value",
            fontName="Helvetica", fontSize=9,
            textColor=C_BLACK, alignment=TA_LEFT, leading=12),
        "timeline_time": ParagraphStyle("ttime",
            fontName="Courier-Bold", fontSize=8,
            textColor=C_ACCENT, alignment=TA_LEFT, leading=11),
        "timeline_event": ParagraphStyle("tevent",
            fontName="Courier", fontSize=8,
            textColor=C_BLACK, alignment=TA_LEFT, leading=11),
        "risk_high": ParagraphStyle("rhigh",
            fontName="Helvetica-Bold", fontSize=24,
            textColor=C_RED, alignment=TA_CENTER, leading=28),
        "risk_medium": ParagraphStyle("rmed",
            fontName="Helvetica-Bold", fontSize=24,
            textColor=C_ORANGE, alignment=TA_CENTER, leading=28),
        "risk_low": ParagraphStyle("rlow",
            fontName="Helvetica-Bold", fontSize=24,
            textColor=C_GREEN, alignment=TA_CENTER, leading=28),
    }


# ── Page canvas with header/footer ──────────────────────────────────────────
def _make_canvas_maker(username, generated_at):
    def on_page(canvas, doc):
        w, h = A4
        # Header bar
        canvas.setFillColor(C_DARK)
        canvas.rect(0, h - 2*cm, w, 2*cm, fill=1, stroke=0)
        canvas.setFillColor(C_ACCENT)
        canvas.rect(0, h - 2*cm, 0.4*cm, 2*cm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(C_WHITE)
        canvas.drawString(1*cm, h - 1.1*cm, "FORENSIC EVIDENCE REPORT")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 1*cm, h - 1.1*cm, f"User: {username}")
        # Footer
        canvas.setFillColor(C_LIGHT)
        canvas.rect(0, 0, w, 0.8*cm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_GREY)
        canvas.drawString(1*cm, 0.25*cm, f"Generated: {generated_at}  |  CONFIDENTIAL — FOR HR/LEGAL USE ONLY")
        canvas.drawRightString(w - 1*cm, 0.25*cm, f"Page {doc.page}")
    return on_page


# ── Load all data ────────────────────────────────────────────────────────────
def _load_data(base_dir, username):
    proc = os.path.join(base_dir, "dataset", "processed")

    risk_df = pd.read_csv(os.path.join(proc, "final_risk_scores.csv"))
    ueba_df = pd.read_csv(os.path.join(proc, "ueba_scores.csv"))
    feat_df = pd.read_csv(os.path.join(proc, "user_behavior_features.csv"))

    risk_row = risk_df[risk_df["user"] == username]
    ueba_row = ueba_df[ueba_df["user"] == username]
    feat_row = feat_df[feat_df["user"] == username]

    if risk_row.empty:
        raise ValueError(f"User {username} not found in risk scores.")

    return risk_row.iloc[0], ueba_row.iloc[0], feat_row.iloc[0]


# ── Load suspicious events timeline ─────────────────────────────────────────
def _load_events(base_dir, username):
    """Load raw events for the user and mark suspicious ones."""
    logon = pd.read_csv(os.path.join(base_dir, "dataset", "logon.csv"))
    device = pd.read_csv(os.path.join(base_dir, "dataset", "device.csv"))
    http = pd.read_csv(
        os.path.join(base_dir, "dataset", "http.csv"),
        header=None, names=["id", "date", "user", "pc", "url"]
    )

    logon["date"]  = pd.to_datetime(logon["date"])
    device["date"] = pd.to_datetime(device["date"])
    http["date"]   = pd.to_datetime(http["date"])

    def is_after_hours(dt):
        return (dt.dt.hour < 8) | (dt.dt.hour >= 18)

    # Filter to user
    u_logon  = logon[logon["user"] == username].copy()
    u_device = device[device["user"] == username].copy()
    u_http   = http[http["user"] == username].copy()

    events = []

    # Logon events
    for _, row in u_logon.iterrows():
        ah = row["date"].hour < 8 or row["date"].hour >= 18
        events.append({
            "date"     : row["date"],
            "type"     : row.get("activity", "Logon"),
            "detail"   : row.get("pc", ""),
            "after_hrs": ah,
            "source"   : "logon"
        })

    # Device Connect only
    for _, row in u_device[u_device["activity"] == "Connect"].iterrows():
        ah = row["date"].hour < 8 or row["date"].hour >= 18
        events.append({
            "date"     : row["date"],
            "type"     : "Device Connect",
            "detail"   : row.get("pc", ""),
            "after_hrs": ah,
            "source"   : "device"
        })

    # HTTP — only after hours for brevity
    ah_http = u_http[is_after_hours(u_http["date"])]
    for _, row in ah_http.head(50).iterrows():
        events.append({
            "date"     : row["date"],
            "type"     : "HTTP Visit",
            "detail"   : str(row.get("url", ""))[:60],
            "after_hrs": True,
            "source"   : "http"
        })

    events_df = pd.DataFrame(events).sort_values("date").reset_index(drop=True)
    return events_df


# ── PDF generator ────────────────────────────────────────────────────────────
def _build_pdf(base_dir, username, output_path):
    s = _styles()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    risk_row, ueba_row, feat_row = _load_data(base_dir, username)
    events_df = _load_events(base_dir, username)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=2.5*cm, bottomMargin=1.5*cm
    )

    story = []

    # ── COVER SECTION ────────────────────────────────────────────────────
    level = str(risk_row.get("risk_level", "UNKNOWN"))
    score = float(risk_row.get("final_risk_score", 0))
    level_color = {"HIGH": C_RED, "MEDIUM": C_ORANGE, "LOW": C_GREEN}.get(level, C_GREY)

    # Title block
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("INSIDER THREAT FORENSIC REPORT", s["title"]))
    story.append(Paragraph(f"Evidence Package — Confidential", s["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT, spaceAfter=10))

    # User identity + risk score side by side
    identity_data = [
        [Paragraph("<b>SUBJECT USER</b>", s["label"]),
         Paragraph(username, s["value"]),
         Paragraph("<b>RISK LEVEL</b>", s["label"]),
         Paragraph(f'<font color="#{level_color.hexval()[2:]}"><b>{level}</b></font>', s["value"])],
        [Paragraph("<b>GENERATED</b>", s["label"]),
         Paragraph(now, s["value"]),
         Paragraph("<b>RISK SCORE</b>", s["label"]),
         Paragraph(f"{score:.4f} / 1.0000", s["value"])],
        [Paragraph("<b>DATASET</b>", s["label"]),
         Paragraph("CERT Insider Threat r1", s["value"]),
         Paragraph("<b>CLASSIFICATION</b>", s["label"]),
         Paragraph("CONFIDENTIAL — HR/LEGAL", s["value"])],
    ]
    id_table = Table(identity_data, colWidths=[3*cm, 6.5*cm, 3*cm, 5*cm])
    id_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_LIGHT),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, C_LIGHT]),
        ("BOX",      (0,0), (-1,-1), 0.5, C_BLUE),
        ("INNERGRID",(0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(id_table)
    story.append(Spacer(1, 0.5*cm))

    # ── DETECTION SUMMARY ────────────────────────────────────────────────
    story.append(Paragraph("1. DETECTION SUMMARY", s["section"]))

    lstm_flag = "FLAGGED" if int(risk_row.get("lstm_anomaly", 0)) else "CLEAR"
    ueba_flag = "FLAGGED" if int(risk_row.get("ueba_anomaly", 0)) else "CLEAR"
    rule_flag = "FLAGGED" if int(risk_row.get("rule_anomaly", 0)) else "CLEAR"
    lstm_col  = C_RED if lstm_flag == "FLAGGED" else C_GREEN
    ueba_col  = C_RED if ueba_flag == "FLAGGED" else C_GREEN
    rule_col  = C_RED if rule_flag == "FLAGGED" else C_GREEN

    det_data = [
        ["Detection Layer", "Status", "Score (Normalized)", "Weight"],
        ["LSTM Autoencoder",
         lstm_flag,
         f"{float(risk_row.get('lstm_score_norm', 0)):.4f}",
         "0.40"],
        ["UEBA Z-Score Analysis",
         ueba_flag,
         f"{float(risk_row.get('ueba_score_norm', 0)):.4f}",
         "0.35"],
        ["Rule-Based Engine",
         rule_flag,
         f"{float(risk_row.get('rule_score_norm', 0)):.4f}",
         "0.25"],
    ]
    det_table = Table(det_data, colWidths=[6*cm, 3*cm, 5*cm, 3.5*cm])
    det_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR",    (0,0), (-1,0), C_WHITE),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, C_LIGHT]),
        ("BOX",          (0,0), (-1,-1), 0.5, C_BLUE),
        ("INNERGRID",    (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("ALIGN",        (1,1), (1,-1), "CENTER"),
        ("ALIGN",        (2,1), (3,-1), "CENTER"),
        ("TEXTCOLOR",    (1,1), (1,1), lstm_col),
        ("TEXTCOLOR",    (1,2), (1,2), ueba_col),
        ("TEXTCOLOR",    (1,3), (1,3), rule_col),
        ("FONTNAME",     (1,1), (1,-1), "Helvetica-Bold"),
    ]))
    story.append(det_table)
    story.append(Spacer(1, 0.3*cm))

    # Unified score bar
    story.append(Paragraph(
        f"<b>Unified Risk Score:</b>  {score:.4f} / 1.0000  "
        f"(threshold: HIGH > 0.60 | MEDIUM > 0.30 | LOW &lt;= 0.30)",
        s["body"]
    ))

    # ── BEHAVIORAL PROFILE ────────────────────────────────────────────────
    story.append(Paragraph("2. BEHAVIORAL PROFILE", s["section"]))

    feat_data = [
        ["Behavioral Feature", "Value", "Risk Significance"],
        ["Total Logon Events",       str(int(feat_row.get("logon_count",0))),              "Baseline"],
        ["After-Hours Logons",       str(int(feat_row.get("after_hours_logon_count",0))),  "HIGH — abnormal timing"],
        ["Unique PCs Accessed",      str(int(feat_row.get("unique_pcs_used",0))),           "MEDIUM-HIGH — lateral movement"],
        ["Device Connect Events",    str(int(feat_row.get("device_connect_count",0))),     "MEDIUM-HIGH — exfiltration risk"],
        ["After-Hours Device Use",   str(int(feat_row.get("after_hours_device_count",0))), "HIGHEST — exfiltration signal"],
        ["HTTP Visits",              str(int(feat_row.get("http_count",0))),                "Baseline"],
        ["Sessions Without Logoff",  str(int(feat_row.get("logon_without_logoff",0))),     "HIGH — unclosed sessions"],
        ["UEBA Weighted Score",      f"{float(ueba_row.get('ueba_score_weighted',0)):.4f}", f"Threshold: {float(ueba_row.get('ueba_threshold',0)):.4f}"],
    ]
    feat_table = Table(feat_data, colWidths=[6.5*cm, 3*cm, 8*cm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, C_LIGHT]),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("ALIGN",         (1,1), (1,-1), "CENTER"),
    ]))
    story.append(feat_table)

    # ── RULE VIOLATIONS ───────────────────────────────────────────────────
    story.append(Paragraph("3. RULE VIOLATION DETAILS", s["section"]))

    explanation = str(risk_row.get("explanation", "No rules violated"))
    if explanation != "No rules violated" and explanation != "nan":
        violations = explanation.split(" || ")
        viol_data = [["#", "Rule Violation", "Severity"]]
        for i, v in enumerate(violations, 1):
            sev = "HIGH" if "HIGH" in v else ("MEDIUM" if "MEDIUM" in v else "LOW")
            sev_col = C_RED if sev == "HIGH" else (C_ORANGE if sev == "MEDIUM" else C_GREEN)
            viol_data.append([str(i), v.replace(" — HIGH","").replace(" — MEDIUM","").replace(" — LOW",""), sev])
        viol_table = Table(viol_data, colWidths=[1*cm, 14*cm, 2.5*cm])
        viol_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), C_DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, C_LIGHT]),
            ("BOX",           (0,0), (-1,-1), 0.5, C_BLUE),
            ("INNERGRID",     (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("ALIGN",         (0,1), (0,-1), "CENTER"),
            ("ALIGN",         (2,1), (2,-1), "CENTER"),
            ("FONTNAME",      (2,1), (2,-1), "Helvetica-Bold"),
        ]))
        # Color severity cells
        for i, v in enumerate(violations, 1):
            sev = "HIGH" if "HIGH" in v else ("MEDIUM" if "MEDIUM" in v else "LOW")
            col = C_RED if sev == "HIGH" else (C_ORANGE if sev == "MEDIUM" else C_GREEN)
            viol_table.setStyle(TableStyle([("TEXTCOLOR", (2,i), (2,i), col)]))
        story.append(viol_table)
    else:
        story.append(Paragraph("No rule violations detected for this user.", s["body"]))

    story.append(PageBreak())

    # ── SUSPICIOUS EVENT TIMELINE ─────────────────────────────────────────
    story.append(Paragraph("4. SUSPICIOUS EVENT TIMELINE", s["section"]))
    story.append(Paragraph(
        "Events marked [AFTER HOURS] occurred before 8:00am or after 6:00pm. "
        "Device Connect events outside working hours are the highest risk indicator.",
        s["body"]
    ))
    story.append(Spacer(1, 0.2*cm))

    # Filter to suspicious events: after-hours logons, all device connects, after-hours http
    susp = events_df[
        ((events_df["source"] == "logon") & (events_df["after_hrs"])) |
        (events_df["source"] == "device") |
        ((events_df["source"] == "http") & (events_df["after_hrs"]))
    ].head(100)

    if not susp.empty:
        tl_data = [["Timestamp", "Event Type", "Detail", "Flag"]]
        for _, ev in susp.iterrows():
            flag = "[AFTER HOURS]" if ev["after_hrs"] else "[NORMAL HOURS]"
            tl_data.append([
                str(ev["date"])[:19],
                str(ev["type"]),
                str(ev["detail"])[:45],
                flag
            ])
        tl_table = Table(tl_data, colWidths=[4.5*cm, 3.5*cm, 6*cm, 3.5*cm])
        tl_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), C_DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, C_LIGHT]),
            ("BOX",           (0,0), (-1,-1), 0.5, C_BLUE),
            ("INNERGRID",     (0,0), (-1,-1), 0.25, colors.HexColor("#dddddd")),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("FONTNAME",      (0,1), (0,-1), "Courier"),
        ]))
        # Highlight after-hours rows
        for i, (_, ev) in enumerate(susp.iterrows(), 1):
            if ev["after_hrs"]:
                tl_table.setStyle(TableStyle([
                    ("TEXTCOLOR", (3,i), (3,i), C_RED),
                    ("FONTNAME",  (3,i), (3,i), "Helvetica-Bold"),
                ]))
        story.append(tl_table)
        if len(susp) == 100:
            story.append(Paragraph(
                "Note: Timeline truncated to 100 most recent suspicious events. "
                "Full event log available in suspicious_events.csv.",
                s["caption"]
            ))
    else:
        story.append(Paragraph("No suspicious events found in the raw logs for this user.", s["body"]))

    # ── RECOMMENDED ACTIONS ───────────────────────────────────────────────
    story.append(Paragraph("5. RECOMMENDED ACTIONS", s["section"]))
    actions = {
        "HIGH":   [
            "Immediately suspend user account pending investigation",
            "Audit all files accessed or copied in the past 30 days",
            "Review device connection logs and recover any connected media",
            "Interview the user and their line manager",
            "Preserve all log evidence under legal hold",
            "Notify CISO and Legal team within 24 hours",
        ],
        "MEDIUM": [
            "Place user account under enhanced monitoring",
            "Review recent file access and device usage with the user's manager",
            "Schedule a security awareness review session",
            "Re-evaluate risk score after 7 days of monitoring",
        ],
        "LOW": [
            "Log this alert for audit trail purposes",
            "No immediate action required",
            "Continue routine monitoring",
        ]
    }
    for action in actions.get(level, actions["LOW"]):
        story.append(Paragraph(f"   {chr(8226)}  {action}", s["body"]))

    # ── CHAIN OF CUSTODY ──────────────────────────────────────────────────
    story.append(Paragraph("6. CHAIN OF CUSTODY", s["section"]))
    coc_data = [
        ["Item",                  "Value"],
        ["Report Generated By",  "UEBA Insider Threat Detection System"],
        ["Generation Timestamp", now],
        ["Dataset",              "CERT Insider Threat Dataset r1"],
        ["System Version",       "v1.0 — Hybrid UEBA (LSTM + UEBA + Rules)"],
        ["Evidence Integrity",   "All source data preserved unmodified"],
        ["Classification",       "CONFIDENTIAL — HR/Legal Use Only"],
    ]
    coc_table = Table(coc_data, colWidths=[6*cm, 11.5*cm])
    coc_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, C_LIGHT]),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
    ]))
    story.append(coc_table)

    on_page = _make_canvas_maker(username, now)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output_path


# ── Main export function ─────────────────────────────────────────────────────
def generate_package(base_dir, username):
    """
    Generate a complete evidence ZIP package for a user.
    Returns the path to the ZIP file.
    """
    proc     = os.path.join(base_dir, "dataset", "processed")
    out_dir  = os.path.join(base_dir, "evidence_packages")
    os.makedirs(out_dir, exist_ok=True)

    safe_name = username.replace("/", "_").replace("\\", "_")
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg_name  = f"evidence_{safe_name}_{ts}"
    zip_path  = os.path.join(out_dir, f"{pkg_name}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:

        # 1. Forensic PDF report
        pdf_path = os.path.join(out_dir, f"{pkg_name}_report.pdf")
        _build_pdf(base_dir, username, pdf_path)
        zf.write(pdf_path, "forensic_report.pdf")
        os.remove(pdf_path)

        # 2. User risk profile CSV
        risk_path = os.path.join(proc, "final_risk_scores.csv")
        if os.path.exists(risk_path):
            risk_df = pd.read_csv(risk_path)
            user_risk = risk_df[risk_df["user"] == username]
            buf = io.StringIO()
            user_risk.to_csv(buf, index=False)
            zf.writestr("user_risk_profile.csv", buf.getvalue())

        # 3. Suspicious events CSV
        events_df = _load_events(base_dir, username)
        susp = events_df[
            ((events_df["source"] == "logon") & (events_df["after_hrs"])) |
            (events_df["source"] == "device") |
            ((events_df["source"] == "http") & (events_df["after_hrs"]))
        ]
        buf = io.StringIO()
        susp.to_csv(buf, index=False)
        zf.writestr("suspicious_events.csv", buf.getvalue())

        # 4. UEBA scores for this user
        ueba_path = os.path.join(proc, "ueba_scores.csv")
        if os.path.exists(ueba_path):
            ueba_df  = pd.read_csv(ueba_path)
            user_ueba = ueba_df[ueba_df["user"] == username]
            buf = io.StringIO()
            user_ueba.to_csv(buf, index=False)
            zf.writestr("ueba_scores.csv", buf.getvalue())

        # 5. Plain text summary for HR/Legal
        risk_df  = pd.read_csv(risk_path)
        risk_row = risk_df[risk_df["user"] == username].iloc[0]
        feat_df  = pd.read_csv(os.path.join(proc, "user_behavior_features.csv"))
        feat_row = feat_df[feat_df["user"] == username].iloc[0]

        summary_lines = [
            "=" * 60,
            "INSIDER THREAT DETECTION — SECURITY ALERT SUMMARY",
            "=" * 60,
            f"Subject User    : {username}",
            f"Risk Level      : {risk_row.get('risk_level','UNKNOWN')}",
            f"Risk Score      : {float(risk_row.get('final_risk_score',0)):.4f} / 1.0000",
            f"Generated At    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Classification  : CONFIDENTIAL",
            "",
            "DETECTION FLAGS",
            "-" * 60,
            f"LSTM Autoencoder    : {'FLAGGED' if int(risk_row.get('lstm_anomaly',0)) else 'CLEAR'}",
            f"UEBA Z-Score        : {'FLAGGED' if int(risk_row.get('ueba_anomaly',0)) else 'CLEAR'}",
            f"Rule-Based Engine   : {'FLAGGED' if int(risk_row.get('rule_anomaly',0)) else 'CLEAR'}",
            "",
            "BEHAVIORAL SUMMARY",
            "-" * 60,
            f"Total Logons            : {int(feat_row.get('logon_count',0))}",
            f"After-Hours Logons      : {int(feat_row.get('after_hours_logon_count',0))}",
            f"Unique PCs Accessed     : {int(feat_row.get('unique_pcs_used',0))}",
            f"Device Connections      : {int(feat_row.get('device_connect_count',0))}",
            f"After-Hours Device Use  : {int(feat_row.get('after_hours_device_count',0))}",
            f"HTTP Visits             : {int(feat_row.get('http_count',0))}",
            f"Sessions Without Logoff : {int(feat_row.get('logon_without_logoff',0))}",
            "",
            "RULE VIOLATIONS",
            "-" * 60,
        ]
        explanation = str(risk_row.get("explanation", "No violations"))
        if explanation != "nan":
            for v in explanation.split(" || "):
                summary_lines.append(f"  -> {v}")
        else:
            summary_lines.append("  No rule violations detected.")

        summary_lines += [
            "",
            "=" * 60,
            "This report was generated automatically by the UEBA",
            "Insider Threat Detection System. All source data has",
            "been preserved unmodified as part of this package.",
            "=" * 60,
        ]
        zf.writestr("detection_summary.txt", "\n".join(summary_lines))

        # 6. README
        readme = (
            "EVIDENCE PACKAGE CONTENTS\n"
            "==========================\n"
            f"Subject User : {username}\n"
            f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Files:\n"
            "  forensic_report.pdf    - Full forensic investigation report (share with Legal)\n"
            "  user_risk_profile.csv  - Complete risk scores from all detection layers\n"
            "  suspicious_events.csv  - Timeline of suspicious raw log events\n"
            "  ueba_scores.csv        - Statistical behavioral scores per feature\n"
            "  detection_summary.txt  - Plain text summary (share with HR)\n"
            "  README.txt             - This file\n"
        )
        zf.writestr("README.txt", readme)

    return zip_path


if __name__ == "__main__":
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user = sys.argv[1] if len(sys.argv) > 1 else "DTAA/SJH0588"
    print(f"Generating evidence package for: {user}")
    path = generate_package(BASE_DIR, user)
    print(f"Package saved -> {path}")