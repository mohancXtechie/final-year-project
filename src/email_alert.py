"""
Email Alert System
===================
Sends a formatted insider threat report email to the configured
admin address after the pipeline completes.

Setup:
  1. Create a .env file in the project root with:
       ALERT_SENDER_EMAIL=your_gmail@gmail.com
       ALERT_SENDER_PASSWORD=your_16_char_app_password
       ALERT_RECEIVER_EMAIL=admin@yourorg.com

  2. For Gmail, generate an App Password at:
       Google Account -> Security -> 2-Step Verification -> App Passwords
     Use "Mail" as the app and copy the 16-character password.

  NEVER hardcode credentials in code.
"""

import os
import sys
import smtplib
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def _load_env(base_dir):
    """Load .env file manually without requiring python-dotenv."""
    env_path = os.path.join(base_dir, ".env")
    config   = {}
    if not os.path.exists(env_path):
        return config
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config

def _build_email_body(high_df, medium_df, total_users, detection_time):
    """Build plain text email body."""

    lines = []
    lines.append("=" * 60)
    lines.append("  INSIDER THREAT DETECTION - SECURITY ALERT REPORT")
    lines.append("=" * 60)
    lines.append(f"  Detection Time : {detection_time}")
    lines.append(f"  Total Users    : {total_users}")
    lines.append(f"  HIGH Risk      : {len(high_df)}")
    lines.append(f"  MEDIUM Risk    : {len(medium_df)}")
    lines.append("=" * 60)

    if len(high_df) > 0:
        lines.append("\nHIGH RISK USERS — Immediate Investigation Required")
        lines.append("-" * 60)
        for _, row in high_df.iterrows():
            lines.append(f"\n  User       : {row['user']}")
            lines.append(f"  Risk Score : {row['final_risk_score']:.4f}")
            lines.append(f"  LSTM Flagged  : {'Yes' if row['lstm_anomaly'] else 'No'}")
            lines.append(f"  UEBA Flagged  : {'Yes' if row['ueba_anomaly'] else 'No'}")
            lines.append(f"  Rules Violated: {row['rules_violated']}")
            if row["explanation"] != "No rules violated":
                lines.append("  Rule Details:")
                for detail in str(row["explanation"]).split(" || "):
                    lines.append(f"    -> {detail}")
    else:
        lines.append("\nNo HIGH risk users detected.")

    if len(medium_df) > 0:
        lines.append(f"\nMEDIUM RISK USERS — Monitor Closely ({len(medium_df)} users)")
        lines.append("-" * 60)
        for _, row in medium_df.iterrows():
            lines.append(
                f"  {row['user']:<22} score={row['final_risk_score']:.4f}  "
                f"violations={row['rules_violated']}"
            )
    else:
        lines.append("\nNo MEDIUM risk users detected.")

    lines.append("\n" + "=" * 60)
    lines.append("This is an automated alert from the UEBA Detection System.")
    lines.append("Review the full report in dataset/processed/final_risk_scores.csv")
    lines.append("=" * 60)

    return "\n".join(lines)

def run(base_dir):
    print("=" * 50)
    print("STEP 9: Email Alert System")
    print("=" * 50)

    # -- Load credentials from .env -----------------------------------------
    config = _load_env(base_dir)

    sender_email   = config.get("ALERT_SENDER_EMAIL", "")
    sender_password = config.get("ALERT_SENDER_PASSWORD", "")
    receiver_email  = config.get("ALERT_RECEIVER_EMAIL", "")

    if not all([sender_email, sender_password, receiver_email]):
        print("Email credentials not configured.")
        print("To enable email alerts, create a .env file in the project root with:")
        print("  ALERT_SENDER_EMAIL=your_gmail@gmail.com")
        print("  ALERT_SENDER_PASSWORD=your_16_char_app_password")
        print("  ALERT_RECEIVER_EMAIL=admin@yourorg.com")
        print("\nSkipping email alert.")
        print("=" * 50)
        return

    # -- Load final risk scores ---------------------------------------------
    risk_path = os.path.join(base_dir, "dataset", "processed", "final_risk_scores.csv")
    if not os.path.exists(risk_path):
        print("ERROR: final_risk_scores.csv not found.")
        print("Please run risk_scorer first.")
        return

    df = pd.read_csv(risk_path)
    high_df   = df[df["risk_level"] == "HIGH"].sort_values("final_risk_score", ascending=False)
    medium_df = df[df["risk_level"] == "MEDIUM"].sort_values("final_risk_score", ascending=False)

    total_users    = len(df)
    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"HIGH risk users   : {len(high_df)}")
    print(f"MEDIUM risk users : {len(medium_df)}")

    # -- Only send if there are high or medium risk users -------------------
    if len(high_df) == 0 and len(medium_df) == 0:
        print("No HIGH or MEDIUM risk users detected. No email sent.")
        print("=" * 50)
        return

    # -- Build email --------------------------------------------------------
    subject = (
        f"[UEBA ALERT] {len(high_df)} High Risk, {len(medium_df)} Medium Risk "
        f"Users Detected — {detection_time}"
    )
    body = _build_email_body(high_df, medium_df, total_users, detection_time)

    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # -- Send via Gmail SMTP ------------------------------------------------
    print(f"\nSending alert to: {receiver_email}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully.")
    except smtplib.SMTPAuthenticationError:
        print("ERROR: Gmail authentication failed.")
        print("Make sure you are using a Gmail App Password, not your regular password.")
        print("Generate one at: Google Account -> Security -> App Passwords")
    except Exception as e:
        print(f"ERROR: Failed to send email: {e}")

    print("=" * 50)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(BASE_DIR)