"""
Live Demo Mode
===============
Injects synthetic malicious users into the processed pipeline data
so the full detection system reacts in real time during a presentation.

Three scenarios available:
  A - Data Exfiltration : heavy device usage + after-hours activity
  B - Credential Theft  : many after-hours logins across multiple PCs
  C - Negligent Insider : borderline behavior, should score MEDIUM

Usage:
    from src.demo_mode import inject_scenario, clear_demo_users
    inject_scenario(base_dir, "A")   # injects scenario A
    clear_demo_users(base_dir)       # removes all demo users
"""

import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore


# ── Scenario definitions ────────────────────────────────────────────────────
SCENARIOS = {
    "A": {
        "label"      : "Data Exfiltration",
        "user"       : "DEMO/EVL0001",
        "description": (
            "High-volume USB device activity after hours. "
            "Classic data exfiltration signature: multiple device connections "
            "late at night across several machines."
        ),
        "expected"   : "HIGH RISK — All three detection layers should fire.",
        "features"   : {
            "logon_count"              : 680,
            "logoff_count"             : 320,
            "after_hours_logon_count"  : 390.0,
            "unique_pcs_used"          : 18,
            "device_connect_count"     : 340.0,
            "after_hours_device_count" : 85.0,
            "http_count"               : 2100,
            "logon_without_logoff"     : 360,
        }
    },
    "B": {
        "label"      : "Credential Theft / Account Compromise",
        "user"       : "DEMO/EVL0002",
        "description": (
            "Massive after-hours login activity across an unusually large "
            "number of distinct machines. Pattern consistent with a compromised "
            "account being used for lateral movement."
        ),
        "expected"   : "HIGH RISK — UEBA and Rule Engine should fire strongly.",
        "features"   : {
            "logon_count"              : 2500,
            "logoff_count"             : 2500,
            "after_hours_logon_count"  : 1500.0,
            "unique_pcs_used"          : 880,
            "device_connect_count"     : 0.0,
            "after_hours_device_count" : 0.0,
            "http_count"               : 1200,
            "logon_without_logoff"     : 0,
        }
    },
    "C": {
        "label"      : "Negligent Insider",
        "user"       : "DEMO/EVL0003",
        "description": (
            "Moderate after-hours activity with some device usage. "
            "Could be an employee working late legitimately, or early-stage "
            "suspicious behavior. Should score MEDIUM risk."
        ),
        "expected"   : "MEDIUM RISK — Borderline case, monitoring recommended.",
        "features"   : {
            "logon_count"              : 420,
            "logoff_count"             : 390,
            "after_hours_logon_count"  : 45.0,
            "unique_pcs_used"          : 4,
            "device_connect_count"     : 28.0,
            "after_hours_device_count" : 8.0,
            "http_count"               : 3800,
            "logon_without_logoff"     : 30,
        }
    }
}

DEMO_USERS = [s["user"] for s in SCENARIOS.values()]
RULE_CONFIGS = [
    ("after_hours_logon_count",  2.0, 2.0),
    ("device_connect_count",     2.0, 1.5),
    ("after_hours_device_count", 1.5, 2.5),
    ("unique_pcs_used",          2.0, 1.5),
    ("logon_without_logoff",     2.0, 2.0),
    ("http_count",               2.5, 1.0),
    ("logon_count",              2.5, 1.0),
]
RULE_DESCS = {
    "after_hours_logon_count"  : "Excessive after-hours logins",
    "device_connect_count"     : "Excessive device connections",
    "after_hours_device_count" : "After-hours device usage",
    "unique_pcs_used"          : "Accessing unusual number of PCs",
    "logon_without_logoff"     : "High number of unclosed sessions",
    "http_count"               : "Extremely high web activity",
    "logon_count"              : "Extremely high login frequency",
}


# ── Backup helpers ───────────────────────────────────────────────────────────
def _backup(path):
    backup = path + ".demo_backup"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

def _restore(path):
    backup = path + ".demo_backup"
    if os.path.exists(backup):
        shutil.copy2(backup, path)
        os.remove(backup)


# ── Core injection ───────────────────────────────────────────────────────────
def inject_scenario(base_dir, scenario_key):
    """
    Inject one or more demo users into the processed pipeline data.
    Runs the risk scoring steps automatically on the augmented data.
    Returns a dict with the injected users and their computed scores.
    """
    if scenario_key == "ALL":
        keys = list(SCENARIOS.keys())
    elif scenario_key in SCENARIOS:
        keys = [scenario_key]
    else:
        raise ValueError(f"Unknown scenario: {scenario_key}. Choose A, B, C, or ALL.")

    proc = os.path.join(base_dir, "dataset", "processed")

    # Paths
    feat_path  = os.path.join(proc, "user_behavior_features.csv")
    ueba_path  = os.path.join(proc, "ueba_scores.csv")
    rule_path  = os.path.join(proc, "rule_scores.csv")
    acc_path   = os.path.join(proc, "accuracy_results.csv")
    risk_path  = os.path.join(proc, "final_risk_scores.csv")

    # Backup originals
    for path in [feat_path, ueba_path, rule_path, acc_path, risk_path]:
        if os.path.exists(path):
            _backup(path)

    # ── Step 1: Inject into user_behavior_features.csv ─────────────────
    feat_df = pd.read_csv(feat_path)
    # Remove any existing demo users first
    feat_df = feat_df[~feat_df["user"].isin(DEMO_USERS)]

    new_rows = []
    for k in keys:
        row = {"user": SCENARIOS[k]["user"]}
        row.update(SCENARIOS[k]["features"])
        new_rows.append(row)

    feat_df = pd.concat([feat_df, pd.DataFrame(new_rows)], ignore_index=True)
    feat_df.to_csv(feat_path, index=False)

    # ── Step 2: Recompute UEBA scores including demo users ──────────────
    feature_cols = [
        "logon_count", "logoff_count", "after_hours_logon_count",
        "unique_pcs_used", "device_connect_count", "after_hours_device_count",
        "http_count", "logon_without_logoff"
    ]
    weights = {
        "logon_count"              : 1.0,
        "logoff_count"             : 1.0,
        "after_hours_logon_count"  : 2.0,
        "unique_pcs_used"          : 1.5,
        "device_connect_count"     : 1.5,
        "after_hours_device_count" : 2.5,
        "http_count"               : 1.0,
        "logon_without_logoff"     : 2.0,
    }

    ueba_df = feat_df.copy()
    for col in feature_cols:
        ueba_df[f"{col}_zscore"] = zscore(ueba_df[col])

    ueba_df["ueba_score"] = sum(
        abs(ueba_df[f"{col}_zscore"]) for col in feature_cols
    )
    ueba_df["ueba_score_weighted"] = sum(
        abs(ueba_df[f"{col}_zscore"]) * weights[col] for col in feature_cols
    )

    wmean = ueba_df["ueba_score_weighted"].mean()
    wstd  = ueba_df["ueba_score_weighted"].std()
    threshold = wmean + (1.5 * wstd)
    ueba_df["ueba_threshold"] = threshold
    ueba_df.to_csv(ueba_path, index=False)

    # ── Step 3: Recompute rule scores including demo users ──────────────
    rule_results = []
    for _, row in feat_df.iterrows():
        rule_score  = 0.0
        violations  = []
        explanations = []
        for col, mult, weight, in RULE_CONFIGS:
            mean_val = feat_df[col].mean()
            std_val  = feat_df[col].std()
            thr      = mean_val + mult * std_val
            val      = row[col]
            if val > thr:
                dev = (val - thr) / std_val if std_val > 0 else 0
                sev     = "HIGH" if dev >= 3 else ("MEDIUM" if dev >= 2 else "LOW")
                sev_pts = 3 if dev >= 3 else (2 if dev >= 2 else 1)
                rule_score += sev_pts * weight
                violations.append(col)
                explanations.append(
                    f"{RULE_DESCS[col]}: {val:.0f} "
                    f"(threshold: {thr:.0f}, {dev:.2f}x above \u2014 {sev})"
                )
        rule_results.append({
            "user"          : row["user"],
            "rule_score"    : round(rule_score, 4),
            "rules_violated": len(violations),
            "violation_list": " | ".join(violations) if violations else "none",
            "explanation"   : " || ".join(explanations) if explanations else "No rules violated"
        })

    rule_df = pd.DataFrame(rule_results)
    max_score = rule_df["rule_score"].max()
    rule_df["rule_score_norm"] = rule_df["rule_score"] / max_score if max_score > 0 else 0
    rule_thresh = rule_df["rule_score"].mean() + 1.5 * rule_df["rule_score"].std()
    rule_df["rule_anomaly"] = (rule_df["rule_score"] > rule_thresh).astype(int)
    rule_df.to_csv(rule_path, index=False)

    # ── Step 4: Compute LSTM proxy scores for demo users ────────────────
    # Since we cannot run actual LSTM inference on injected users
    # (they have no sequences), we derive a proxy score from their
    # behavioral features relative to the population
    acc_df  = pd.read_csv(acc_path)
    acc_df  = acc_df[~acc_df["user"].isin(DEMO_USERS)]

    # Compute proxy reconstruction error for demo users
    # Using normalised feature deviation as a proxy for LSTM error
    lstm_mean = acc_df["avg_reconstruction_error"].mean()
    lstm_std  = acc_df["avg_reconstruction_error"].std()

    demo_acc_rows = []
    for k in keys:
        sc = SCENARIOS[k]
        feats = sc["features"]
        # How extreme is this user vs population
        ah_logon_z  = abs(feats["after_hours_logon_count"] - feat_df["after_hours_logon_count"].mean()) / feat_df["after_hours_logon_count"].std() if feat_df["after_hours_logon_count"].std() > 0 else 0
        device_z    = abs(feats["device_connect_count"] - feat_df["device_connect_count"].mean()) / feat_df["device_connect_count"].std() if feat_df["device_connect_count"].std() > 0 else 0
        ah_device_z = abs(feats["after_hours_device_count"] - feat_df["after_hours_device_count"].mean()) / feat_df["after_hours_device_count"].std() if feat_df["after_hours_device_count"].std() > 0 else 0

        combined_z  = (ah_logon_z * 0.35 + device_z * 0.3 + ah_device_z * 0.35)
        proxy_error = lstm_mean + (combined_z * lstm_std * 0.8)

        lstm_thresh = acc_df["avg_reconstruction_error"].mean() + 2 * acc_df["avg_reconstruction_error"].std()
        lstm_anom   = 1 if proxy_error > lstm_thresh else 0
        ueba_anom   = 1 if ueba_df[ueba_df["user"] == sc["user"]]["ueba_score_weighted"].values[0] > threshold else 0

        demo_acc_rows.append({
            "user"                    : sc["user"],
            "avg_reconstruction_error": proxy_error,
            "lstm_anomaly"            : lstm_anom,
            "ueba_anomaly"            : ueba_anom,
        })

    acc_df = pd.concat([acc_df, pd.DataFrame(demo_acc_rows)], ignore_index=True)
    acc_df.to_csv(acc_path, index=False)

    # ── Step 5: Recompute unified risk scores ────────────────────────────
    risk_df = pd.read_csv(risk_path)
    risk_df = risk_df[~risk_df["user"].isin(DEMO_USERS)]

    # Merge all sources
    ueba_sub = ueba_df[["user", "ueba_score_weighted", "ueba_threshold"]].copy()
    acc_sub  = acc_df[["user", "avg_reconstruction_error", "lstm_anomaly", "ueba_anomaly"]].copy()
    rule_sub = rule_df[["user", "rule_score", "rule_score_norm", "rule_anomaly",
                         "rules_violated", "violation_list", "explanation"]].copy()

    merged = acc_sub.merge(ueba_sub, on="user", how="inner")
    merged = merged.merge(rule_sub, on="user", how="inner")

    ueba_max = merged["ueba_score_weighted"].max()
    ueba_min = merged["ueba_score_weighted"].min()
    merged["ueba_score_norm"] = (merged["ueba_score_weighted"] - ueba_min) / (ueba_max - ueba_min) if ueba_max > ueba_min else 0

    lstm_max = merged["avg_reconstruction_error"].max()
    lstm_min = merged["avg_reconstruction_error"].min()
    merged["lstm_score_norm"] = (merged["avg_reconstruction_error"] - lstm_min) / (lstm_max - lstm_min) if lstm_max > lstm_min else 0

    merged["final_risk_score"] = (
        merged["lstm_score_norm"]  * 0.40 +
        merged["ueba_score_norm"]  * 0.35 +
        merged["rule_score_norm"]  * 0.25
    ).round(4)

    def classify(s):
        return "HIGH" if s > 0.60 else ("MEDIUM" if s > 0.30 else "LOW")

    merged["risk_level"]  = merged["final_risk_score"].apply(classify)
    merged["any_anomaly"] = ((merged["lstm_anomaly"]==1)|(merged["ueba_anomaly"]==1)|(merged["rule_anomaly"]==1)).astype(int)
    merged = merged.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    merged.to_csv(risk_path, index=False)

    # ── Return results for dashboard display ─────────────────────────────
    results = {}
    for k in keys:
        user = SCENARIOS[k]["user"]
        row  = merged[merged["user"] == user]
        if not row.empty:
            r = row.iloc[0]
            results[k] = {
                "user"            : user,
                "scenario"        : SCENARIOS[k]["label"],
                "description"     : SCENARIOS[k]["description"],
                "expected"        : SCENARIOS[k]["expected"],
                "final_risk_score": float(r["final_risk_score"]),
                "risk_level"      : str(r["risk_level"]),
                "lstm_anomaly"    : int(r["lstm_anomaly"]),
                "ueba_anomaly"    : int(r["ueba_anomaly"]),
                "rule_anomaly"    : int(r["rule_anomaly"]),
                "rules_violated"  : int(r["rules_violated"]),
                "explanation"     : str(r["explanation"]),
            }
    return results


def clear_demo_users(base_dir):
    """
    Remove all injected demo users and restore original files from backup.
    """
    proc = os.path.join(base_dir, "dataset", "processed")
    files = [
        os.path.join(proc, "user_behavior_features.csv"),
        os.path.join(proc, "ueba_scores.csv"),
        os.path.join(proc, "rule_scores.csv"),
        os.path.join(proc, "accuracy_results.csv"),
        os.path.join(proc, "final_risk_scores.csv"),
    ]
    restored = 0
    for path in files:
        backup = path + ".demo_backup"
        if os.path.exists(backup):
            shutil.copy2(backup, path)
            os.remove(backup)
            restored += 1
    return restored


def get_scenario_info():
    """Return scenario descriptions for display in dashboard."""
    return {k: {
        "label"      : v["label"],
        "user"       : v["user"],
        "description": v["description"],
        "expected"   : v["expected"],
    } for k, v in SCENARIOS.items()}


if __name__ == "__main__":
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    action   = sys.argv[1] if len(sys.argv) > 1 else "inject"
    scenario = sys.argv[2] if len(sys.argv) > 2 else "ALL"

    if action == "inject":
        print(f"Injecting scenario: {scenario}")
        results = inject_scenario(BASE_DIR, scenario)
        for k, r in results.items():
            print(f"\nScenario {k}: {r['scenario']}")
            print(f"  User       : {r['user']}")
            print(f"  Risk Level : {r['risk_level']}")
            print(f"  Risk Score : {r['final_risk_score']:.4f}")
            print(f"  LSTM       : {'FLAGGED' if r['lstm_anomaly'] else 'clear'}")
            print(f"  UEBA       : {'FLAGGED' if r['ueba_anomaly'] else 'clear'}")
            print(f"  Rules      : {'FLAGGED' if r['rule_anomaly'] else 'clear'} ({r['rules_violated']} violations)")
    elif action == "clear":
        n = clear_demo_users(BASE_DIR)
        print(f"Restored {n} files from backup. Demo users removed.")