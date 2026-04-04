"""
Rule-Based Detection Engine
============================
Applies dynamic threshold rules to user behavior features.
Thresholds are calculated from the actual data distribution
(mean + multiplier * std) so the system self-calibrates to
any dataset without hardcoded values.

Each rule violation is scored by severity based on how far
the user exceeds the threshold:
  1-2x std above threshold -> LOW    -> 1 point
  2-3x std above threshold -> MEDIUM -> 2 points
  3x+  std above threshold -> HIGH   -> 3 points
"""

import pandas as pd
import numpy as np
import os

# Rule definitions
# Each rule: (feature_column, std_multiplier_for_threshold, weight, description)
RULES = [
    ("after_hours_logon_count",  2.0, 2.0, "Excessive after-hours logins"),
    ("device_connect_count",     2.0, 1.5, "Excessive device connections"),
    ("after_hours_device_count", 1.5, 2.5, "After-hours device usage"),
    ("unique_pcs_used",          2.0, 1.5, "Accessing unusual number of PCs"),
    ("logon_without_logoff",     2.0, 2.0, "High number of unclosed sessions"),
    ("http_count",               2.5, 1.0, "Extremely high web activity"),
    ("logon_count",              2.5, 1.0, "Extremely high login frequency"),
]

def run(base_dir):
    print("=" * 50)
    print("STEP 7: Rule-Based Detection Engine")
    print("=" * 50)

    # -- Load features ------------------------------------------------------
    features_path = os.path.join(base_dir, "dataset", "processed", "user_behavior_features.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} users for rule evaluation.")

    # -- Calculate dynamic thresholds from data -----------------------------
    print("\nCalculating dynamic thresholds from data distribution...")
    thresholds = {}
    for col, multiplier, weight, desc in RULES:
        mean = df[col].mean()
        std  = df[col].std()
        thresh = mean + (multiplier * std)
        thresholds[col] = {
            "mean"      : mean,
            "std"       : std,
            "threshold" : thresh,
            "weight"    : weight,
            "desc"      : desc
        }
        print(f"  {desc:<40} threshold: {thresh:.2f}  (mean={mean:.2f}, std={std:.2f})")

    # -- Evaluate each user against all rules -------------------------------
    print("\nEvaluating users against rules...")

    results = []
    for _, row in df.iterrows():
        user         = row["user"]
        rule_score   = 0.0
        violations   = []
        explanations = []

        for col, multiplier, weight, desc in RULES:
            t    = thresholds[col]
            val  = row[col]
            thr  = t["threshold"]
            std  = t["std"]

            if val > thr:
                # How many std deviations above threshold
                deviation = (val - thr) / std if std > 0 else 0

                # Severity based on deviation
                if deviation >= 3:
                    severity       = "HIGH"
                    severity_pts   = 3
                elif deviation >= 2:
                    severity       = "MEDIUM"
                    severity_pts   = 2
                else:
                    severity       = "LOW"
                    severity_pts   = 1

                weighted_pts = severity_pts * weight
                rule_score  += weighted_pts
                violations.append(col)

                explanations.append(
                    f"{desc}: {val:.0f} (threshold: {thr:.0f}, "
                    f"{deviation:.2f}x above — {severity})"
                )

        results.append({
            "user"            : user,
            "rule_score"      : round(rule_score, 4),
            "rules_violated"  : len(violations),
            "violation_list"  : " | ".join(violations) if violations else "none",
            "explanation"     : " || ".join(explanations) if explanations else "No rules violated"
        })

    rule_df = pd.DataFrame(results)

    # -- Normalize rule score to 0-1 ----------------------------------------
    max_score = rule_df["rule_score"].max()
    rule_df["rule_score_norm"] = (
        rule_df["rule_score"] / max_score if max_score > 0 else 0
    )

    # -- Rule-based anomaly: top users by rule score ------------------------
    rule_threshold = rule_df["rule_score"].mean() + (1.5 * rule_df["rule_score"].std())
    rule_df["rule_anomaly"] = (rule_df["rule_score"] > rule_threshold).astype(int)

    flagged = rule_df[rule_df["rule_anomaly"] == 1]
    print(f"\nRule-based anomaly threshold : {rule_threshold:.4f}")
    print(f"Users flagged by rule engine : {len(flagged)} / {len(rule_df)}")

    print("\nTop 10 users by rule score:")
    top10 = rule_df.sort_values("rule_score", ascending=False).head(10)
    for _, r in top10.iterrows():
        print(f"  {r['user']:<20} score={r['rule_score']:.2f}  violations={r['rules_violated']}")
        if r["explanation"] != "No rules violated":
            for line in r["explanation"].split(" || "):
                print(f"    -> {line}")

    # -- Save ---------------------------------------------------------------
    out_path = os.path.join(base_dir, "dataset", "processed", "rule_scores.csv")
    rule_df.to_csv(out_path, index=False)
    print(f"\nRule scores saved -> {out_path}")
    print("=" * 50)