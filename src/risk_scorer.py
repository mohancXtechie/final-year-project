"""
Unified Risk Scorer
====================
Combines UEBA, LSTM, and Rule-based scores into a single
final risk score per user using weighted combination.

Weights (justified):
  LSTM  : 0.40  - most sophisticated, learns behavioral patterns
  UEBA  : 0.35  - statistically grounded z-score analysis
  Rules : 0.25  - simple threshold checks, quick detection

Final risk score is normalized to 0-1 and classified as:
  HIGH   : score > 0.60
  MEDIUM : score > 0.30
  LOW    : score <= 0.30
"""

import pandas as pd
import numpy as np
import os

LSTM_WEIGHT = 0.40
UEBA_WEIGHT = 0.35
RULE_WEIGHT = 0.25

def run(base_dir):
    print("=" * 50)
    print("STEP 8: Unified Risk Scoring")
    print("=" * 50)

    # -- Load all three score sources ---------------------------------------
    ueba_path  = os.path.join(base_dir, "dataset", "processed", "ueba_scores.csv")
    acc_path   = os.path.join(base_dir, "dataset", "processed", "accuracy_results.csv")
    rule_path  = os.path.join(base_dir, "dataset", "processed", "rule_scores.csv")

    for path in [ueba_path, acc_path, rule_path]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run previous steps first.")
            return

    ueba = pd.read_csv(ueba_path)[["user", "ueba_score_weighted", "ueba_threshold"]]
    acc  = pd.read_csv(acc_path)[["user", "avg_reconstruction_error", "lstm_anomaly", "ueba_anomaly"]]
    rule = pd.read_csv(rule_path)[["user", "rule_score", "rule_score_norm",
                                   "rule_anomaly", "rules_violated", "violation_list", "explanation"]]

    # -- Normalize UEBA score to 0-1 ----------------------------------------
    ueba_max = ueba["ueba_score_weighted"].max()
    ueba_min = ueba["ueba_score_weighted"].min()
    ueba["ueba_score_norm"] = (
        (ueba["ueba_score_weighted"] - ueba_min) / (ueba_max - ueba_min)
    )

    # -- Normalize LSTM reconstruction error to 0-1 -------------------------
    lstm_max = acc["avg_reconstruction_error"].max()
    lstm_min = acc["avg_reconstruction_error"].min()
    acc["lstm_score_norm"] = (
        (acc["avg_reconstruction_error"] - lstm_min) / (lstm_max - lstm_min)
    )

    # -- Merge all scores on user -------------------------------------------
    merged = ueba.merge(acc, on="user", how="inner")
    merged = merged.merge(rule, on="user", how="inner")

    # -- Calculate weighted final risk score --------------------------------
    merged["final_risk_score"] = (
        (merged["lstm_score_norm"]  * LSTM_WEIGHT) +
        (merged["ueba_score_norm"]  * UEBA_WEIGHT) +
        (merged["rule_score_norm"]  * RULE_WEIGHT)
    ).round(4)

    # -- Classify risk level ------------------------------------------------
    def classify_risk(score):
        if score > 0.60:
            return "HIGH"
        elif score > 0.30:
            return "MEDIUM"
        else:
            return "LOW"

    merged["risk_level"] = merged["final_risk_score"].apply(classify_risk)

    # -- Any method flagging this user as anomaly ---------------------------
    merged["any_anomaly"] = (
        (merged["lstm_anomaly"] == 1) |
        (merged["ueba_anomaly"] == 1) |
        (merged["rule_anomaly"] == 1)
    ).astype(int)

    # -- Sort by final risk score -------------------------------------------
    merged = merged.sort_values("final_risk_score", ascending=False).reset_index(drop=True)

    # -- Summary stats -------------------------------------------------------
    high   = merged[merged["risk_level"] == "HIGH"]
    medium = merged[merged["risk_level"] == "MEDIUM"]
    low    = merged[merged["risk_level"] == "LOW"]

    print(f"\nWeights used:")
    print(f"  LSTM  : {LSTM_WEIGHT}")
    print(f"  UEBA  : {UEBA_WEIGHT}")
    print(f"  Rules : {RULE_WEIGHT}")

    print(f"\nRisk Classification:")
    print(f"  HIGH   (score > 0.60) : {len(high)} users")
    print(f"  MEDIUM (score > 0.30) : {len(medium)} users")
    print(f"  LOW    (score <= 0.30): {len(low)} users")

    print(f"\nTop 15 highest risk users:")
    print(f"{'Rank':<5} {'User':<20} {'Score':<8} {'Level':<8} {'LSTM':<6} {'UEBA':<6} {'Rules':<6} {'Violations'}")
    print("-" * 90)
    for i, row in merged.head(15).iterrows():
        print(
            f"{i+1:<5} {row['user']:<20} {row['final_risk_score']:<8.4f} "
            f"{row['risk_level']:<8} {row['lstm_anomaly']:<6} {row['ueba_anomaly']:<6} "
            f"{row['rule_anomaly']:<6} {row['rules_violated']}"
        )

    print(f"\nDetailed explanations for HIGH risk users:")
    print("-" * 60)
    for _, row in high.iterrows():
        print(f"\nUser: {row['user']}  |  Score: {row['final_risk_score']}  |  Level: HIGH")
        print(f"  LSTM flagged : {'Yes' if row['lstm_anomaly'] else 'No'}")
        print(f"  UEBA flagged : {'Yes' if row['ueba_anomaly'] else 'No'}")
        print(f"  Rules flagged: {'Yes' if row['rule_anomaly'] else 'No'} ({row['rules_violated']} violations)")
        if row["explanation"] != "No rules violated":
            for line in str(row["explanation"]).split(" || "):
                print(f"    -> {line}")

    # -- Select columns to save ---------------------------------------------
    out_cols = [
        "user", "final_risk_score", "risk_level",
        "ueba_score_weighted", "ueba_score_norm", "ueba_anomaly",
        "avg_reconstruction_error", "lstm_score_norm", "lstm_anomaly",
        "rule_score", "rule_score_norm", "rule_anomaly",
        "rules_violated", "violation_list", "explanation", "any_anomaly"
    ]
    out_df = merged[out_cols]

    out_path = os.path.join(base_dir, "dataset", "processed", "final_risk_scores.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nFinal risk scores saved -> {out_path}")
    print("=" * 50)