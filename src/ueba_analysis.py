# import pandas as pd
# import os
# from scipy.stats import zscore

# def run(base_dir):
#     print("=" * 50)
#     print("STEP 2: UEBA Analysis")
#     print("=" * 50)

#     print("Loading user behavior dataset...")
#     in_path = os.path.join(base_dir, "dataset", "processed", "user_behavior_features.csv")
#     df = pd.read_csv(in_path)

#     print("Calculating statistical deviations (Z-scores)...")

#     # All feature columns to include in anomaly scoring
#     feature_cols = [
#         "logon_count",
#         "logoff_count",
#         "after_hours_logon_count",
#         "unique_pcs_used",
#         "device_connect_count",
#         "after_hours_device_count",
#         "http_count",
#         "logon_without_logoff"
#     ]

#     # Calculate zscore for each feature
#     for col in feature_cols:
#         df[f"{col}_zscore"] = zscore(df[col])

#     # Combined UEBA anomaly score — sum of absolute Z-scores across all features
#     df["ueba_score"] = sum(abs(df[f"{col}_zscore"]) for col in feature_cols)

#     # Weighted score — extra weight on high-risk features
#     df["ueba_score_weighted"] = (
#         abs(df["logon_count_zscore"])              * 1.0 +
#         abs(df["after_hours_logon_count_zscore"])  * 2.0 +
#         abs(df["unique_pcs_used_zscore"])           * 1.5 +
#         abs(df["device_connect_count_zscore"])      * 1.5 +
#         abs(df["after_hours_device_count_zscore"])  * 2.5 +
#         abs(df["http_count_zscore"])                * 1.0 +
#         abs(df["logon_without_logoff_zscore"])      * 2.0
#     )

#     # Threshold = mean + 2 standard deviations
#     # This is statistically grounded and adapts to the actual data distribution
#     weighted_mean = df["ueba_score_weighted"].mean()
#     weighted_std  = df["ueba_score_weighted"].std()
#     threshold     = weighted_mean + (2 * weighted_std)

#     print(f"\nWeighted score stats:")
#     print(f"  Mean      : {weighted_mean:.4f}")
#     print(f"  Std       : {weighted_std:.4f}")
#     print(f"  Threshold : {threshold:.4f}  (mean + 2 * std)")

#     suspicious = df[df["ueba_score_weighted"] > threshold]
#     print(f"\nSuspicious users detected: {len(suspicious)} / {len(df)}")

#     # Save threshold into csv so accuracy.py and graph_analysis.py use same value
#     df["ueba_threshold"] = threshold

#     print("\nUEBA Scores (sample):")
#     display_cols = [
#         "user", "logon_count", "after_hours_logon_count",
#         "device_connect_count", "http_count", "ueba_score_weighted", "ueba_threshold"
#     ]
#     print(df[display_cols].head())

#     out_path = os.path.join(base_dir, "dataset", "processed", "ueba_scores.csv")
#     df.to_csv(out_path, index=False)
#     print(f"Saved -> {out_path}")

import pandas as pd
import os
from scipy.stats import zscore

def run(base_dir):
    print("=" * 50)
    print("STEP 2: UEBA Analysis")
    print("=" * 50)

    print("Loading user behavior dataset...")
    in_path = os.path.join(base_dir, "dataset", "processed", "user_behavior_features.csv")
    df = pd.read_csv(in_path)

    print("Calculating statistical deviations (Z-scores)...")

    # All feature columns to include in anomaly scoring
    feature_cols = [
        "logon_count",
        "logoff_count",
        "after_hours_logon_count",
        "unique_pcs_used",
        "device_connect_count",
        "after_hours_device_count",
        "http_count",
        "logon_without_logoff"
    ]

    # Calculate zscore for each feature
    for col in feature_cols:
        df[f"{col}_zscore"] = zscore(df[col])

    # Combined UEBA anomaly score — sum of absolute Z-scores across all features
    df["ueba_score"] = sum(abs(df[f"{col}_zscore"]) for col in feature_cols)

    # Weighted score — extra weight on high-risk features
    df["ueba_score_weighted"] = (
        abs(df["logon_count_zscore"])              * 1.0 +
        abs(df["after_hours_logon_count_zscore"])  * 2.0 +
        abs(df["unique_pcs_used_zscore"])           * 1.5 +
        abs(df["device_connect_count_zscore"])      * 1.5 +
        abs(df["after_hours_device_count_zscore"])  * 2.5 +
        abs(df["http_count_zscore"])                * 1.0 +
        abs(df["logon_without_logoff_zscore"])      * 2.0
    )

    # Threshold = mean + 1.5 standard deviations
    # Lowered from 2.0 to 1.5 to catch more borderline suspicious users
    # who are clearly anomalous from raw data but were missed at 2.0
    weighted_mean = df["ueba_score_weighted"].mean()
    weighted_std  = df["ueba_score_weighted"].std()
    threshold     = weighted_mean + (1.5 * weighted_std)

    print(f"\nWeighted score stats:")
    print(f"  Mean      : {weighted_mean:.4f}")
    print(f"  Std       : {weighted_std:.4f}")
    print(f"  Threshold : {threshold:.4f}  (mean + 1.5 * std)")

    suspicious = df[df["ueba_score_weighted"] > threshold]
    print(f"\nSuspicious users detected: {len(suspicious)} / {len(df)}")

    # Save threshold into csv so accuracy.py and graph_analysis.py use same value
    df["ueba_threshold"] = threshold

    print("\nUEBA Scores (sample):")
    display_cols = [
        "user", "logon_count", "after_hours_logon_count",
        "device_connect_count", "http_count", "ueba_score_weighted", "ueba_threshold"
    ]
    print(df[display_cols].head())

    out_path = os.path.join(base_dir, "dataset", "processed", "ueba_scores.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")