"""
Insider Threat Detection System
================================
Single entry point - runs the full pipeline in order:
  1. Feature Engineering   (skipped if processed files exist)
  2. UEBA Analysis         (skipped if processed files exist)
  3. Sequence Building     (skipped if processed files exist)
  4. LSTM Model Training   (skipped if model exists)
  5. Graph Analysis
  6. Accuracy Evaluation
  7. Rule-Based Detection  (always runs)
  8. Unified Risk Scoring  (always runs)
  9. Email Alert           (always runs, skipped if .env not configured)

Run from the project root:
    python main.py             # skips steps that are already done
    python main.py --retrain   # forces retraining and reprocessing everything
"""

import os
import sys

# -- Suppress TensorFlow logs before anything is imported -------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import argparse

# -- Resolve project root ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Parse arguments --------------------------------------------------------
parser = argparse.ArgumentParser(description="Insider Threat Detection Pipeline")
parser.add_argument(
    "--retrain",
    action="store_true",
    help="Force reprocess and retrain everything from scratch"
)
args = parser.parse_args()

# -- Create required folders ------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "dataset", "processed"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# -- Verify raw dataset files -----------------------------------------------
required_files = [
    os.path.join(BASE_DIR, "dataset", "logon.csv"),
    os.path.join(BASE_DIR, "dataset", "device.csv"),
    os.path.join(BASE_DIR, "dataset", "http.csv"),
]

print("\nChecking dataset files...")
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print("\nERROR: Missing dataset files:")
    for f in missing:
        print(f"   - {f}")
    print("\nPlease place logon.csv, device.csv, and http.csv inside the dataset/ folder.")
    sys.exit(1)
print("All dataset files found.\n")

# -- File paths to check before skipping steps ------------------------------
features_csv  = os.path.join(BASE_DIR, "dataset", "processed", "user_behavior_features.csv")
ueba_csv      = os.path.join(BASE_DIR, "dataset", "processed", "ueba_scores.csv")
sequences_csv = os.path.join(BASE_DIR, "dataset", "processed", "lstm_sequences.csv")

# Check for either .keras or .h5 model
model_keras  = os.path.join(BASE_DIR, "models", "insider_threat_lstm.keras")
model_h5     = os.path.join(BASE_DIR, "models", "insider_threat_lstm.h5")
model_exists = os.path.exists(model_keras) or os.path.exists(model_h5)

# -- Import pipeline modules ------------------------------------------------
from src import (
    feature_engineering,
    ueba_analysis,
    sequence_builder,
    train_lstm,
    graph_analysis,
    accuracy,
    rule_engine,
    risk_scorer,
    email_alert
)

# -- Run pipeline -----------------------------------------------------------
print("=" * 50)
print("   INSIDER THREAT DETECTION - FULL PIPELINE")
print("=" * 50 + "\n")

try:
    # STEP 1 - Feature Engineering
    if args.retrain or not os.path.exists(features_csv):
        feature_engineering.run(BASE_DIR)
    else:
        print("=" * 50)
        print("STEP 1: Feature Engineering")
        print("=" * 50)
        print("Processed file already exists, skipping.")
        print(f"File: {features_csv}")
    print()

    # STEP 2 - UEBA Analysis
    if args.retrain or not os.path.exists(ueba_csv):
        ueba_analysis.run(BASE_DIR)
    else:
        print("=" * 50)
        print("STEP 2: UEBA Analysis")
        print("=" * 50)
        print("Processed file already exists, skipping.")
        print(f"File: {ueba_csv}")
    print()

    # STEP 3 - Sequence Building
    if args.retrain or not os.path.exists(sequences_csv):
        sequence_builder.run(BASE_DIR)
    else:
        print("=" * 50)
        print("STEP 3: Sequence Builder")
        print("=" * 50)
        print("Processed file already exists, skipping.")
        print(f"File: {sequences_csv}")
    print()

    # STEP 4 - LSTM Training
    if args.retrain or not model_exists:
        if args.retrain and model_exists:
            print("Retrain flag detected - retraining model from scratch...")
        train_lstm.run(BASE_DIR)
    else:
        print("=" * 50)
        print("STEP 4: LSTM Model Training")
        print("=" * 50)
        print("Trained model already exists, skipping.")
        print("To force retrain, run:  python main.py --retrain")
    print()

    # STEP 5 - Graph Analysis
    graph_analysis.run(BASE_DIR)
    print()

    # STEP 6 - Accuracy Evaluation
    accuracy.run(BASE_DIR)
    print()

    # STEP 7 - Rule-Based Detection (always runs)
    rule_engine.run(BASE_DIR)
    print()

    # STEP 8 - Unified Risk Scoring (always runs)
    risk_scorer.run(BASE_DIR)
    print()

    # STEP 9 - Email Alert (always runs, gracefully skips if not configured)
    email_alert.run(BASE_DIR)
    print()

    print("=" * 50)
    print("Pipeline complete.")
    print("=" * 50)

except Exception as e:
    print(f"\nERROR: Pipeline failed: {e}")
    raise