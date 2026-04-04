# import os
# import sys
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix
# )

# def run(base_dir):
#     print("=" * 50)
#     print("STEP 6: Model Accuracy Evaluation")
#     print("=" * 50)

#     # -- Load model ---------------------------------------------------------
#     model_path = os.path.join(base_dir, "models", "insider_threat_lstm.keras")
#     if not os.path.exists(model_path):
#         model_path = os.path.join(base_dir, "models", "insider_threat_lstm.h5")
#         if not os.path.exists(model_path):
#             print("ERROR: No trained model found in models/")
#             print("Please run:  python main.py")
#             sys.exit(1)

#     print(f"Loading model from: {model_path}")
#     model = tf.keras.models.load_model(model_path, compile=False)
#     print("Model loaded.\n")

#     # -- Load sequences -----------------------------------------------------
#     seq_path = os.path.join(base_dir, "dataset", "processed", "lstm_sequences.csv")
#     if not os.path.exists(seq_path):
#         print("ERROR: lstm_sequences.csv not found.")
#         print("Please run:  python main.py")
#         sys.exit(1)

#     print("Loading sequences...")
#     seq_df = pd.read_csv(seq_path)
#     X_all = seq_df.values
#     print(f"Total sequences loaded: {len(X_all)}")

#     # -- Load test indices --------------------------------------------------
#     test_indices_path = os.path.join(base_dir, "dataset", "processed", "test_indices.npy")

#     if os.path.exists(test_indices_path):
#         test_indices = np.load(test_indices_path)
#         X = X_all[test_indices]
#         print(f"Evaluating on held-out test set: {len(X)} sequences (20% of total)")
#         print("Note: These sequences were NOT seen during training.\n")
#     else:
#         X = X_all
#         test_indices = np.arange(len(X_all))
#         print("WARNING: test_indices.npy not found.")
#         print("Evaluating on full dataset (includes training data).")
#         print("For proper evaluation, run:  python main.py --retrain\n")

#     X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

#     # -- Rebuild user-to-sequence mapping -----------------------------------
#     # Must match sequence_builder.py exactly: same filtering, same sort order
#     print("Loading activity logs to map sequences back to users...")
#     logon  = pd.read_csv(os.path.join(base_dir, "dataset", "logon.csv"))
#     device = pd.read_csv(os.path.join(base_dir, "dataset", "device.csv"))
#     http   = pd.read_csv(
#         os.path.join(base_dir, "dataset", "http.csv"),
#         header=None,
#         names=["id", "date", "user", "pc", "url"]
#     )

#     logon["date"]  = pd.to_datetime(logon["date"])
#     device["date"] = pd.to_datetime(device["date"])
#     http["date"]   = pd.to_datetime(http["date"])

#     # Only Connect events — same as sequence_builder.py
#     device_connect = device[device["activity"] == "Connect"].copy()

#     events = pd.concat([
#         logon[["user", "date"]],
#         device_connect[["user", "date"]],
#         http[["user", "date"]]
#     ])
#     events = events.sort_values(["user", "date"]).reset_index(drop=True)

#     # Build full user list — one entry per sequence per user
#     window_size = 5
#     all_sequence_users = []
#     for user, group in events.groupby("user"):
#         n_seqs = len(group) - window_size
#         if n_seqs > 0:
#             all_sequence_users.extend([user] * n_seqs)

#     all_sequence_users  = np.array(all_sequence_users)
#     test_sequence_users = all_sequence_users[test_indices]
#     print(f"Sequence-to-user mapping built for test set: {len(test_sequence_users)} entries\n")

#     # -- Run predictions ----------------------------------------------------
#     print("Running model predictions on test set...")
#     X_pred = model.predict(X_reshaped, batch_size=512, verbose=1)

#     # -- Reconstruction error per sequence ----------------------------------
#     mse_per_seq = np.mean(np.power(X - X_pred, 2), axis=1)

#     # -- Average error per user ---------------------------------------------
#     user_errors = pd.DataFrame({
#         "user" : test_sequence_users,
#         "error": mse_per_seq
#     })
#     user_avg_error = user_errors.groupby("user")["error"].mean().reset_index()
#     user_avg_error.columns = ["user", "avg_reconstruction_error"]

#     # -- LSTM anomaly threshold: mean + 2 std -------------------------------
#     mean_err  = user_avg_error["avg_reconstruction_error"].mean()
#     std_err   = user_avg_error["avg_reconstruction_error"].std()
#     threshold = mean_err + (2 * std_err)

#     user_avg_error["lstm_anomaly"] = (
#         user_avg_error["avg_reconstruction_error"] > threshold
#     ).astype(int)

#     # -- Load UEBA ground truth — threshold stored in csv ------------------
#     ueba_path = os.path.join(base_dir, "dataset", "processed", "ueba_scores.csv")
#     if not os.path.exists(ueba_path):
#         print("ERROR: ueba_scores.csv not found.")
#         print("Please run:  python main.py")
#         sys.exit(1)

#     ueba = pd.read_csv(ueba_path)[["user", "ueba_score_weighted", "ueba_threshold"]]
#     ueba_threshold_val = ueba["ueba_threshold"].iloc[0]
#     ueba["ueba_anomaly"] = (ueba["ueba_score_weighted"] > ueba_threshold_val).astype(int)
#     print(f"UEBA threshold used: {ueba_threshold_val:.4f}")

#     # -- Merge on user ------------------------------------------------------
#     merged = user_avg_error.merge(ueba, on="user", how="inner")
#     print(f"Users evaluated: {len(merged)}")

#     y_true = merged["ueba_anomaly"].values
#     y_pred = merged["lstm_anomaly"].values

#     # -- Metrics ------------------------------------------------------------
#     acc       = accuracy_score(y_true, y_pred) * 100
#     precision = precision_score(y_true, y_pred, zero_division=0) * 100
#     recall    = recall_score(y_true, y_pred, zero_division=0) * 100
#     f1        = f1_score(y_true, y_pred, zero_division=0) * 100
#     cm        = confusion_matrix(y_true, y_pred)

#     print("\n" + "=" * 50)
#     print("RESULTS  (evaluated on held-out test set)")
#     print("=" * 50)
#     print(f"Accuracy   : {acc:.2f}%")
#     print(f"Precision  : {precision:.2f}%")
#     print(f"Recall     : {recall:.2f}%")
#     print(f"F1 Score   : {f1:.2f}%")
#     print(f"\nAnomaly threshold (LSTM)      : {threshold:.8f}")
#     print(f"LSTM anomalies detected       : {y_pred.sum()} / {len(y_pred)} users")
#     print(f"UEBA anomalies (ground truth) : {y_true.sum()} / {len(y_true)} users")
#     print("\nConfusion Matrix:")
#     print(f"                  Predicted Normal  Predicted Anomaly")
#     print(f"Actual Normal   : {cm[0][0]:<18} {cm[0][1]}")
#     print(f"Actual Anomaly  : {cm[1][0]:<18} {cm[1][1]}")

#     # -- Save results -------------------------------------------------------
#     out_path = os.path.join(base_dir, "dataset", "processed", "accuracy_results.csv")
#     merged.to_csv(out_path, index=False)
#     print(f"\nDetailed results saved -> {out_path}")
#     print("=" * 50)

# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     run(BASE_DIR)

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def run(base_dir):
    print("=" * 50)
    print("STEP 6: Model Accuracy Evaluation")
    print("=" * 50)

    # -- Load model ---------------------------------------------------------
    model_path = os.path.join(base_dir, "models", "insider_threat_lstm.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models", "insider_threat_lstm.h5")
        if not os.path.exists(model_path):
            print("ERROR: No trained model found in models/")
            print("Please run:  python main.py")
            sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded.\n")

    # -- Load ALL sequences (normal + suspicious) ---------------------------
    seq_path = os.path.join(base_dir, "dataset", "processed", "lstm_sequences.csv")
    if not os.path.exists(seq_path):
        print("ERROR: lstm_sequences.csv not found.")
        print("Please run:  python main.py")
        sys.exit(1)

    print("Loading sequences...")
    seq_df = pd.read_csv(seq_path)

    # user column is saved alongside sequences — extract before converting to array
    all_users = seq_df["user"].values
    X_all     = seq_df.drop(columns=["user"]).values
    print(f"Total sequences loaded: {len(X_all)}")

    # -- Load test indices --------------------------------------------------
    test_indices_path = os.path.join(base_dir, "dataset", "processed", "test_indices.npy")

    if os.path.exists(test_indices_path):
        test_indices = np.load(test_indices_path)
        X            = X_all[test_indices]
        test_users   = all_users[test_indices]
        print(f"Evaluating on held-out test set: {len(X)} sequences (20% of total)")
        print("Note: Model was trained on NORMAL users only.\n")
    else:
        X            = X_all
        test_indices = np.arange(len(X_all))
        test_users   = all_users
        print("WARNING: test_indices.npy not found.")
        print("Evaluating on full dataset.")
        print("For proper evaluation, run:  python main.py --retrain\n")

    X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

    # -- Run predictions ----------------------------------------------------
    print("Running model predictions on test set...")
    X_pred = model.predict(X_reshaped, batch_size=512, verbose=1)

    # -- Reconstruction error per sequence ----------------------------------
    mse_per_seq = np.mean(np.power(X - X_pred, 2), axis=1)

    # -- Average error per user ---------------------------------------------
    user_errors = pd.DataFrame({
        "user" : test_users,
        "error": mse_per_seq
    })
    user_avg_error = user_errors.groupby("user")["error"].mean().reset_index()
    user_avg_error.columns = ["user", "avg_reconstruction_error"]

    # -- LSTM anomaly threshold: mean + 2 std -------------------------------
    mean_err  = user_avg_error["avg_reconstruction_error"].mean()
    std_err   = user_avg_error["avg_reconstruction_error"].std()
    threshold = mean_err + (2 * std_err)

    user_avg_error["lstm_anomaly"] = (
        user_avg_error["avg_reconstruction_error"] > threshold
    ).astype(int)

    # -- Load UEBA ground truth ---------------------------------------------
    ueba_path = os.path.join(base_dir, "dataset", "processed", "ueba_scores.csv")
    if not os.path.exists(ueba_path):
        print("ERROR: ueba_scores.csv not found.")
        print("Please run:  python main.py")
        sys.exit(1)

    ueba = pd.read_csv(ueba_path)[["user", "ueba_score_weighted", "ueba_threshold"]]
    ueba_threshold_val = ueba["ueba_threshold"].iloc[0]
    ueba["ueba_anomaly"] = (ueba["ueba_score_weighted"] > ueba_threshold_val).astype(int)
    print(f"UEBA threshold used : {ueba_threshold_val:.4f}")

    # -- Merge on user ------------------------------------------------------
    merged = user_avg_error.merge(ueba, on="user", how="inner")
    print(f"Users evaluated     : {len(merged)}")

    y_true = merged["ueba_anomaly"].values
    y_pred = merged["lstm_anomaly"].values

    # -- Metrics ------------------------------------------------------------
    acc       = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, zero_division=0) * 100
    cm        = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("RESULTS  (model trained on normal users only)")
    print("=" * 50)
    print(f"Accuracy   : {acc:.2f}%")
    print(f"Precision  : {precision:.2f}%")
    print(f"Recall     : {recall:.2f}%")
    print(f"F1 Score   : {f1:.2f}%")
    print(f"\nAnomaly threshold (LSTM)      : {threshold:.8f}")
    print(f"LSTM anomalies detected       : {y_pred.sum()} / {len(y_pred)} users")
    print(f"UEBA anomalies (ground truth) : {y_true.sum()} / {len(y_true)} users")
    print("\nConfusion Matrix:")
    print(f"                  Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal   : {cm[0][0]:<18} {cm[0][1]}")
    print(f"Actual Anomaly  : {cm[1][0]:<18} {cm[1][1]}")

    # -- Save results -------------------------------------------------------
    out_path = os.path.join(base_dir, "dataset", "processed", "accuracy_results.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nDetailed results saved -> {out_path}")
    print("=" * 50)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(BASE_DIR)