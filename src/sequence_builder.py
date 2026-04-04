# import pandas as pd
# import os

# def run(base_dir):
#     print("=" * 50)
#     print("STEP 3: Sequence Builder")
#     print("=" * 50)

#     print("Loading activity logs...")
#     logon  = pd.read_csv(os.path.join(base_dir, "dataset", "logon.csv"))
#     device = pd.read_csv(os.path.join(base_dir, "dataset", "device.csv"))
#     http   = pd.read_csv(
#         os.path.join(base_dir, "dataset", "http.csv"),
#         header=None,
#         names=["id", "date", "user", "pc", "url"]
#     )

#     print("Parsing timestamps...")
#     logon["date"]  = pd.to_datetime(logon["date"])
#     device["date"] = pd.to_datetime(device["date"])
#     http["date"]   = pd.to_datetime(http["date"])

#     # After hours = before 8am or at/after 6pm
#     def is_after_hours(dt_series):
#         return (dt_series.dt.hour < 8) | (dt_series.dt.hour >= 18)

#     print("Encoding events with time context...")

#     # -- Logon events -------------------------------------------------------
#     # 0 = normal logon, 1 = after-hours logon
#     logon["event_code"] = 0
#     logon.loc[is_after_hours(logon["date"]), "event_code"] = 1

#     # -- Device events ------------------------------------------------------
#     # Only Connect events (not Disconnect — disconnects are not suspicious)
#     # 2 = normal device connect, 3 = after-hours device connect
#     device_connect = device[device["activity"] == "Connect"].copy()
#     device_connect["event_code"] = 2
#     device_connect.loc[is_after_hours(device_connect["date"]), "event_code"] = 3

#     # -- HTTP events --------------------------------------------------------
#     # 4 = http visit (timing not tracked in this dataset)
#     http["event_code"] = 4

#     # -- Combine all events -------------------------------------------------
#     logon_events  = logon[["user", "date", "event_code"]]
#     device_events = device_connect[["user", "date", "event_code"]]
#     http_events   = http[["user", "date", "event_code"]]

#     events = pd.concat([logon_events, device_events, http_events])
#     events = events.sort_values(["user", "date"]).reset_index(drop=True)

#     print("\nEvent encoding used:")
#     print("  0 = logon (normal hours)")
#     print("  1 = logon (after hours)  <- suspicious")
#     print("  2 = device connect (normal hours)")
#     print("  3 = device connect (after hours)  <- suspicious")
#     print("  4 = http visit")

#     print("\nEvent code distribution:")
#     print(events["event_code"].value_counts().sort_index())

#     print("\nBuilding sliding-window sequences...")
#     window_size = 5
#     sequences = []

#     for user, group in events.groupby("user"):
#         event_list = group["event_code"].tolist()
#         for i in range(len(event_list) - window_size):
#             seq = event_list[i : i + window_size]
#             sequences.append(seq)

#     print(f"Total sequences built: {len(sequences)}")

#     seq_df = pd.DataFrame(sequences)
#     out_path = os.path.join(base_dir, "dataset", "processed", "lstm_sequences.csv")
#     seq_df.to_csv(out_path, index=False)
#     print(f"Saved -> {out_path}")
import pandas as pd
import os

def run(base_dir):
    print("=" * 50)
    print("STEP 3: Sequence Builder")
    print("=" * 50)

    print("Loading activity logs...")
    logon  = pd.read_csv(os.path.join(base_dir, "dataset", "logon.csv"))
    device = pd.read_csv(os.path.join(base_dir, "dataset", "device.csv"))
    http   = pd.read_csv(
        os.path.join(base_dir, "dataset", "http.csv"),
        header=None,
        names=["id", "date", "user", "pc", "url"]
    )

    print("Parsing timestamps...")
    logon["date"]  = pd.to_datetime(logon["date"])
    device["date"] = pd.to_datetime(device["date"])
    http["date"]   = pd.to_datetime(http["date"])

    # After hours = before 8am or at/after 6pm
    def is_after_hours(dt_series):
        return (dt_series.dt.hour < 8) | (dt_series.dt.hour >= 18)

    print("Encoding events with time context...")

    # 0 = normal logon, 1 = after-hours logon
    logon["event_code"] = 0
    logon.loc[is_after_hours(logon["date"]), "event_code"] = 1

    # Only Connect events — 2 = normal, 3 = after-hours
    device_connect = device[device["activity"] == "Connect"].copy()
    device_connect["event_code"] = 2
    device_connect.loc[is_after_hours(device_connect["date"]), "event_code"] = 3

    # 4 = http visit
    http["event_code"] = 4

    logon_events  = logon[["user", "date", "event_code"]]
    device_events = device_connect[["user", "date", "event_code"]]
    http_events   = http[["user", "date", "event_code"]]

    events = pd.concat([logon_events, device_events, http_events])
    events = events.sort_values(["user", "date"]).reset_index(drop=True)

    print("\nEvent encoding used:")
    print("  0 = logon (normal hours)")
    print("  1 = logon (after hours)  <- suspicious")
    print("  2 = device connect (normal hours)")
    print("  3 = device connect (after hours)  <- suspicious")
    print("  4 = http visit")

    print("\nEvent code distribution:")
    print(events["event_code"].value_counts().sort_index())

    # -- Load UEBA scores to identify normal vs suspicious users ------------
    ueba_path = os.path.join(base_dir, "dataset", "processed", "ueba_scores.csv")
    ueba = pd.read_csv(ueba_path)[["user", "ueba_score_weighted", "ueba_threshold"]]
    threshold_val  = ueba["ueba_threshold"].iloc[0]
    suspicious_users = set(ueba[ueba["ueba_score_weighted"] > threshold_val]["user"])
    normal_users     = set(ueba[ueba["ueba_score_weighted"] <= threshold_val]["user"])

    print(f"\nNormal users     : {len(normal_users)}")
    print(f"Suspicious users : {len(suspicious_users)}")
    print(f"UEBA threshold   : {threshold_val:.4f}")

    # -- Build sequences with user label ------------------------------------
    print("\nBuilding sliding-window sequences...")
    window_size = 5
    all_sequences  = []
    all_users      = []

    for user, group in events.groupby("user"):
        event_list = group["event_code"].tolist()
        for i in range(len(event_list) - window_size):
            seq = event_list[i : i + window_size]
            all_sequences.append(seq)
            all_users.append(user)

    print(f"Total sequences built: {len(all_sequences)}")

    # -- Save ALL sequences (used for evaluation in accuracy.py) ------------
    seq_df = pd.DataFrame(all_sequences)
    seq_df["user"] = all_users
    all_path = os.path.join(base_dir, "dataset", "processed", "lstm_sequences.csv")
    seq_df.to_csv(all_path, index=False)
    print(f"All sequences saved -> {all_path}")

    # -- Save NORMAL-ONLY sequences (used for training in train_lstm.py) ----
    normal_mask    = [u in normal_users for u in all_users]
    normal_seq_df  = seq_df[normal_mask].drop(columns=["user"])
    normal_path    = os.path.join(base_dir, "dataset", "processed", "lstm_sequences_normal.csv")
    normal_seq_df.to_csv(normal_path, index=False)
    print(f"Normal-only sequences saved -> {normal_path}")
    print(f"Normal sequences : {len(normal_seq_df)}")
    print(f"Suspicious sequences (excluded from training): {len(seq_df) - len(normal_seq_df)}")