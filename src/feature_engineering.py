import pandas as pd
import os

def run(base_dir):
    print("=" * 50)
    print("STEP 1: Feature Engineering")
    print("=" * 50)

    print("Loading datasets...")
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

    print("Creating behavioral features...")

    # -- Logon features -----------------------------------------------------

    # Only actual Logon events (not Logoff)
    logon_only  = logon[logon["activity"] == "Logon"]
    logoff_only = logon[logon["activity"] == "Logoff"]

    logon_counts  = logon_only.groupby("user").size().reset_index(name="logon_count")
    logoff_counts = logoff_only.groupby("user").size().reset_index(name="logoff_count")

    # After-hours logons: before 8am or after 6pm
    after_hours_logon = logon_only[
        (logon_only["date"].dt.hour < 8) | (logon_only["date"].dt.hour >= 18)
    ]
    after_hours_counts = after_hours_logon.groupby("user").size().reset_index(name="after_hours_logon_count")

    # Unique PCs used per user
    unique_pcs = logon_only.groupby("user")["pc"].nunique().reset_index(name="unique_pcs_used")

    # -- Device features ----------------------------------------------------

    # Only Connect events (not Disconnect)
    connect_only = device[device["activity"] == "Connect"]
    device_counts = connect_only.groupby("user").size().reset_index(name="device_connect_count")

    # After-hours device connections
    after_hours_device = connect_only[
        (connect_only["date"].dt.hour < 8) | (connect_only["date"].dt.hour >= 18)
    ]
    after_hours_device_counts = after_hours_device.groupby("user").size().reset_index(
        name="after_hours_device_count"
    )

    # -- HTTP features ------------------------------------------------------
    http_counts = http.groupby("user").size().reset_index(name="http_count")

    # -- Merge all features -------------------------------------------------
    features = logon_counts
    features = features.merge(logoff_counts,              on="user", how="outer")
    features = features.merge(after_hours_counts,         on="user", how="outer")
    features = features.merge(unique_pcs,                 on="user", how="outer")
    features = features.merge(device_counts,              on="user", how="outer")
    features = features.merge(after_hours_device_counts,  on="user", how="outer")
    features = features.merge(http_counts,                on="user", how="outer")
    features = features.fillna(0)

    # Logons without a corresponding logoff (sessions left open - suspicious)
    features["logon_without_logoff"] = (
        features["logon_count"] - features["logoff_count"]
    ).clip(lower=0)

    print("\nBehavioral Features (sample):")
    print(features.head())
    print("\nTotal Users:", len(features))
    print("\nFeature columns:", list(features.columns))

    out_path = os.path.join(base_dir, "dataset", "processed", "user_behavior_features.csv")
    features.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")