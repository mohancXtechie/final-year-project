# import pandas as pd
# import numpy as np
# import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.model_selection import train_test_split

# def run(base_dir):
#     print("=" * 50)
#     print("STEP 4: LSTM Model Training")
#     print("=" * 50)

#     print("Loading sequence dataset...")
#     in_path = os.path.join(base_dir, "dataset", "processed", "lstm_sequences.csv")
#     data = pd.read_csv(in_path)

#     X = data.values
#     print("Dataset shape:", X.shape)

#     # Split using indices so we can save the test indices for evaluation
#     all_indices = np.arange(len(X))
#     train_indices, test_indices = train_test_split(
#         all_indices, test_size=0.2, random_state=42
#     )

#     X_train = X[train_indices].reshape((len(train_indices), X.shape[1], 1))
#     X_test  = X[test_indices].reshape((len(test_indices), X.shape[1], 1))

#     print(f"Training samples : {len(X_train)}")
#     print(f"Test samples     : {len(X_test)}")

#     # Save test indices so accuracy.py can evaluate on the same held-out set
#     test_indices_path = os.path.join(base_dir, "dataset", "processed", "test_indices.npy")
#     np.save(test_indices_path, test_indices)
#     print(f"Test indices saved -> {test_indices_path}")

#     print("Building LSTM autoencoder model...")
#     model = Sequential([
#         LSTM(64, input_shape=(X.shape[1], 1), return_sequences=True),
#         LSTM(32),
#         Dense(X.shape[1])
#     ])

#     model.compile(optimizer="adam", loss="mse")
#     model.summary()

#     print("\nTraining model...")
#     model.fit(
#         X_train, X_train,
#         epochs=5,
#         batch_size=64,
#         validation_data=(X_test, X_test)
#     )

#     out_path = os.path.join(base_dir, "models", "insider_threat_lstm.keras")
#     model.save(out_path)
#     print(f"\nModel saved -> {out_path}")

import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

def run(base_dir):
    print("=" * 50)
    print("STEP 4: LSTM Model Training")
    print("=" * 50)

    # -- Load NORMAL-ONLY sequences for training ----------------------------
    normal_path = os.path.join(base_dir, "dataset", "processed", "lstm_sequences_normal.csv")
    if not os.path.exists(normal_path):
        print("ERROR: lstm_sequences_normal.csv not found.")
        print("Please run sequence builder first.")
        return

    print("Loading normal-user sequences for training...")
    data = pd.read_csv(normal_path)
    X = data.values
    print(f"Training data shape: {X.shape}")
    print("Note: Training on NORMAL users only — suspicious users excluded.")

    # 80/20 split within normal sequences for validation during training
    all_indices = np.arange(len(X))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.2, random_state=42
    )

    X_train = X[train_indices].reshape((len(train_indices), X.shape[1], 1))
    X_val   = X[val_indices].reshape((len(val_indices), X.shape[1], 1))

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")

    # -- Save test indices from ALL sequences for evaluation ----------------
    # accuracy.py evaluates on ALL sequences (normal + suspicious)
    all_path = os.path.join(base_dir, "dataset", "processed", "lstm_sequences.csv")
    all_data = pd.read_csv(all_path)
    all_X    = all_data.drop(columns=["user"]).values

    all_seq_indices = np.arange(len(all_X))
    _, test_indices = train_test_split(
        all_seq_indices, test_size=0.2, random_state=42
    )

    test_indices_path = os.path.join(base_dir, "dataset", "processed", "test_indices.npy")
    np.save(test_indices_path, test_indices)
    print(f"Test indices saved -> {test_indices_path}")
    print(f"Test set size      : {len(test_indices)} sequences from ALL users")

    # -- Build and train model ----------------------------------------------
    print("\nBuilding LSTM autoencoder model...")
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], 1), return_sequences=True),
        LSTM(32),
        Dense(X.shape[1])
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    print("\nTraining model on normal user sequences only...")
    model.fit(
        X_train, X_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_val, X_val)
    )

    out_path = os.path.join(base_dir, "models", "insider_threat_lstm.keras")
    model.save(out_path)
    print(f"\nModel saved -> {out_path}")