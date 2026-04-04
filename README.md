# UEBA Insider Threat Detection System

A hybrid cybersecurity platform that detects insider threats by analyzing user behavior across authentication logs, device usage, and web activity. The system combines three independent detection layers — statistical UEBA analysis, LSTM autoencoder deep learning, and a dynamic rule-based engine — into a unified risk score per user, visualized through a real-time SOC dashboard with automated email alerts.

---

## Project Structure

```
final-year-project/
│
├── dataset/
│   ├── logon.csv                  <- raw input: login/logout events
│   ├── device.csv                 <- raw input: USB device events
│   ├── http.csv                   <- raw input: web browsing events
│   └── processed/                 <- all generated files (auto-created)
│       ├── user_behavior_features.csv
│       ├── ueba_scores.csv
│       ├── lstm_sequences.csv
│       ├── lstm_sequences_normal.csv
│       ├── test_indices.npy
│       ├── accuracy_results.csv
│       ├── rule_scores.csv
│       ├── final_risk_scores.csv
│       ├── graph_scores.csv
│       └── network_graph.png
│
├── models/
│   └── insider_threat_lstm.keras  <- trained model (auto-created)
│
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py     <- Step 1: extract behavioral features
│   ├── ueba_analysis.py           <- Step 2: z-score anomaly scoring
│   ├── sequence_builder.py        <- Step 3: build LSTM sequences
│   ├── train_lstm.py              <- Step 4: train LSTM autoencoder
│   ├── graph_analysis.py          <- Step 5: network graph analysis
│   ├── accuracy.py                <- Step 6: model evaluation
│   ├── rule_engine.py             <- Step 7: dynamic rule-based detection
│   ├── risk_scorer.py             <- Step 8: unified risk scoring
│   └── email_alert.py             <- Step 9: email alert dispatch
│
├── main.py                        <- single pipeline entry point
├── dashboard.py                   <- Streamlit SOC dashboard
├── requirements.txt
├── .env                           <- email credentials (not committed)
├── .env.example                   <- credential template
└── .gitignore
```

---

## Detection Architecture

```
Raw Logs (logon, device, http)
             |
             v
   Feature Engineering (8 features per user)
             |
      -------+--------+
      |               |
      v               v
 UEBA Analysis    Sequence Builder
 (z-score)        (sliding window)
      |               |
      |               v
      |          LSTM Autoencoder
      |          (normal-only training)
      |               |
      +-------+--------+
              |
              v
       Rule-Based Engine
       (dynamic thresholds)
              |
              v
     Unified Risk Scorer
     (LSTM 40% + UEBA 35% + Rules 25%)
              |
         -----+-----
         |         |
         v         v
    SOC Dashboard  Email Alert
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset files

```
dataset/logon.csv
dataset/device.csv
dataset/http.csv
```

### 3. Configure email alerts (optional)

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
ALERT_SENDER_EMAIL=your_gmail@gmail.com
ALERT_SENDER_PASSWORD=your_16_char_app_password
ALERT_RECEIVER_EMAIL=admin@yourorg.com
```

To generate a Gmail App Password:
`Google Account → Security → 2-Step Verification → App Passwords`

### 4. Run the pipeline

```bash
# First run (trains the model — takes ~35-40 minutes)
python main.py

# Subsequent runs (skips training — takes ~2-3 minutes)
python main.py

# Force full reprocessing and retraining
python main.py --retrain
```

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

---

## Pipeline Steps

| Step | Module                 | Description                                           | Always Runs            |
| ---- | ---------------------- | ----------------------------------------------------- | ---------------------- |
| 1    | feature_engineering.py | Extracts 8 behavioral features per user               | Only if output missing |
| 2    | ueba_analysis.py       | Z-score based anomaly scoring with weighted features  | Only if output missing |
| 3    | sequence_builder.py    | Builds time-aware sliding window sequences for LSTM   | Only if output missing |
| 4    | train_lstm.py          | Trains LSTM autoencoder on normal users only          | Only if model missing  |
| 5    | graph_analysis.py      | Builds user-PC-website relationship network graph     | Yes                    |
| 6    | accuracy.py            | Evaluates model on held-out test set                  | Yes                    |
| 7    | rule_engine.py         | Applies dynamic threshold rules with severity scoring | Yes                    |
| 8    | risk_scorer.py         | Combines all three scores into unified risk per user  | Yes                    |
| 9    | email_alert.py         | Sends threat report email to configured admin         | Yes (if .env set)      |

---

## Detection Methods

### UEBA Analysis

Statistical z-score analysis across 8 behavioral features. Features are weighted by risk level — after-hours activity (weight 2.0-2.5) is treated as more suspicious than raw counts (weight 1.0). Threshold is dynamically calculated as `mean + 1.5 * std` of the weighted score.

### LSTM Autoencoder

Trained exclusively on sequences from normal users using a 5-event sliding window with time-aware encoding (5 event types: normal logon, after-hours logon, normal device connect, after-hours device connect, http visit). When shown sequences from suspicious users it has never seen, it fails to reconstruct them accurately, producing high reconstruction error that triggers an alert.

### Rule-Based Engine

Seven dynamic rules applied to behavioral features. Every threshold is computed from the data distribution (`mean + multiplier * std`) — nothing is hardcoded. Violations are severity-scored based on how far the user exceeds the threshold (LOW/MEDIUM/HIGH) and weighted by rule risk level.

### Unified Risk Score

```
final_score = (lstm_normalized * 0.40) + (ueba_normalized * 0.35) + (rule_normalized * 0.25)
```

| Level  | Score Range | Action                  |
| ------ | ----------- | ----------------------- |
| HIGH   | > 0.60      | Immediate investigation |
| MEDIUM | 0.30 - 0.60 | Monitor closely         |
| LOW    | <= 0.30     | Normal behavior         |

---

## Model Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 95.70% |
| Precision | 64.71% |
| Recall    | 41.51% |
| F1 Score  | 50.57% |

Evaluated on a held-out test set (20% of all sequences) using UEBA scores as ground truth labels. The model was trained on normal users only, ensuring it has never seen suspicious behavior during training.

---

## Dataset

Uses the **CERT Insider Threat Dataset r1** from Carnegie Mellon University's Software Engineering Institute — a standard benchmark dataset for insider threat research.

| File       | Events | Description                                       |
| ---------- | ------ | ------------------------------------------------- |
| logon.csv  | ~500K  | Login/logout events (Logon/Logoff activities)     |
| device.csv | ~65K   | USB device events (Connect/Disconnect activities) |
| http.csv   | ~3.4M  | Web browsing events (URL visits)                  |

1000 unique users across all three files.

---

## Dashboard Features

- **Overview tab** — risk distribution chart, detection method coverage, top 15 risk users
- **User Risk Table** — searchable and filterable table of all 1000 users with risk scores
- **User Investigation** — select any user to see full breakdown: gauge meters for each score, activity profile, and rule violation explanations with severity levels
- **System Analytics** — model metrics, confusion matrix, network graph visualization

---

## Requirements

```
pandas
numpy
tensorflow
scikit-learn
networkx
matplotlib
scipy
streamlit
plotly
pillow
```

Install: `pip install -r requirements.txt`

---

## Security Notes

- Never commit `.env` to version control — it is listed in `.gitignore`
- Use Gmail App Passwords, not your regular Gmail password
- The `.env.example` file shows the required format with no real credentials
