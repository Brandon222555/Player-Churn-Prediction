# Player Churn Prediction — Gaming ML Project

An advanced end-to-end data science project predicting which players will stop playing within 7 days, using behavioral telemetry, RFM analysis, K-Means player segmentation, SHAP explainability, and a live ops Streamlit dashboard.

---

## Project Overview

| Item | Detail |
|---|---|
| Dataset | Online Gaming Behavior Dataset (Kaggle, free) |
| Players | 40,000+ records |
| Features | 13 base + ~30 engineered |
| Target | Player churn (binary) |
| Churn rate | ~26% |
| Best AUC | 0.847 (LightGBM) |

---

## Why This Project Matters

Player churn is the #1 data science problem at every major game studio — Riot Games, Epic, EA, Activision, Ubisoft. Retaining a player is 5–10x cheaper than acquiring a new one. This project demonstrates every skill a gaming data team actually uses.

---

## Project Structure

```
player-churn-prediction/
├── data/                        # Download from Kaggle (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_rfm_features.ipynb    # RFM feature engineering
│   └── 03_modeling_shap.ipynb   # Modeling + SHAP
├── src/
│   ├── features.py              # RFM + behavioral feature engineering
│   ├── train.py                 # Full training pipeline
│   ├── segment.py               # K-Means player segmentation
│   └── ab_test.py               # A/B test statistical framework
├── models/                      # Saved models (gitignored)
├── app.py                       # Streamlit live ops dashboard
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/player-churn-prediction.git
cd player-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Go to: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
Download and place `online_gaming_behavior_dataset.csv` in the `data/` folder.

### 4. Train the model
```bash
python src/train.py
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## Advanced Techniques Used

### RFM Analysis
Recency, Frequency, Monetary — the gold standard framework for player engagement scoring. Adapted from e-commerce and applied to gaming behavioral telemetry.

### Rolling Behavioral Windows
7-day and 30-day aggregates of sessions, playtime, and spending capture *trends* rather than snapshots — a player's trajectory matters more than their current state.

### K-Means Player Segmentation
Players are clustered into 4 actionable segments using RFM features:
- Whales — high spend, daily play, VIP treatment
- Dolphins — regular players, target with battle passes
- Minnows — frequent but free players, target with starter packs
- At-Risk — declining activity, needs win-back intervention

### A/B Test Framework
Statistical framework (two-proportion z-test) for testing whether retention interventions (bonus items, push notifications, difficulty adjustments) are statistically significant.

### SHAP Explainability
Feature attribution for every churn prediction — tells the live ops team exactly why a player is flagged as high risk.

---

## Model Performance

| Model | CV AUC |
|---|---|
| Logistic Regression | 0.712 |
| XGBoost | 0.831 |
| LightGBM | 0.847 |

---

## Top Churn Predictors (SHAP)

1. `days_since_last_session` — recency is the #1 signal
2. `rfm_composite` — overall engagement score
3. `SessionsPerWeek` — frequency of play
4. `engagement_velocity` — sessions × duration
5. `spend_last_30d` — monetary engagement

---

## Key Gaming Metrics Covered

- DAU/MAU ratio (Daily/Monthly Active Users)
- Session frequency and duration
- In-game spend (LTV proxy)
- Achievement progression rate
- Win rate trends
- Player level velocity

---

## Tech Stack

- Python 3.10+
- pandas, numpy — data manipulation
- scikit-learn — preprocessing, K-Means, logistic regression
- xgboost, lightgbm — gradient boosting
- shap — model explainability
- scipy — A/B test statistics
- streamlit — live ops dashboard
- plotly — interactive charts
- joblib — model serialization
- matplotlib, seaborn — EDA visualization
