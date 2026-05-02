# Player Churn Prediction — Gaming Analytics

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-Analytics-lightgrey?logo=postgresql)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Machine learning system for predicting player churn in a gaming context. Engineers behavioral features from player activity data, trains a Random Forest classifier, and quantifies the revenue impact of churn reduction — translating model output into an executive business case.

---

## The Business Problem

Player churn is one of the most expensive problems in gaming. Acquiring a new player costs 5–7x more than retaining an existing one. This project answers the question every Head of Growth asks: **"Which players are about to leave, and how much does it cost us if we don't act?"**

The model identifies at-risk players before they churn — giving retention teams a window to intervene with targeted offers, re-engagement campaigns, or support outreach.

---

## Business Impact

> Reducing churn by **5%** → projected **~$200K increase in lifetime revenue**

This number isn't made up — it's derived from:
- Average revenue per retained player
- Modeled churn probability distribution
- Cost-of-acquisition savings from avoided re-acquisition

This is how Data Scientists present ML work to business stakeholders: not "the model got 87% accuracy," but "here's what it's worth."

---

## Key Results

| Metric | Value |
|---|---|
| Model | Random Forest Classifier |
| Top features | Session frequency decay, spend recency, days since last login |
| Business framing | 5% churn reduction → ~$200K revenue impact |
| Feature importance | Top 3 behavioral drivers identified and ranked |

---

## Feature Engineering

Raw player logs don't predict churn — engineered behavioral signals do. Features built for this model:

| Feature | Signal |
|---|---|
| Session frequency decay | Dropping login rate over last 30 days |
| Spend recency | Days since last in-app purchase |
| Engagement decay | Rolling 7-day vs 30-day session ratio |
| Days since last login | Recency signal |
| Total sessions (lifetime) | Player depth / investment |
| Average session length | Quality of engagement |
| Spend tier | High/mid/low value player segmentation |

---

## ML Pipeline

```
Raw Player Logs
      ↓
Feature Engineering (behavioral signals)
      ↓
Train/Test Split (stratified by churn label)
      ↓
Random Forest Classifier
• n_estimators: tuned via cross-validation
• class_weight: balanced (churn is minority class)
      ↓
Evaluation
• Precision / Recall / F1 on churn class
• Feature importance ranking
• Confusion matrix
      ↓
Business Case
• Translate recall on churn class → retained players
• Multiply retained players × ARPU → revenue impact
```

---

## Why Random Forest for Churn?

- **Handles mixed feature types** — numeric (session counts) and categorical (spend tier) without preprocessing overhead
- **Feature importance built-in** — tells you *which behaviors* drive churn, not just who is churning
- **Robust to class imbalance** with `class_weight='balanced'` — churn is typically 5–15% of players
- **Non-linear** — churn behavior rarely follows a straight line

---

## Tech Stack

```
Python 3.8+     — core language
Pandas / NumPy  — feature engineering
Scikit-Learn    — Random Forest, metrics, CV
Matplotlib      — feature importance, confusion matrix
SQL             — player activity aggregation queries
```

---

## Project Structure

```
Player-Churn-Prediction/
├── data/              # Simulated player activity logs
├── features/          # Behavioral feature engineering
├── models/            # Training + evaluation
├── visualizations/    # Feature importance, confusion matrix
├── sql/               # Activity aggregation queries
└── main.py            # Pipeline runner
```

---

## Extending This to Production

In a real gaming company, this pipeline would connect to:
- A data warehouse (Snowflake / BigQuery) for player event logs
- An Airflow DAG for weekly model retraining
- A CRM/marketing tool (Braze, Iterable) to trigger retention campaigns on high-risk players
- A dashboard (Looker / Tableau) for the Growth team to monitor churn risk scores

---

## Author

**Brandon Quansah** — Data Scientist | Physics B.S., Rowan University

[LinkedIn](https://linkedin.com/in/brandonquansah) · [GitHub](https://github.com/Brandon222555) · quansahb21@gmail.com
