"""
app.py — Player Churn Live Ops Dashboard
==========================================
Streamlit dashboard for the live ops team showing real-time
churn risk scores by player segment.

Usage:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Player Retention Dashboard",
    page_icon="🎮",
    layout="wide"
)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        model    = joblib.load('models/lgb_churn.pkl')
        features = joblib.load('models/feature_names.pkl')
        return model, features
    except FileNotFoundError:
        return None, None


model, feature_names = load_model()

# ─────────────────────────────────────────────
# Load player data
# ─────────────────────────────────────────────

@st.cache_data
def load_players():
    try:
        df = pd.read_csv('data/player_snapshot.csv')
        return df
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("Player Retention — Live Ops Dashboard")
st.markdown("Real-time churn risk scoring · Segment analysis · Intervention queue")
st.divider()

if model is None:
    st.error("Run `python src/train.py` first to train the model.")
    st.stop()

df = load_players()
if df is None:
    st.error("No player snapshot found. Run `python src/train.py` to generate one.")
    st.stop()

# Score players
X = df[feature_names].fillna(0)
df['churn_prob'] = model.predict_proba(X)[:, 1]
df['risk_level'] = pd.cut(
    df['churn_prob'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low risk', 'Medium risk', 'High risk']
)

# Assign simple segments based on spend + sessions
def assign_segment(row):
    if row.get('spend_last_30d', 0) > 20 and row.get('SessionsPerWeek', 0) > 5:
        return 'Whales'
    elif row.get('spend_last_30d', 0) > 5 or row.get('SessionsPerWeek', 0) > 3:
        return 'Dolphins'
    elif row.get('churn_prob', 0) > 0.5:
        return 'At-Risk'
    else:
        return 'Minnows'

df['segment'] = df.apply(assign_segment, axis=1)

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total players", f"{len(df):,}")
c2.metric("High risk", f"{(df['risk_level']=='High risk').sum():,}",
          delta=f"{(df['risk_level']=='High risk').mean():.1%}", delta_color="inverse")
c3.metric("Avg churn prob", f"{df['churn_prob'].mean():.1%}")
c4.metric("Predicted churners", f"{(df['churn_prob']>0.5).sum():,}")
c5.metric("Whales", f"{(df['segment']=='Whales').sum():,}")

st.divider()

# ─────────────────────────────────────────────
# Charts row
# ─────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn risk distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {'Low risk': '#1D9E75', 'Medium risk': '#BA7517', 'High risk': '#E24B4A'}
    counts = df['risk_level'].value_counts()
    ax.bar(counts.index, counts.values,
           color=[colors.get(l, 'gray') for l in counts.index])
    ax.set_ylabel('Players')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Churn rate by segment")
    fig, ax = plt.subplots(figsize=(6, 3))
    seg_churn = df.groupby('segment')['churned'].mean().sort_values()
    seg_colors = {'Whales': '#1D9E75', 'Dolphins': '#378ADD',
                  'Minnows': '#BA7517', 'At-Risk': '#E24B4A'}
    ax.barh(seg_churn.index, seg_churn.values,
            color=[seg_colors.get(s, 'gray') for s in seg_churn.index])
    ax.set_xlabel('Churn rate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

st.divider()

# ─────────────────────────────────────────────
# Segment action cards
# ─────────────────────────────────────────────

st.subheader("Segment intervention guide")
s1, s2, s3, s4 = st.columns(4)

actions = {
    'Whales':   ('High value · Low churn', 'VIP rewards, exclusive content, early access'),
    'Dolphins': ('Regular · Mid churn',    'Battle pass offers, limited events, social push'),
    'Minnows':  ('Frequent · No spend',    'Starter pack 50% off, first-purchase bonus'),
    'At-Risk':  ('Churning · Urgent',      'Free bonus item, win-back notification, easier match')
}

for col, (seg, (subtitle, action)) in zip([s1, s2, s3, s4], actions.items()):
    count = (df['segment'] == seg).sum()
    with col:
        st.markdown(f"**{seg}** ({count:,})")
        st.caption(subtitle)
        st.info(action)

st.divider()

# ─────────────────────────────────────────────
# High-risk intervention queue
# ─────────────────────────────────────────────

st.subheader("High-risk player queue — intervention needed")

high_risk = df[df['risk_level'] == 'High risk'].sort_values(
    'churn_prob', ascending=False
).head(100)

display_cols = [c for c in [
    'churn_prob', 'risk_level', 'segment',
    'SessionsPerWeek', 'days_since_last_session',
    'spend_last_30d', 'engagement_velocity'
] if c in high_risk.columns]

st.dataframe(
    high_risk[display_cols].style.background_gradient(
        subset=['churn_prob'], cmap='RdYlGn_r'
    ).format({'churn_prob': '{:.1%}', 'spend_last_30d': '${:.2f}'}),
    use_container_width=True,
    height=400
)

# ─────────────────────────────────────────────
# Single player risk checker
# ─────────────────────────────────────────────

st.divider()
st.subheader("Single player risk check")

scol1, scol2, scol3 = st.columns(3)
with scol1:
    sessions = st.slider("Sessions per week", 0, 14, 3)
    duration = st.slider("Avg session duration (min)", 5, 120, 40)
with scol2:
    days_away = st.slider("Days since last session", 0, 30, 4)
    spend     = st.slider("Spend last 30d ($)", 0, 100, 10)
with scol3:
    level        = st.slider("Player level", 1, 100, 25)
    achievements = st.slider("Achievements unlocked", 0, 50, 10)

if st.button("Check churn risk", type="primary"):
    row = {feat: 0 for feat in feature_names}
    overrides = {
        'SessionsPerWeek':            sessions,
        'AvgSessionDurationMinutes':  duration,
        'days_since_last_session':    days_away,
        'spend_last_30d':             spend,
        'PlayerLevel':                level,
        'AchievementsUnlocked':       achievements,
        'engagement_velocity':        sessions * duration,
        'recency_score':              1 / (days_away + 1),
        'frequency_score':            sessions / 14,
        'monetary_score':             min(spend / 100, 1),
        'rfm_composite':              (1/(days_away+1) + sessions/14 + min(spend/100,1)) / 3,
    }
    for k, v in overrides.items():
        if k in row:
            row[k] = v

    X_single = pd.DataFrame([row])[feature_names]
    prob     = model.predict_proba(X_single)[0][1]
    pct      = round(prob * 100, 1)

    if prob > 0.6:
        st.error(f"HIGH RISK — {pct}% churn probability. Recommend immediate intervention.")
    elif prob > 0.3:
        st.warning(f"MEDIUM RISK — {pct}% churn probability. Monitor closely.")
    else:
        st.success(f"LOW RISK — {pct}% churn probability. Player is healthy.")

st.divider()
st.caption("Player Churn Prediction — portfolio project. Data: Kaggle Online Gaming Behavior Dataset.")
