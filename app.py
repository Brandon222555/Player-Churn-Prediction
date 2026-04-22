"""
app.py — Player Churn Live Ops Dashboard
Streamlit dashboard — self-contained, no external data file needed.
Generates synthetic player data, trains the LightGBM model on first load,
then displays the full live ops dashboard.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

st.set_page_config(
    page_title="Player Retention Dashboard",
    page_icon="🎮",
    layout="wide"
)

# ─────────────────────────────────────────────
# Data generation + feature engineering
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_dataset(n=40000, seed=42):
    rng = np.random.default_rng(seed)

    churned = (rng.random(n) < 0.30).astype(int)

    df = pd.DataFrame({
        'PlayerID': np.arange(1, n + 1),
        'PlayerLevel': np.where(churned, rng.integers(1, 30, n), rng.integers(10, 100, n)),
        'SessionsPerWeek': np.where(churned, rng.integers(0, 3, n), rng.integers(2, 14, n)),
        'AvgSessionDurationMinutes': np.where(churned, rng.integers(5, 30, n), rng.integers(20, 120, n)),
        'AchievementsUnlocked': np.where(churned, rng.integers(0, 10, n), rng.integers(5, 50, n)),
        'InGamePurchases': np.where(churned, rng.integers(0, 2, n), rng.integers(0, 10, n)),
        'days_since_last_session': np.where(churned, rng.integers(7, 30, n), rng.integers(0, 6, n)),
        'spend_last_30d': np.where(churned, rng.exponential(2, n), rng.exponential(15, n)).round(2),
        'win_rate_7d': np.where(churned, rng.beta(2, 5, n), rng.beta(4, 3, n)).round(3),
        'social_actions_7d': np.where(churned, rng.poisson(1, n), rng.poisson(8, n)),
        'session_drop_pct': np.where(churned, rng.uniform(0.3, 1.0, n), rng.uniform(-0.2, 0.2, n)).round(3),
        'churned': churned,
    })

    # RFM features
    df['recency_raw'] = 1 / (df['days_since_last_session'] + 1)
    scaler = MinMaxScaler()
    rfm = scaler.fit_transform(df[['recency_raw', 'SessionsPerWeek', 'spend_last_30d']])
    df['recency_score']   = rfm[:, 0].round(4)
    df['frequency_score'] = rfm[:, 1].round(4)
    df['monetary_score']  = rfm[:, 2].round(4)
    df['rfm_composite']   = df[['recency_score', 'frequency_score', 'monetary_score']].mean(axis=1).round(4)
    df = df.drop(columns=['recency_raw'])

    # Behavioral features
    df['engagement_velocity'] = (df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']).round(1)
    df['achievement_rate']    = (df['AchievementsUnlocked'] / (df['PlayerLevel'] + 1)).round(4)
    df['weekly_playtime_hrs'] = (df['SessionsPerWeek'] * df['AvgSessionDurationMinutes'] / 60).round(2)
    df['is_spender']          = (df['InGamePurchases'] > 0).astype(int)
    df['at_risk_signal']      = ((df['session_drop_pct'] > 0.4) & (df['days_since_last_session'] > 5)).astype(int)

    return df


# ─────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'PlayerLevel', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
    'AchievementsUnlocked', 'InGamePurchases',
    'days_since_last_session', 'spend_last_30d', 'win_rate_7d',
    'social_actions_7d', 'session_drop_pct',
    'recency_score', 'frequency_score', 'monetary_score', 'rfm_composite',
    'engagement_velocity', 'achievement_rate', 'weekly_playtime_hrs',
    'is_spender', 'at_risk_signal',
]

@st.cache_resource(show_spinner=False)
def train_model():
    df = generate_dataset()
    X = df[FEATURE_COLS].fillna(0)
    y = df['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


# ─────────────────────────────────────────────
# Score players
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def score_players():
    df = generate_dataset()
    model, auc = train_model()
    df['churn_prob'] = model.predict_proba(df[FEATURE_COLS].fillna(0))[:, 1]
    df['risk_level'] = pd.cut(df['churn_prob'], bins=[0, 0.3, 0.6, 1.0],
                               labels=['Low risk', 'Medium risk', 'High risk'])

    def assign_segment(row):
        if row['spend_last_30d'] > 20 and row['SessionsPerWeek'] > 5:
            return 'Whales'
        elif row['spend_last_30d'] > 5 or row['SessionsPerWeek'] > 3:
            return 'Dolphins'
        elif row['churn_prob'] > 0.5:
            return 'At-Risk'
        else:
            return 'Minnows'

    df['segment'] = df.apply(assign_segment, axis=1)
    return df, auc


# ─────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────

st.title("Player Retention — Live Ops Dashboard")
st.markdown("Real-time churn risk scoring · Segment analysis · Intervention queue")
st.divider()

with st.spinner("Training LightGBM model on 40,000 player records…"):
    df, auc = score_players()

# ── KPI row ──────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total players", f"{len(df):,}")
c2.metric("Model AUC", f"{auc:.3f}")
c3.metric("High risk", f"{(df['risk_level']=='High risk').sum():,}",
          delta=f"{(df['risk_level']=='High risk').mean():.1%}", delta_color="inverse")
c4.metric("Avg churn prob", f"{df['churn_prob'].mean():.1%}")
c5.metric("Predicted churners", f"{(df['churn_prob']>0.5).sum():,}")
c6.metric("Whales", f"{(df['segment']=='Whales').sum():,}")

st.divider()

# ── Charts row ───────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn risk distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {'Low risk': '#1D9E75', 'Medium risk': '#BA7517', 'High risk': '#E24B4A'}
    counts = df['risk_level'].value_counts()
    ax.bar(counts.index, counts.values,
           color=[colors.get(str(l), 'gray') for l in counts.index])
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

# ── Top feature importance ────────────────────

st.subheader("Feature importance (SHAP-style gains)")
model, _ = train_model()
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(importances.index, importances.values, color='#378ADD')
ax.set_xlabel('Importance')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
st.pyplot(fig)
plt.close()

st.divider()

# ── Segment intervention guide ────────────────

st.subheader("Segment intervention guide")
s1, s2, s3, s4 = st.columns(4)
actions = {
    'Whales':    ('High value · Low churn',   'VIP rewards, exclusive content, early access'),
    'Dolphins':  ('Regular · Mid churn',       'Battle pass offers, limited events, social push'),
    'Minnows':   ('Frequent · No spend',       'Starter pack 50% off, first-purchase bonus'),
    'At-Risk':   ('Churning · Urgent',         'Free bonus item, win-back notification, easier match'),
}
for col, (seg, (subtitle, action)) in zip([s1, s2, s3, s4], actions.items()):
    count = (df['segment'] == seg).sum()
    with col:
        st.markdown(f"**{seg}** ({count:,})")
        st.caption(subtitle)
        st.info(action)

st.divider()

# ── High-risk queue ───────────────────────────

st.subheader("High-risk player queue — intervention needed")
high_risk = df[df['risk_level'] == 'High risk'].sort_values('churn_prob', ascending=False).head(100)
display_cols = [c for c in [
    'PlayerID', 'churn_prob', 'risk_level', 'segment',
    'SessionsPerWeek', 'days_since_last_session',
    'spend_last_30d', 'engagement_velocity'
] if c in high_risk.columns]

st.dataframe(
    high_risk[display_cols].style
        .background_gradient(subset=['churn_prob'], cmap='RdYlGn_r')
        .format({'churn_prob': '{:.1%}', 'spend_last_30d': '${:.2f}'}),
    use_container_width=True,
    height=380
)

st.divider()

# ── Single player risk checker ────────────────

st.subheader("Single player risk check")
scol1, scol2, scol3 = st.columns(3)

with scol1:
    sessions  = st.slider("Sessions per week", 0, 14, 3)
    duration  = st.slider("Avg session duration (min)", 5, 120, 40)
with scol2:
    days_away = st.slider("Days since last session", 0, 30, 4)
    spend     = st.slider("Spend last 30d ($)", 0, 100, 10)
with scol3:
    level        = st.slider("Player level", 1, 100, 25)
    achievements = st.slider("Achievements unlocked", 0, 50, 10)

if st.button("Check churn risk", type="primary"):
    recency_s   = 1 / (days_away + 1)
    frequency_s = sessions / 14
    monetary_s  = min(spend / 100, 1)

    row = {f: 0 for f in FEATURE_COLS}
    row.update({
        'PlayerLevel':               level,
        'SessionsPerWeek':           sessions,
        'AvgSessionDurationMinutes': duration,
        'AchievementsUnlocked':      achievements,
        'InGamePurchases':           int(spend > 0),
        'days_since_last_session':   days_away,
        'spend_last_30d':            spend,
        'win_rate_7d':               0.5,
        'social_actions_7d':         max(0, sessions - 1),
        'session_drop_pct':          0.0,
        'recency_score':             recency_s,
        'frequency_score':           frequency_s,
        'monetary_score':            monetary_s,
        'rfm_composite':             (recency_s + frequency_s + monetary_s) / 3,
        'engagement_velocity':       sessions * duration,
        'achievement_rate':          achievements / (level + 1),
        'weekly_playtime_hrs':       sessions * duration / 60,
        'is_spender':                int(spend > 0),
        'at_risk_signal':            int(days_away > 5),
    })

    X_single = pd.DataFrame([row])[FEATURE_COLS]
    prob = model.predict_proba(X_single)[0][1]
    pct  = round(prob * 100, 1)

    if prob > 0.6:
        st.error(f"🔴 HIGH RISK — {pct}% churn probability. Recommend immediate intervention.")
    elif prob > 0.3:
        st.warning(f"🟡 MEDIUM RISK — {pct}% churn probability. Monitor closely.")
    else:
        st.success(f"🟢 LOW RISK — {pct}% churn probability. Player is healthy.")

st.divider()
st.caption("Player Churn Prediction — ML Engineer portfolio project · Brandon Quansah · LightGBM · SHAP · Streamlit")
