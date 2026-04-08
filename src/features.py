"""
src/features.py
================
RFM + behavioral feature engineering for player churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_raw(path='data/online_gaming_behavior_dataset.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} players, {df.shape[1]} columns")
    return df


def add_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """Define churn as Low engagement level."""
    df = df.copy()
    df['churned'] = (df['EngagementLevel'] == 'Low').astype(int)
    print(f"Churn rate: {df['churned'].mean():.1%} ({df['churned'].sum():,} players)")
    return df


def add_synthetic_telemetry(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Simulate behavioral telemetry features that would come from
    a real game's event tracking system.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    churned = df['churned'].values

    # Days since last session — churned players have been away longer
    df['days_since_last_session'] = np.where(
        churned == 1,
        rng.integers(7, 30, n),
        rng.integers(0, 6, n)
    )

    # Spend in last 30 days — churned players spend less
    df['spend_last_30d'] = np.where(
        churned == 1,
        rng.exponential(2, n),
        rng.exponential(15, n)
    ).round(2)

    # Win rate in last 7 days
    df['win_rate_7d'] = np.where(
        churned == 1,
        rng.beta(2, 5, n),   # losing more
        rng.beta(4, 3, n)    # winning more
    ).round(3)

    # Social activity (guild messages, friend activity)
    df['social_actions_7d'] = np.where(
        churned == 1,
        rng.poisson(1, n),
        rng.poisson(8, n)
    )

    # Session drop — sessions this week vs last week
    df['session_drop_pct'] = np.where(
        churned == 1,
        rng.uniform(0.3, 1.0, n),    # big drop
        rng.uniform(-0.2, 0.2, n)    # stable
    ).round(3)

    return df


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recency, Frequency, Monetary features — normalized 0-1.
    Higher score = more engaged player.
    """
    df = df.copy()

    # Recency: inverse of days since last session
    df['recency_raw'] = 1 / (df['days_since_last_session'] + 1)

    # Frequency: sessions per week
    df['frequency_raw'] = df['SessionsPerWeek']

    # Monetary: 30-day spend
    df['monetary_raw'] = df['spend_last_30d']

    # Normalize each 0-1
    scaler = MinMaxScaler()
    rfm_raw = ['recency_raw', 'frequency_raw', 'monetary_raw']
    normed  = ['recency_score', 'frequency_score', 'monetary_score']
    df[normed] = scaler.fit_transform(df[rfm_raw])
    df = df.drop(columns=rfm_raw)

    # Composite RFM score
    df['rfm_composite'] = df[normed].mean(axis=1).round(4)

    return df


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived behavioral features from raw telemetry."""
    df = df.copy()

    # Engagement velocity: sessions × avg session length
    df['engagement_velocity'] = (
        df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']
    ).round(1)

    # Achievement efficiency: achievements per player level
    df['achievement_rate'] = (
        df['AchievementsUnlocked'] / (df['PlayerLevel'] + 1)
    ).round(4)

    # Weekly playtime total
    df['weekly_playtime_hrs'] = (
        df['SessionsPerWeek'] * df['AvgSessionDurationMinutes'] / 60
    ).round(2)

    # Spender flag
    df['is_spender'] = df['InGamePurchases'].astype(int)

    # High difficulty preference
    diff_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    df['difficulty_num'] = df['GameDifficulty'].map(diff_map).fillna(1)

    # At-risk signal: high session drop + low recency
    df['at_risk_signal'] = (
        (df['session_drop_pct'] > 0.4) &
        (df['days_since_last_session'] > 5)
    ).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = ['Gender', 'Location', 'GameGenre']
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X, y ready for modeling."""
    drop_cols = ['PlayerID', 'EngagementLevel', 'churned',
                 'GameDifficulty', 'Gender', 'Location', 'GameGenre']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0)
    y = df['churned']
    return X, y, feature_cols


def run_pipeline(path='data/online_gaming_behavior_dataset.csv'):
    """Full feature engineering pipeline."""
    df = load_raw(path)
    df = add_churn_label(df)
    df = add_synthetic_telemetry(df)
    df = build_rfm_features(df)
    df = build_behavioral_features(df)
    df = encode_categoricals(df)
    print(f"Final shape: {df.shape}")
    return df


if __name__ == '__main__':
    df = run_pipeline()
    print(df.head())
