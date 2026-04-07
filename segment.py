"""
src/segment.py
===============
K-Means player segmentation using RFM features.

Usage:
    python src/segment.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import run_pipeline


SEGMENT_LABELS = {
    0: 'Whales',
    1: 'Dolphins',
    2: 'Minnows',
    3: 'At-Risk'
}

SEGMENT_ACTIONS = {
    'Whales':   'VIP rewards, exclusive content, early access to new content',
    'Dolphins': 'Battle pass offers, limited-time events, social features',
    'Minnows':  'Starter packs, value bundles, first-purchase 50% discount',
    'At-Risk':  'Win-back push notification, free bonus items, reduced difficulty'
}


def find_optimal_k(rfm_scaled: np.ndarray, max_k: int = 10) -> None:
    """Plot elbow curve to find optimal number of clusters."""
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_k + 1), inertias, 'bo-', markersize=6)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow method — optimal K for player segmentation')
    plt.xticks(range(2, max_k + 1))
    plt.tight_layout()
    plt.savefig('outputs/elbow_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Elbow curve saved to outputs/elbow_curve.png")


def fit_segments(df: pd.DataFrame, n_clusters: int = 4):
    """Fit K-Means on RFM features and assign segment labels."""
    rfm_cols = ['recency_score', 'frequency_score',
                'monetary_score', 'engagement_velocity']
    rfm_cols = [c for c in rfm_cols if c in df.columns]

    rfm = df[rfm_cols].fillna(0)
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(rfm_scaled)

    # Label clusters by their mean RFM profile
    profile = df.groupby('cluster')[rfm_cols + ['churned']].mean()
    profile['rfm_mean'] = profile[rfm_cols].mean(axis=1)
    order   = profile['rfm_mean'].rank(ascending=False).astype(int) - 1
    label_map = {cluster: list(SEGMENT_LABELS.values())[rank]
                 for cluster, rank in order.items()}
    df['segment'] = df['cluster'].map(label_map)

    return df, km, scaler, profile


def print_segment_report(df: pd.DataFrame) -> None:
    """Print summary stats per segment."""
    print("\nPlayer segment profiles:")
    print("=" * 60)
    summary = df.groupby('segment').agg(
        players=('churned', 'count'),
        churn_rate=('churned', 'mean'),
        avg_sessions=('SessionsPerWeek', 'mean'),
        avg_spend=('spend_last_30d', 'mean'),
        avg_rfm=('rfm_composite', 'mean')
    ).round(3)
    print(summary.to_string())

    print("\nRecommended actions per segment:")
    print("=" * 60)
    for seg, action in SEGMENT_ACTIONS.items():
        count = (df['segment'] == seg).sum()
        print(f"\n  {seg} ({count:,} players)")
        print(f"  -> {action}")


def plot_segments(df: pd.DataFrame) -> None:
    """Scatter plot of segments on RFM space."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'Whales': '#1D9E75', 'Dolphins': '#378ADD',
              'Minnows': '#BA7517', 'At-Risk': '#E24B4A'}

    for seg, grp in df.groupby('segment'):
        axes[0].scatter(grp['recency_score'], grp['frequency_score'],
                       label=seg, alpha=0.4, s=8, color=colors.get(seg, 'gray'))
    axes[0].set_xlabel('Recency score')
    axes[0].set_ylabel('Frequency score')
    axes[0].set_title('Player segments — recency vs frequency')
    axes[0].legend()

    churn_by_seg = df.groupby('segment')['churned'].mean().sort_values()
    churn_by_seg.plot(kind='barh', ax=axes[1],
                      color=[colors.get(s, 'gray') for s in churn_by_seg.index])
    axes[1].set_xlabel('Churn rate')
    axes[1].set_title('Churn rate by segment')

    plt.tight_layout()
    plt.savefig('outputs/player_segments.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Segment plot saved to outputs/player_segments.png")


def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df = run_pipeline()
    df, km, scaler, profile = fit_segments(df)
    print_segment_report(df)
    plot_segments(df)

    joblib.dump(km,     'models/kmeans_segments.pkl')
    joblib.dump(scaler, 'models/segment_scaler.pkl')
    df[['segment']].to_csv('data/player_segments.csv', index=False)
    print("\nSegmentation model saved.")


if __name__ == '__main__':
    main()
