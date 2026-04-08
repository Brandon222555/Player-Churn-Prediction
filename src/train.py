"""
src/train.py
=============
Full churn model training pipeline.

Usage:
    python src/train.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import run_pipeline, get_feature_matrix


def evaluate_models(X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nCross-validating models (5-fold stratified AUC)...")

    lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    s  = cross_val_score(lr, X_train, y_train, cv=skf, scoring='roc_auc')
    print(f"  Logistic Regression: {s.mean():.4f} ± {s.std():.4f}")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc', random_state=42, verbosity=0
    )
    s = cross_val_score(xgb_clf, X_train, y_train, cv=skf, scoring='roc_auc')
    print(f"  XGBoost:             {s.mean():.4f} ± {s.std():.4f}")

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        random_state=42, verbose=-1
    )
    s = cross_val_score(lgb_clf, X_train, y_train, cv=skf, scoring='roc_auc')
    print(f"  LightGBM:            {s.mean():.4f} ± {s.std():.4f}")

    return lr, xgb_clf, lgb_clf


def fit_best_model(lgb_clf, X_train, y_train, X_test, y_test):
    print("\nFitting LightGBM on full training set...")
    lgb_clf.fit(X_train, y_train)

    y_proba = lgb_clf.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    print(f"  Test AUC: {auc:.4f}")

    y_pred = (y_proba >= 0.5).astype(int)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
          target_names=['Retained', 'Churned']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return lgb_clf, y_proba


def save_artifacts(model, feature_names, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model,              os.path.join(model_dir, 'lgb_churn.pkl'))
    joblib.dump(list(feature_names), os.path.join(model_dir, 'feature_names.pkl'))
    print(f"\nModel saved to {model_dir}/")


def main():
    df = run_pipeline()
    X, y, feature_cols = get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    lr, xgb_clf, lgb_clf = evaluate_models(X_train, y_train)
    model, y_proba       = fit_best_model(lgb_clf, X_train, y_train, X_test, y_test)

    save_artifacts(model, feature_cols)

    # Save test predictions for dashboard demo
    X_test_df = X_test.copy()
    X_test_df['churn_prob'] = y_proba
    X_test_df['churned']    = y_test.values
    os.makedirs('data', exist_ok=True)
    X_test_df.to_csv('data/player_snapshot.csv', index=False)
    print("Player snapshot saved to data/player_snapshot.csv")

    print("\nDone! Run: streamlit run app.py")


if __name__ == '__main__':
    main()
