from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

def train_logistic(X_train, y_train, random_state=42):
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train, random_state=42):
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train, random_state=42):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise ImportError("xgboost is not installed. Install it or skip XGBoost training.")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1)
    xgb.fit(X_train, y_train)
    return xgb

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)
