# %% [markdown]
# # Churn prediction — Logistic Regression vs Random Forest (and XGBoost)
# This notebook runs an end-to-end comparison and visualises the results.

# %%
# Imports and helpers
import pandas as pd
from src.data_loader import load_and_inspect
from src.preprocessing import preprocess_and_split
from src.models import train_logistic, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, compare_metrics, plot_confusion

# %%
# Load dataset
DATA_PATH = 'data/churn.csv'  # change if needed
df, target = load_and_inspect(DATA_PATH, target=None)

# %%
# Preprocess and split
X_train, X_test, y_train, y_test, preproc = preprocess_and_split(df, target)

# %%
# Train Logistic Regression
lr = train_logistic(X_train, y_train)
metrics_lr = evaluate_model(lr, X_test, y_test)
print('Logistic metrics:')
print(metrics_lr['classification_report'])

# %%
# Train Random Forest
rf = train_random_forest(X_train, y_train)
metrics_rf = evaluate_model(rf, X_test, y_test)
print('Random Forest metrics:')
print(metrics_rf['classification_report'])

# %%
# Try XGBoost (optional)
try:
    xgb = train_xgboost(X_train, y_train)
    metrics_xgb = evaluate_model(xgb, X_test, y_test)
    print('XGBoost metrics:')
    print(metrics_xgb['classification_report'])
except Exception as e:
    print('XGBoost not available or failed:', e)
    metrics_xgb = None

# %%
# Compare
metrics_all = {
    'logistic_regression': metrics_lr,
    'random_forest': metrics_rf,
}
if metrics_xgb is not None:
    metrics_all['xgboost'] = metrics_xgb

comp_df = compare_metrics(metrics_all)
comp_df

# %%
# Plot confusion matrices
for name, m in metrics_all.items():
    plot_confusion(m['confusion_matrix'], title=f'Confusion: {name}')
