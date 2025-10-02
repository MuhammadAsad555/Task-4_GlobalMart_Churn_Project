"""run_compare.py
Usage:
  python run_compare.py --data data/churn.csv --target Churn

This script will:
 - load data
 - preprocess and split
 - train logistic regression and random forest (and xgboost if installed)
 - evaluate models and print a comparison table
 - save best model to models/best_model.joblib
"""
import argparse
import os
from src.data_loader import load_and_inspect
from src.preprocessing import preprocess_and_split
from src.models import train_logistic, train_random_forest, train_xgboost, save_model
from src.evaluate import evaluate_model, compare_metrics, plot_confusion
from src.utils import save_metrics

def main(args):
    df, target_col = load_and_inspect(args.data, args.target)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df, target_col, test_size=args.test_size)
    results = {}

    print('\\nTraining Logistic Regression...')
    lr = train_logistic(X_train, y_train)
    m_lr = evaluate_model(lr, X_test, y_test)
    results['logistic_regression'] = m_lr
    save_model(lr, 'models/logistic_regression.joblib')

    print('\\nTraining Random Forest...')
    rf = train_random_forest(X_train, y_train)
    m_rf = evaluate_model(rf, X_test, y_test)
    results['random_forest'] = m_rf
    save_model(rf, 'models/random_forest.joblib')

    try:
        print('\\nTraining XGBoost...')
        xgb = train_xgboost(X_train, y_train)
        m_xgb = evaluate_model(xgb, X_test, y_test)
        results['xgboost'] = m_xgb
        save_model(xgb, 'models/xgboost.joblib')
    except Exception as e:
        print('XGBoost training skipped:', str(e))

    print('\\n--- Model comparison ---')
    df_comp = compare_metrics(results)

    save_metrics({k: {kk: (vv if not hasattr(vv, 'tolist') else vv.tolist()) for kk, vv in m.items()} for k, m in results.items()}, 'models/metrics.json')

    preferred_metric = args.preferred_metric
    print(f"\\nChoosing best model by '{preferred_metric}'")
    best_model = df_comp[preferred_metric].idxmax()
    print('Best model:', best_model)

    if best_model == 'random_forest':
        save_model(rf, 'models/best_model.joblib')
    elif best_model == 'logistic_regression':
        save_model(lr, 'models/best_model.joblib')
    elif best_model == 'xgboost' and 'xgb' in locals():
        save_model(xgb, 'models/best_model.joblib')

    for name, m in results.items():
        print(f"\\nConfusion matrix for {name}:")
        plot_confusion(m['confusion_matrix'], title=f"Confusion - {name}")

    print('\\nAll done. Models saved to models/ directory.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--target', type=str, default=None, help='Target column name (optional)')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--preferred-metric', dest='preferred_metric', choices=['f1','precision','recall','accuracy','roc_auc'], default='f1')
    args = parser.parse_args()
    main(args)
