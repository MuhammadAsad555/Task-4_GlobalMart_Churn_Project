import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, average='binary'):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0, average=average),
        'recall': recall_score(y_test, y_pred, zero_division=0, average=average),
        'f1': f1_score(y_test, y_pred, zero_division=0, average=average)
    }

    if y_proba is not None and len(np.unique(y_test)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)

    return metrics

def plot_confusion(cm, title='Confusion Matrix'):
    import seaborn as sns
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def compare_metrics(metrics_dict: dict):
    import pandas as pd
    rows = []
    for name, m in metrics_dict.items():
        rows.append({
            'model': name,
            'accuracy': m.get('accuracy'),
            'precision': m.get('precision'),
            'recall': m.get('recall'),
            'f1': m.get('f1'),
            'roc_auc': m.get('roc_auc')
        })
    df = pd.DataFrame(rows).set_index('model')
    print(df.round(4))
    return df
