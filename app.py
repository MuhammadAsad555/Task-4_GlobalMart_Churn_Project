"""
Streamlit app for GlobalMart churn model comparison.

Usage:
    streamlit run app.py

The app allows you to upload a CSV, specify the target column, train models (Logistic, RandomForest, optional XGBoost),
and view evaluation metrics and confusion matrices. You can download the trained best model.
"""
import streamlit as st
import pandas as pd
import io
import joblib
import matplotlib.pyplot as plt
from src.data_loader import load_and_inspect
from src.preprocessing import preprocess_and_split
from src.models import train_logistic, train_random_forest, train_xgboost, save_model
from src.evaluate import evaluate_model, compare_metrics, plot_confusion
import tempfile
import os

st.set_page_config(page_title="GlobalMart Churn — Model Comparison", layout="centered")

st.title("GlobalMart — Churn Model Trainer & Comparison")

uploaded_file = st.file_uploader("Upload CSV dataset (with target column)", type=["csv"])
target_col_input = st.text_input("Target column name (optional). Leave blank to auto-detect.", value="")
test_size = st.slider("Test set proportion", 0.1, 0.5, 0.2, 0.05)
preferred_metric = st.selectbox("Preferred metric to choose best model", ["f1", "precision", "recall", "accuracy", "roc_auc"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data (first 5 rows):")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    st.info("Upload a dataset to enable training.")

if st.button("Train & Evaluate") and uploaded_file is not None:
    with st.spinner("Running training pipeline..."):
        try:
            # Save temporary file so data_loader can read by path if needed
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            tmp.close()
            df_loaded, target_col = load_and_inspect(tmp.name, target_col_input or None)
            X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df_loaded, target_col, test_size=test_size)
            results = {}

            st.write("Training Logistic Regression...")
            lr = train_logistic(X_train, y_train)
            m_lr = evaluate_model(lr, X_test, y_test)
            results['logistic_regression'] = m_lr
            save_model(lr, 'models/logistic_regression.joblib')

            st.write("Training Random Forest...")
            rf = train_random_forest(X_train, y_train)
            m_rf = evaluate_model(rf, X_test, y_test)
            results['random_forest'] = m_rf
            save_model(rf, 'models/random_forest.joblib')

            try:
                st.write("Training XGBoost (if installed)...")
                xgb = train_xgboost(X_train, y_train)
                m_xgb = evaluate_model(xgb, X_test, y_test)
                results['xgboost'] = m_xgb
                save_model(xgb, 'models/xgboost.joblib')
            except Exception as e:
                st.warning(f"XGBoost training skipped or failed: {e}")

            st.write("## Model comparison")
            comp_df = compare_metrics(results)
            st.dataframe(comp_df.round(4))

            # Choose best model
            best_model_name = comp_df[preferred_metric].idxmax()
            st.success(f"Best model by '{preferred_metric}': {best_model_name}")

            # Save best model to models/best_model.joblib
            if best_model_name == 'random_forest':
                save_model(rf, 'models/best_model.joblib')
                best_path = 'models/random_forest.joblib'
            elif best_model_name == 'logistic_regression':
                save_model(lr, 'models/best_model.joblib')
                best_path = 'models/logistic_regression.joblib'
            elif best_model_name == 'xgboost' and 'xgb' in locals():
                save_model(xgb, 'models/best_model.joblib')
                best_path = 'models/xgboost.joblib'
            else:
                best_path = None

            # Show confusion matrices
            for name, m in results.items():
                st.write(f"### Confusion matrix — {name}")
                fig, ax = plt.subplots()
                import seaborn as sns
                sns.heatmap(m['confusion_matrix'], annot=True, fmt='d', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

            if best_path and os.path.exists(best_path):
                with open(best_path, "rb") as f:
                    bytes_data = f.read()
                st.download_button("Download best model (joblib)", data=bytes_data, file_name="best_model.joblib")
            else:
                st.info("No best model file available to download.")

        except Exception as e:
            st.error(f"Training pipeline failed: {e}")
