import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Tuple

def build_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")
    return preprocessor, numeric_cols, categorical_cols

def preprocess_and_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    y = df[target_col].copy()
    if y.dtype == object or y.dtype.name == 'category':
        y = y.replace({"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0})
        y = pd.to_numeric(y, errors='coerce')

    X = df.drop(columns=[target_col])
    preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique())>1 else None
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    return X_train_trans, X_test_trans, y_train.values, y_test.values, preprocessor
