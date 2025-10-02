import pandas as pd
from typing import Tuple, Optional

COMMON_TARGETS = ["Churn", "churn", "Exited", "is_churn", "Exited?", "target"]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def detect_target_column(df: pd.DataFrame, target: Optional[str] = None) -> str:
    if target:
        if target in df.columns:
            return target
        else:
            raise ValueError(f"Provided target column '{target}' not found in dataframe columns.")
    for c in COMMON_TARGETS:
        if c in df.columns:
            return c
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 2:
        return last_col
    raise ValueError("Could not detect target column automatically. Please pass `--target` with the correct column name.")

def load_and_inspect(path: str, target: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    df = load_csv(path)
    target_col = detect_target_column(df, target)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns. Detected target: '{target_col}'")
    return df, target_col
