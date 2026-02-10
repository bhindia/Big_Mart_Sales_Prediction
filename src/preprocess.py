# preprocess.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(train_path: str, test_path: str = None):
    """Load train and optional test dataset"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def log_zero_negative(df: pd.DataFrame, exclude_cols=None):
    """Log zero and negative values in numeric columns"""
    exclude_cols = exclude_cols or []
    num_cols = df.select_dtypes(include="number").columns
    num_cols = [col for col in num_cols if col not in exclude_cols]

    logs = []
    for col in num_cols:
        zero_count = (df[col] == 0).sum()
        neg_count = (df[col] < 0).sum()
        total = df[col].notna().sum()

        logs.append({
            "column": col,
            "zeros": zero_count,
            "negatives": neg_count,
            "zero_percent": zero_count / total * 100 if total else 0,
            "negative_percent": neg_count / total * 100 if total else 0
        })

    return pd.DataFrame(logs)


def fit_preprocessor(train: pd.DataFrame):
    """Learn preprocessing statistics from TRAIN only"""

    stats = {}

    # Item_Weight median per Item_Type
    if "Item_Weight" in train.columns and "Item_Type" in train.columns:
        stats["item_weight_median_by_type"] = (
            train.groupby("Item_Type")["Item_Weight"].median()
        )
        stats["item_weight_global_median"] = train["Item_Weight"].median()

    # Item_Visibility median
    if "Item_Visibility" in train.columns:
        stats["item_visibility_median"] = train["Item_Visibility"].median()

    return stats


def transform_data(df: pd.DataFrame, stats: dict):
    """Apply preprocessing using precomputed statistics"""

    # Replace invalid values
    if "Item_Visibility" in df.columns:
        df.loc[df["Item_Visibility"] <= 0, "Item_Visibility"] = np.nan

    # Item_Weight imputation
    if "Item_Weight" in df.columns and "Item_Type" in df.columns:
        df["Item_Weight"] = df["Item_Weight"].fillna(
            df["Item_Type"].map(stats["item_weight_median_by_type"])
        )
        df["Item_Weight"] = df["Item_Weight"].fillna(
            stats["item_weight_global_median"]
        )

    # Item_Visibility imputation
    if "Item_Visibility" in df.columns:
        df["Item_Visibility"] = df["Item_Visibility"].fillna(
            stats["item_visibility_median"]
        )

    # Outlet_Size
    if "Outlet_Size" in df.columns:
        df["Outlet_Size"] = df["Outlet_Size"].fillna("Unknown")

    return df

def preprocess(train_path: str, test_path: str = None):
    train, test = load_data(train_path, test_path)

    log_zero_negative(train, exclude_cols=["Outlet_Establishment_Year"])

    log_zero_negative(test, exclude_cols=["Outlet_Establishment_Year"])

    # Fit only on train
    stats = fit_preprocessor(train)

    # Transform train & test
    train = transform_data(train, stats)

    test = transform_data(test, stats)

    return train, test


