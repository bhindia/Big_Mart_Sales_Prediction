# feature_engineering.py
import pandas as pd
import numpy as np


def fit_feature_engineering(train: pd.DataFrame):
    """
    Learn feature-engineering statistics from TRAIN only
    """
    fe_stats = {}

    # Mean visibility per item (for normalization)
    if "Item_Identifier" in train.columns and "Item_Visibility" in train.columns:
        fe_stats["mean_visibility_per_item"] = (
            train.groupby("Item_Identifier")["Item_Visibility"].mean()
        )

    return fe_stats


def transform_features(df: pd.DataFrame, fe_stats: dict):
    """
    Apply feature engineering using learned statistics
    """

    # 1. Outlet Age
    if "Outlet_Establishment_Year" in df.columns:
        CURRENT_YEAR = 2026
        df["Outlet_Age"] = CURRENT_YEAR - df["Outlet_Establishment_Year"]

    # 2. Item Category from Item_Identifier
    if "Item_Identifier" in df.columns:
        df["Item_Category"] = df["Item_Identifier"].str[:2].map({
            "FD": "Food",
            "NC": "Non-Consumable",
            "DR": "Drinks"
        })

    # 3. Visibility Mean Ratio
    if (
        "Item_Identifier" in df.columns and
        "Item_Visibility" in df.columns and
        "mean_visibility_per_item" in fe_stats
    ):
        df["Visibility_Mean_Ratio"] = (
            df["Item_Visibility"] /
            df["Item_Identifier"].map(fe_stats["mean_visibility_per_item"])
        )

        # Handle unseen items in test
        df["Visibility_Mean_Ratio"] = df["Visibility_Mean_Ratio"].fillna(1.0)

    # 4. Clean Item_Fat_Content
    if "Item_Fat_Content" in df.columns:
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
            "LF": "Low Fat",
            "reg": "Regular"
        })

    return df


def feature_engineering(train: pd.DataFrame, test: pd.DataFrame = None):
    """
    Full feature engineering pipeline
    """
    # Fit on train only
    fe_stats = fit_feature_engineering(train)

    # Transform train
    train = transform_features(train, fe_stats)

    # Transform test
    test = transform_features(test, fe_stats)

    return train, test
