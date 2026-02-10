# new_main.py
# -*- coding: utf-8 -*-
"""
Big Mart Sales Prediction
Single-file pipeline: preprocessing, model training, prediction, submission
No GridSearch, intuition-based hyperparameters
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------
# Utility
# ----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "Item_Outlet_Sales"

    combined = pd.concat([train.drop(columns=[target]), test], axis=0)

    # Fill missing Item_Weight
    combined["Item_Weight"] = combined["Item_Weight"].fillna(
        combined.groupby("Item_Identifier")["Item_Weight"].transform("mean")
    )

    # Fill missing Outlet_Size
    combined["Outlet_Size"] = combined["Outlet_Size"].fillna("Missing")

    # Replace zero Item_Visibility and fill
    combined["Item_Visibility"] = combined["Item_Visibility"].replace(0, np.nan)
    combined["Item_Visibility"] = combined["Item_Visibility"].fillna(
        combined.groupby("Item_Type")["Item_Visibility"].transform("mean")
    )

    # Encode categorical columns
    cat_cols = combined.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])

    # Split back
    X_train = combined.iloc[:len(train), :]
    X_test = combined.iloc[len(train):, :]
    y = train[target]

    return X_train, y, X_test, test[["Item_Identifier", "Outlet_Identifier"]]

# ----------------------------
# Models
# ----------------------------
def get_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

# ----------------------------
# Train + select model
# ----------------------------
def train_and_select_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        results[name] = rmse(y_val, preds)
        print(f"{name} | RMSE: {results[name]:.2f}")

    best_model_name = min(results, key=results.get)
    print(f"\nBest model selected: {best_model_name}")

    return models[best_model_name]

# ----------------------------
# Train full + predict
# ----------------------------
def train_full_and_predict(model, X, y, X_test):
    model.fit(X, y)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # No negative sales
    return preds

# ----------------------------
# Save submission
# ----------------------------
def save_submission(ids, predictions):
    submission = pd.DataFrame({
        "Item_Identifier": ids["Item_Identifier"],
        "Outlet_Identifier": ids["Outlet_Identifier"],
        "Item_Outlet_Sales": predictions
    })
    submission.to_csv("submission.csv", index=False)
    print("submission.csv saved!")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    train_csv = "data/train_v9rqX0R.csv"
    test_csv = "data/test_AbJTz2l.csv"

    X, y, X_test, test_ids = preprocess(train_csv, test_csv)

    best_model = train_and_select_model(X, y)
    predictions = train_full_and_predict(best_model, X, y, X_test)

    save_submission(test_ids, predictions)


