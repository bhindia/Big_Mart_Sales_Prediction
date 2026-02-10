# model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings("ignore")


def encode_features(train: pd.DataFrame, test: pd.DataFrame):
    """Label encode all categorical columns safely with unseen handling"""
    cat_cols = train.select_dtypes(include="object").columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])

        # For test: map unseen categories to a new label (-1)
        test[col] = test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        encoders[col] = le

    return train, test, encoders


def tune_models(X_train, y_train, X_val, y_val):
    """
    Grid search hyperparameter tuning for multiple models and evaluation
    """
    models = {
        "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10]}),
        "Lasso": (Lasso(), {"alpha": [0.001, 0.01, 0.1, 1]}),
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5]}
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, objective="reg:squarederror"),
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5]}
        ),
        "LightGBM": (
            LGBMRegressor(
                random_state=42,
                force_row_wise=True  # fixed parameter
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [5, 10, -1],
                "min_child_samples": [5, 10],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        )
    }

    results = {}

    for name, (model, params) in models.items():
        print(f"\nTuning {name} ...")
        grid = GridSearchCV(model, params, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        val_preds = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        results[name] = {"rmse": rmse, "best_model": best_model, "best_params": grid.best_params_}

        print(f"{name} best RMSE: {rmse:.4f}")
        print(f"Best params: {grid.best_params_}")

    # Identify overall best model
    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nBest model overall: {best_name} with RMSE {results[best_name]['rmse']:.4f}")

    return results, best_name, results[best_name]["best_model"]


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame):
    """Full training + hyperparameter tuning pipeline"""

    # Separate features and target
    X = train.drop("Item_Outlet_Sales", axis=1)
    y = train["Item_Outlet_Sales"]

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode categorical features
    X_train, X_val, _ = encode_features(X_train, X_val)

    # Tune models
    results, best_name, best_model = tune_models(X_train, y_train, X_val, y_val)

    # Train best model on full train data
    X_full, test_encoded, _ = encode_features(X, test.copy())
    best_model.fit(X_full, y)

    # Predict test
    test_preds = best_model.predict(test_encoded)

    # Submission
    submission = pd.DataFrame({
        "Item_Identifier": test["Item_Identifier"],
        "Outlet_Identifier": test["Outlet_Identifier"],
        "Item_Outlet_Sales": test_preds
    })
    submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].round(2)
    submission.to_csv("data/submission.csv", index=False)
    print("\nSubmission saved to submission.csv")

    return results, best_name, submission
