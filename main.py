from src.preprocess import preprocess
from src.feature_engineering import feature_engineering
from src.model import train_and_predict

if __name__ == "__main__":
    train_clean, test_clean = preprocess(
        "data/train_v9rqX0R.csv",
        "data/test_AbJTz2l.csv"
    )
    train, test = feature_engineering(train_clean, test_clean)

    # 3. Train models with hyperparameter tuning and generate submission
    results, best_model_name, submission = train_and_predict(train, test)
