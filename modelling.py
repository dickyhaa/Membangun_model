import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


TARGET_COLUMN = "Churn"
DATA_DIR = "telco_customer_churn_preprocessing"

TRAIN_PATH = f"{DATA_DIR}/train_preprocessed.csv"
TEST_PATH = f"{DATA_DIR}/test_preprocessed.csv"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def main():
    mlflow.set_experiment("Telco Customer Churn Basic Modelling")

    X_train, X_test, y_train, y_test = load_data()

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()