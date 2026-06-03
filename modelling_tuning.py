import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from mlflow.models.signature import infer_signature


TARGET_COLUMN = "Churn"
DATA_DIR = "telco_customer_churn_preprocessing"

TRAIN_PATH = f"{DATA_DIR}/train_preprocessed.csv"
TEST_PATH = f"{DATA_DIR}/test_preprocessed.csv"

ARTIFACT_DIR = "artifacts"


def setup_mlflow():
    """
    Mengatur MLflow Tracking.
    Untuk Advanced, tracking diarahkan ke DagsHub.
    """

    dagshub.init(
        repo_owner="dickyhaa",
        repo_name="MSML-Telco-Churn",
        mlflow=True
    )

    mlflow.set_experiment("Telco Customer Churn Advanced Modelling")

    print("MLflow tracking diarahkan ke DagsHub.")


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def create_artifact_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)

    report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")

    with open(report_path, "w") as file:
        file.write(report)

    return report_path


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot()
    plt.title("Confusion Matrix - Telco Customer Churn")
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"))
    plt.close()


def save_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - Telco Customer Churn")
    plt.savefig(os.path.join(ARTIFACT_DIR, "roc_curve.png"))
    plt.close()


def save_feature_importance(model, feature_names):
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    feature_importance.to_csv(
        os.path.join(ARTIFACT_DIR, "feature_importance.csv"),
        index=False
    )


def save_metrics_json(metrics):
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    return metrics_path


def save_best_params_json(best_params):
    params_path = os.path.join(ARTIFACT_DIR, "best_params.json")

    with open(params_path, "w") as file:
        json.dump(best_params, file, indent=4)

    return params_path


def save_dataset_info(X_train, X_test, y_train, y_test):
    dataset_info = {
        "train_rows": int(X_train.shape[0]),
        "train_features": int(X_train.shape[1]),
        "test_rows": int(X_test.shape[0]),
        "test_features": int(X_test.shape[1]),
        "target_column": TARGET_COLUMN,
        "train_target_distribution": {
            str(key): int(value) for key, value in y_train.value_counts().to_dict().items()
        },
        "test_target_distribution": {
            str(key): int(value) for key, value in y_test.value_counts().to_dict().items()
        }
    }

    dataset_info_path = os.path.join(ARTIFACT_DIR, "dataset_info.json")

    with open(dataset_info_path, "w") as file:
        json.dump(dataset_info, file, indent=4)

    return dataset_info_path


def train_with_tuning(X_train, y_train):
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def main():
    setup_mlflow()
    create_artifact_dir()

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="RandomForest_Tuning_Manual_Logging") as run:
        grid_search = train_with_tuning(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "best_cv_score": float(grid_search.best_score_)
        }

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("cv", 3)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        mlflow.log_metrics(metrics)

        save_classification_report(y_test, y_pred)
        save_confusion_matrix(y_test, y_pred)
        save_roc_curve(best_model, X_test, y_test)
        save_feature_importance(best_model, X_train.columns)
        save_metrics_json(metrics)
        save_best_params_json(grid_search.best_params_)
        save_dataset_info(X_train, X_test, y_train, y_test)

        mlflow.log_artifacts(ARTIFACT_DIR)

        input_example = X_test.head(5)
        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        print("Training dan tuning selesai.")
        print("Run ID:", run.info.run_id)
        print("Best Params:", grid_search.best_params_)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC:", roc_auc)


if __name__ == "__main__":
    main()