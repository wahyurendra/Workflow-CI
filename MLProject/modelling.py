import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "dropout_label"
RANDOM_STATE = 42


def load_dataset(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def main() -> None:
    parser = argparse.ArgumentParser(description="MLProject training untuk GitHub Actions.")
    parser.add_argument(
        "--data-path",
        default="students_dropout_preprocessing/students_dropout_preprocessed.csv",
        help="Path dataset hasil preprocessing relatif terhadap folder MLProject.",
    )
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", default="8")
    args = parser.parse_args()

    max_depth = None if args.max_depth == "None" else int(args.max_depth)
    x, y = load_dataset(args.data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    with mlflow.start_run(run_name="ci_random_forest_training"):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("class_weight", "balanced")

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
        mlflow.log_metric("precision", precision_score(y_test, predictions, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, predictions, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, predictions, zero_division=0))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probabilities))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=x_test.head(5),
            pip_requirements=[
                "mlflow==2.19.0",
                "numpy<2.0.0",
                "pandas>=2.2.0,<2.3.0",
                "scikit-learn>=1.5.0,<=1.5.2",
            ],
        )


if __name__ == "__main__":
    main()
