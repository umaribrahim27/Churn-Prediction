import argparse
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


def train(processed_path: str, target_col: str, model_out: str, test_size: float, random_state: int) -> None:
    df = pd.read_csv(processed_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {processed_path}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If target is text labels, sklearn will still handle it; metrics will work if binary.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 20 else None
    )

    model = LogisticRegression(max_iter=200, n_jobs=None)

    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_iter", 200)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="binary" if y.nunique() == 2 else "weighted")
        prec = precision_score(y_test, preds, average="binary" if y.nunique() == 2 else "weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="binary" if y.nunique() == 2 else "weighted", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(model, model_out)

        # Log artifacts to MLflow
        mlflow.log_artifact(model_out)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("[OK] Trained model + saved to:", model_out)
        print(f"[METRICS] accuracy={acc:.4f} f1={f1:.4f} precision={prec:.4f} recall={rec:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/churn_processed.csv")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--model_out", default="models/churn_model.pkl")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    train(args.processed, args.target, args.model_out, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
