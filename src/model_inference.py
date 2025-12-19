import argparse
import joblib
import pandas as pd


def predict(model_path: str, processed_path: str, n_rows: int = 5) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(processed_path)

    # If your processed file still contains target, drop it safely
    for possible_target in ["Churn", "churn", "target", "label"]:
        if possible_target in df.columns:
            df = df.drop(columns=[possible_target])

    sample = df.head(n_rows)
    preds = model.predict(sample)

    print("[OK] Predictions for first rows:")
    out = sample.copy()
    out["prediction"] = preds
    print(out.head(n_rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/churn_model.pkl")
    parser.add_argument("--processed", default="data/processed/churn_processed.csv")
    parser.add_argument("--n_rows", type=int, default=5)
    args = parser.parse_args()

    predict(args.model, args.processed, args.n_rows)


if __name__ == "__main__":
    main()
