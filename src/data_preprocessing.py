import argparse
import os
import pandas as pd


def preprocess(input_path: str, output_path: str, target_col: str, sheet_name: str | None) -> None:

    if input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip()

    # Basic cleanup
    df = df.drop_duplicates()

    # If target exists, keep it; otherwise just preprocess features
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    # Split into numeric and categorical
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Fill missing values
    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median(numeric_only=True))
    if categorical_cols:
        X[categorical_cols] = X[categorical_cols].fillna("UNKNOWN")

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Re-attach target
    if y is not None:
        out_df = pd.concat([X, y], axis=1)
    else:
        out_df = X

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[OK] Saved processed data to: {output_path}")
    print(f"[INFO] Shape: {out_df.shape}")


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/churn.csv")
    parser.add_argument("--output", default="data/processed/churn_processed.csv")
    parser.add_argument("--target", default="Churn", help="Target column name in the dataset")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (if input is .xlsx)")

    args = parser.parse_args()

    preprocess(args.input, args.output, args.target, args.sheet)



if __name__ == "__main__":
    main()
