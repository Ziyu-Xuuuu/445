# main.py
import argparse
import numpy as np
import pandas as pd

from helper import get_project_data

def summarize_train_table(X_train: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean = X_train.mean(axis=0)
    q1   = np.percentile(X_train, 25, axis=0)
    q3   = np.percentile(X_train, 75, axis=0)
    iqr  = q3 - q1
    table = pd.DataFrame({
        "Feature": feature_names,
        "Mean Value": np.round(mean, 4),
        "Interquartile Range": np.round(iqr, 4),
    })
    return table

def main(n_jobs: int = -1, save_csv: bool = True, csv_path: str = "section1d_feature_stats_train.csv"):

    out = get_project_data(n_jobs=n_jobs)
    X_train, y_train, X_test, y_test, feature_names = out
    table = summarize_train_table(X_train, feature_names)

    print(table.to_string(index=False))
    if save_csv:
        table.to_csv(csv_path, index=False)
        print(f"\nSaved table to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EECS 445 Project 1(d)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="")
    parser.add_argument("--no_save", action="store_true", help="")
    parser.add_argument("--csv_path", type=str, default="section1d_feature_stats_train.csv", help="")
    args = parser.parse_args()

    main(n_jobs=args.n_jobs, save_csv=not args.no_save, csv_path=args.csv_path)
