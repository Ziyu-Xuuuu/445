# main_2d.py
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

from helper import get_project_data
from project1 import performance


def main(n_jobs: int = -1, save_csv: bool = True, csv_path: str = "section2d_bootstrap_results.csv"):
    X_train, y_train, X_test, y_test, feature_names = get_project_data(n_jobs=n_jobs)
    best_C = 1.0
    best_penalty = "l2"
    print(f"Using fixed parameters: C={best_C}, penalty={best_penalty}")
    clf = LogisticRegression(
        penalty=best_penalty,
        C=best_C,
        solver="liblinear",
        fit_intercept=False,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]
    results = []
    for metric in metrics:
        median, lower, upper = performance(
            clf, X_test, y_test, metric=metric, bootstrap=True
        )
        results.append([metric, median, lower, upper])

    table = pd.DataFrame(
        results,
        columns=["Performance Measure", "Median", "Lower 95% CI", "Upper 95% CI"],
    )
    print(table.to_string(index=False, float_format="%.4f"))

    if save_csv:
        table.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\nSaved table to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EECS 445 Project 2(d)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores")
    parser.add_argument("--no_save", action="store_true", help="Do not save CSV")
    parser.add_argument("--csv_path", type=str, default="section2d_bootstrap_results.csv", help="Output CSV path")
    args = parser.parse_args()

    main(n_jobs=args.n_jobs, save_csv=not args.no_save, csv_path=args.csv_path)
