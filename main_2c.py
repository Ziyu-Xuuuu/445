# main.py
import argparse
import pandas as pd
from helper import get_project_data
from project1 import cv_performance, select_param_logreg
from sklearn.linear_model import LogisticRegression

def main(n_jobs: int = -1, save_csv: bool = True, csv_path: str = "section2c_cv_results.csv"):
    X_train, y_train, X_test, y_test, feature_names = get_project_data(n_jobs=n_jobs)

    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    C_range = [10**i for i in range(-3, 4)]
    penalties = ["l1", "l2"]

    results = []
    for metric in metrics:
        best_C, best_penalty = select_param_logreg(X_train, y_train, C_range, penalties, metric=metric, k=5)
        clf = LogisticRegression(
            penalty=best_penalty,
            C=best_C,
            solver="liblinear",
            fit_intercept=False,
            random_state=42,
        )
        mean_score, min_score, max_score = cv_performance(clf, X_train, y_train, metric=metric, k=5)
        results.append([metric, best_C, best_penalty, mean_score, min_score, max_score])

    table = pd.DataFrame(results, columns=["Performance Measure", "C", "Penalty", "Mean", "Min", "Max"])
    print(table.to_string(index=False, float_format="%.4f"))

    if save_csv:
        table.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\nSaved table to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EECS 445 Project 2(c)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores")
    parser.add_argument("--no_save", action="store_true", help="Do not save CSV")
    parser.add_argument("--csv_path", type=str, default="section2c_cv_results.csv", help="Output CSV path")
    args = parser.parse_args()

    main(n_jobs=args.n_jobs, save_csv=not args.no_save, csv_path=args.csv_path)
