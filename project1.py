"""
EECS 445 Fall 2025

This script contains most of the work for the project. You will need to fill in every TODO comment.
"""

import random

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import helper
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.metrics import roc_curve, auc

__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)
random.seed(seed)


# def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
#     """
#     Reads a dataframe containing all measurements for a single patient
#     within the first 48 hours of the ICU admission, and convert it into
#     a feature vector.

#     Args:
#         df: DataFrame with columns [Time, Variable, Value]

#     Returns:
#         a python dictionary of format {feature_name: feature_value}
#         for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
#     """
#     static_variables = config["static"]
#     timeseries_variables = config["timeseries"]
    
#     # TODO: 1) Replace unknown values with np.nan
#     # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
#     df_replaced = df.copy()
#     df_replaced["Value"] = pd.to_numeric(df_replaced["Value"], errors="coerce")
#     df_replaced["Value"] = df_replaced["Value"].replace(-1, np.nan)


#     # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
#     static, timeseries = df.iloc[0:5], df.iloc[5:]

#     feature_dict = {}

    
#     # TODO: 2) extract raw values of time-invariant variables into feature dict
#     if not static.empty:
#         static_map = static.set_index("Variable")["Value"].to_dict()
#     else:
#         static_map = {}

#     for var in static_variables:
#         val = static_map.get(var, np.nan)
#         feature_dict[var] = float(val) if pd.notna(val) else np.nan


#     # TODO  3) extract max of time-varying variables into feature dict
#     if not timeseries.empty:
#         ts_filtered = timeseries[timeseries["Variable"].isin(timeseries_variables)].copy()
#         if not ts_filtered.empty:
#             max_series = ts_filtered.groupby("Variable", sort=False)["Value"].max()
#         else:
#             max_series = pd.Series(dtype=float)
#     else:
#         max_series = pd.Series(dtype=float)

#     for var in timeseries_variables:
#         max_val = max_series.get(var, np.nan)
#         feature_dict[f"max_{var}"] = float(max_val) if pd.notna(max_val) else np.nan

#     return feature_dict

def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]

    df_replaced = df.copy()
    df_replaced["Value"] = pd.to_numeric(df_replaced["Value"], errors="coerce")
    df_replaced["Value"] = df_replaced["Value"].replace(-1, np.nan)

    static, timeseries = df_replaced.iloc[0:5], df_replaced.iloc[5:]
    feature_dict = {}

    # 2) time-invariant variables
    if not static.empty:
        static_map = static.set_index("Variable")["Value"].to_dict()
    else:
        static_map = {}

    for var in static_variables:
        val = static_map.get(var, np.nan)
        feature_dict[var] = float(val) if pd.notna(val) else np.nan

    # 3) Enhanced time-varying variables with multiple statistics
    if not timeseries.empty:
        ts_filtered = timeseries[timeseries["Variable"].isin(timeseries_variables)].copy()
        if not ts_filtered.empty:
            ts_filtered["Hour"] = ts_filtered["Time"].apply(lambda t: int(t.split(":")[0]))

            for var in timeseries_variables:
                var_data = ts_filtered[ts_filtered["Variable"] == var]["Value"]

                # Basic statistics
                feature_dict[f"max_{var}"] = var_data.max() if not var_data.empty else np.nan
                feature_dict[f"min_{var}"] = var_data.min() if not var_data.empty else np.nan
                feature_dict[f"mean_{var}"] = var_data.mean() if not var_data.empty else np.nan
                feature_dict[f"std_{var}"] = var_data.std() if not var_data.empty else np.nan
                feature_dict[f"median_{var}"] = var_data.median() if not var_data.empty else np.nan
                feature_dict[f"count_{var}"] = len(var_data.dropna()) if not var_data.empty else 0

                # Range and percentiles
                if not var_data.empty and len(var_data.dropna()) > 1:
                    feature_dict[f"range_{var}"] = var_data.max() - var_data.min()
                    feature_dict[f"q75_{var}"] = var_data.quantile(0.75)
                    feature_dict[f"q25_{var}"] = var_data.quantile(0.25)
                    feature_dict[f"iqr_{var}"] = var_data.quantile(0.75) - var_data.quantile(0.25)
                else:
                    feature_dict[f"range_{var}"] = np.nan
                    feature_dict[f"q75_{var}"] = np.nan
                    feature_dict[f"q25_{var}"] = np.nan
                    feature_dict[f"iqr_{var}"] = np.nan

                # Time window features (0-24h and 24-48h)
                vals_0_24 = ts_filtered[(ts_filtered["Variable"] == var) & (ts_filtered["Hour"] < 24)]["Value"]
                vals_24_48 = ts_filtered[(ts_filtered["Variable"] == var) & (ts_filtered["Hour"] >= 24) & (ts_filtered["Hour"] < 48)]["Value"]

                # Early period (0-24h)
                feature_dict[f"max_{var}_0_24"] = vals_0_24.max() if not vals_0_24.empty else np.nan
                feature_dict[f"mean_{var}_0_24"] = vals_0_24.mean() if not vals_0_24.empty else np.nan
                feature_dict[f"count_{var}_0_24"] = len(vals_0_24.dropna()) if not vals_0_24.empty else 0

                # Late period (24-48h)
                feature_dict[f"max_{var}_24_48"] = vals_24_48.max() if not vals_24_48.empty else np.nan
                feature_dict[f"mean_{var}_24_48"] = vals_24_48.mean() if not vals_24_48.empty else np.nan
                feature_dict[f"count_{var}_24_48"] = len(vals_24_48.dropna()) if not vals_24_48.empty else 0

                # Trend features (late vs early)
                if not vals_0_24.empty and not vals_24_48.empty:
                    early_mean = vals_0_24.mean()
                    late_mean = vals_24_48.mean()
                    if pd.notna(early_mean) and pd.notna(late_mean) and early_mean != 0:
                        feature_dict[f"trend_{var}"] = (late_mean - early_mean) / abs(early_mean)
                    else:
                        feature_dict[f"trend_{var}"] = np.nan
                else:
                    feature_dict[f"trend_{var}"] = np.nan

        else:
            # Empty case - set all features to NaN or 0 for counts
            for var in timeseries_variables:
                feature_dict[f"max_{var}"] = np.nan
                feature_dict[f"min_{var}"] = np.nan
                feature_dict[f"mean_{var}"] = np.nan
                feature_dict[f"std_{var}"] = np.nan
                feature_dict[f"median_{var}"] = np.nan
                feature_dict[f"count_{var}"] = 0
                feature_dict[f"range_{var}"] = np.nan
                feature_dict[f"q75_{var}"] = np.nan
                feature_dict[f"q25_{var}"] = np.nan
                feature_dict[f"iqr_{var}"] = np.nan
                feature_dict[f"max_{var}_0_24"] = np.nan
                feature_dict[f"mean_{var}_0_24"] = np.nan
                feature_dict[f"count_{var}_0_24"] = 0
                feature_dict[f"max_{var}_24_48"] = np.nan
                feature_dict[f"mean_{var}_24_48"] = np.nan
                feature_dict[f"count_{var}_24_48"] = 0
                feature_dict[f"trend_{var}"] = np.nan
    else:
        # No timeseries data case
        for var in timeseries_variables:
            feature_dict[f"max_{var}"] = np.nan
            feature_dict[f"min_{var}"] = np.nan
            feature_dict[f"mean_{var}"] = np.nan
            feature_dict[f"std_{var}"] = np.nan
            feature_dict[f"median_{var}"] = np.nan
            feature_dict[f"count_{var}"] = 0
            feature_dict[f"range_{var}"] = np.nan
            feature_dict[f"q75_{var}"] = np.nan
            feature_dict[f"q25_{var}"] = np.nan
            feature_dict[f"iqr_{var}"] = np.nan
            feature_dict[f"max_{var}_0_24"] = np.nan
            feature_dict[f"mean_{var}_0_24"] = np.nan
            feature_dict[f"count_{var}_0_24"] = 0
            feature_dict[f"max_{var}_24_48"] = np.nan
            feature_dict[f"mean_{var}_24_48"] = np.nan
            feature_dict[f"count_{var}_24_48"] = 0
            feature_dict[f"trend_{var}"] = np.nan

    return feature_dict



def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: (n, d) feature matrix, which could contain missing values

    Returns:
        X: (n, d) feature matrix, without missing values
    """
    X_col = np.array(X, dtype=float, copy=True)

    col_means = np.nanmean(X_col, axis=0)

    all_nan_cols = np.isnan(col_means)
    if np.any(all_nan_cols):
        global_mean = np.nanmean(X_col)
        fallback = 0.0 if np.isnan(global_mean) else float(global_mean)
        col_means[all_nan_cols] = fallback

    nan_rows, nan_cols = np.where(np.isnan(X_col))
    X_col[nan_rows, nan_cols] = col_means[nan_cols]

    return X_col


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (n, d) feature matrix

    Returns:
        X: (n, d) feature matrix with values that are normalized per column
    """
    X = np.asarray(X, dtype=float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    is_zero = denom == 0
    denom[is_zero] = 1.0
    X_norm = (X - X_min) / denom
    if np.any(is_zero):
        X_norm[:, is_zero] = 0.0

    return X_norm


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: The name of the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel: The name of the Kernel used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but recommended): implement function based on docstring

    if loss == "logistic":
        raise NotImplementedError()
    elif loss == "squared_error":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown loss function: {loss}")


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy",
    bootstrap: bool = False,
    random_state: int = seed,
) -> float | tuple[float, float, float]:
    """
    Calculates the performance metric as evaluated on the true labels y_true versus the predicted scores from
    clf_trained and X. Returns single sample performance if bootstrap is False, otherwise returns the median
    and the empirical 95% confidence interval. You may want to implement an additional helper function to
    reduce code redundancy.

    Args:
        clf_trained: a fitted sklearn estimator
        X: (n, d) feature matrix
        y_true: (n, ) vector of labels in {+1, -1}
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1_score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
        bootstrap: whether to use bootstrap sampling for performance estimation
    
    Returns:
        If bootstrap is False, returns the performance for the specific metric. If bootstrap is True, returns
        the median and the empirical 95% confidence interval.
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    rng = np.random.default_rng(random_state)

    def compute_metric(X_eval, y_eval):
        if metric in ["auroc", "average_precision"]:
            y_score = clf_trained.decision_function(X_eval)
        else:
            y_pred = clf_trained.predict(X_eval)

        if metric == "accuracy":
            return accuracy_score(y_eval, y_pred)
        elif metric == "precision":
            return precision_score(y_eval, y_pred, zero_division=0)
        elif metric == "f1_score":
            return f1_score(y_eval, y_pred, zero_division=0)
        elif metric == "auroc":
            return roc_auc_score(y_eval, y_score)
        elif metric == "average_precision":
            return average_precision_score(y_eval, y_score)
        elif metric == "sensitivity":
            tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[-1, 1]).ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "specificity":
            tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[-1, 1]).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    if not bootstrap:
        return compute_metric(X, y_true)

    n = len(y_true)
    scores = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        X_sample, y_sample = X[idx], y_true[idx]
        scores.append(compute_metric(X_sample, y_sample))

    scores = np.array(scores)
    median = np.percentile(scores, 50)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return float(median), float(lower), float(upper)


def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
        k: the number of folds
        metric: the performance metric (default="accuracy"
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        score = performance(clf, X_val, y_val, metric)
        scores.append(score)

    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))


def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of true labels in {+1, -1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision", "sensitivity",
                and "specificity")
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    best_score = -np.inf
    best_C = None
    best_penalty = None

    for C in C_range:
        for penalty in penalties:
            clf = LogisticRegression(
                penalty=penalty,
                C=C,
                solver="liblinear",
                fit_intercept=False,
                random_state=seed
            )
            mean_score, _, _ = cv_performance(clf, X, y, metric=metric, k=k)

            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_penalty = penalty

    print(f"Best {metric}: {best_score:.4f} with C={best_C}, penalty={best_penalty}")
    return best_C, best_penalty


def select_param_RBF(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    gamma_range: list[float],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of binary labels {1, -1}
        k: the number of folds 
        metric: the performance metric (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    best_score = -np.inf
    best_C = None
    best_gamma = None

    for C in C_range:
        for gamma in gamma_range:
            kr_clf = KernelRidge(
                alpha=1/(2*C),
                kernel="rbf",
                gamma=gamma
            )
            # wrapper
            class KRWrapper:
                def __init__(self, model):
                    self.model = model
                def fit(self, X, y): 
                    self.model.fit(X, y)
                    return self
                def predict(self, X):
                    return np.where(self.model.predict(X) >= 0, 1, -1)
                def decision_function(self, X):
                    return self.model.predict(X)

            wrapper = KRWrapper(kr_clf)
            mean_score, _, _ = cv_performance(wrapper, X, y, metric=metric, k=k)

            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_gamma = gamma

    print(f"Best {metric}: {best_score:.4f} with C={best_C}, gamma={best_gamma}")
    return best_C, best_gamma


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    
    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            clf = LogisticRegression(
                penalty=penalty,
                C=C,
                solver="liblinear",
                fit_intercept=False,
            )
            clf.fit(X, y)
            w = clf.coef_.flatten()

            non_zero_count = np.sum(w != 0)
            norm0.append(non_zero_count)

        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()

def top_coefficients(X_train, y_train, feature_names):
    clf = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        fit_intercept=False
    )
    clf.fit(X_train, y_train)

    coefs = clf.coef_.flatten()

    top_pos_idx = np.argsort(coefs)[-4:][::-1]
    top_neg_idx = np.argsort(coefs)[:4]

    pos_data = [(coefs[i], feature_names[i]) for i in top_pos_idx]
    neg_data = [(coefs[i], feature_names[i]) for i in top_neg_idx]

    table = pd.DataFrame({
        "Positive Coefficient": [f"{coef:.4f}" for coef, _ in pos_data],
        "Feature Name (Pos)": [name for _, name in pos_data],
        "Negative Coefficient": [f"{coef:.4f}" for coef, _ in neg_data],
        "Feature Name (Neg)": [name for _, name in neg_data],
    })

    print(table.to_string(index=False))
    table.to_csv("section2f_top_features.csv", index=False)
    print("\nSaved table to section2f_top_features.csv")

def main():
    print(f"Using Seed = {seed}")
    X_train, y_train, X_test, y_test, feature_names = helper.get_project_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")

    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # Section 2: plot weights and coefficients
    C_range = [10**i for i in range(-3, 4)]
    penalties = ["l1", "l2"]

    plot_weight(X_train, y_train, C_range, penalties)
    print("Saved L0_Norm.png")
    top_coefficients(X_train, y_train, feature_names)

    # Section 3.2(b): Logistic regression with class weights
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        class_weight={-1: 1, 1: 50}, 
        random_state=seed
    )
    clf.fit(X_train, y_train)
    results = []
    for metric in metrics:
        median, lower, upper = performance(clf, X_test, y_test, metric=metric, bootstrap=True)
        results.append([metric, f"{median:.4f}", f"[{lower:.4f}, {upper:.4f}]"])
    table = pd.DataFrame(results, columns=["Performance Measure", "Median", "95% Confidence Interval"])
    print(table.to_string(index=False))
    table.to_csv("section3_2b_results.csv", index=False)
    print("Saved table to section3_2b_results.csv")

    # Section 3.3(a): ROC curves
    clf_base = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        random_state=seed
    )
    clf_base.fit(X_train, y_train)
    y_score_base = clf_base.decision_function(X_test)
    fpr_base, tpr_base, _ = roc_curve(y_test, y_score_base)
    auc_base = auc(fpr_base, tpr_base)

    clf_rew = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        class_weight={-1: 1, 1: 5},
        random_state=seed
    )
    clf_rew.fit(X_train, y_train)
    y_score_rew = clf_rew.decision_function(X_test)
    fpr_rew, tpr_rew, _ = roc_curve(y_test, y_score_rew)
    auc_rew = auc(fpr_rew, tpr_rew)

    plt.figure()
    plt.plot(fpr_base, tpr_base, label=f"Wn=1, Wp=1 (AUC={auc_base:.4f})")
    plt.plot(fpr_rew, tpr_rew, label=f"Wn=1, Wp=5 (AUC={auc_rew:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (C=1.0, L2 penalty)")
    plt.legend(loc="lower right")
    plt.savefig("section3_3a_roc.png", dpi=200)
    plt.close()
    print("Saved ROC plot to section3_3a_roc.png")

    # Section 4.1(b): Compare Logistic Regression vs Kernel Ridge
    print("\nSection 4.1(b): Logistic Regression vs Kernel Ridge")

    log_clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        random_state=seed
    )
    log_clf.fit(X_train, y_train)

    kr_clf = KernelRidge(
        alpha=1/(2*1.0),  # C=1.0
        kernel="linear"
    )
    kr_clf.fit(X_train, y_train)

    class KRWrapper:
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            self.model.fit(X, y)
            return self 

        def predict(self, X):
            return np.where(self.model.predict(X) >= 0, 1, -1)

        def decision_function(self, X):
            return self.model.predict(X)



    kr_wrapper = KRWrapper(kr_clf)


    results = []
    for metric in metrics:
        median_log, lower_log, upper_log = performance(log_clf, X_test, y_test, metric=metric, bootstrap=True)
        median_kr, lower_kr, upper_kr = performance(kr_wrapper, X_test, y_test, metric=metric, bootstrap=True)

        results.append([
            metric,
            f"{median_log:.4f}", f"[{lower_log:.4f}, {upper_log:.4f}]",
            f"{median_kr:.4f}", f"[{lower_kr:.4f}, {upper_kr:.4f}]"
        ])

    table = pd.DataFrame(results, columns=[
        "Metric",
        "Logistic Median", "Logistic 95% CI",
        "Kernel Ridge Median", "Kernel Ridge 95% CI"
    ])
    print(table.to_string(index=False))
    table.to_csv("section4_1b_results.csv", index=False)
    print("Saved table to section4_1b_results.csv")


    gamma_range = [0.001, 0.01, 0.1, 1, 10, 100]
    results_gamma = []
    for gamma in gamma_range:
        kr_clf = KernelRidge(alpha=1/(2*1.0), kernel="rbf", gamma=gamma)
        class KRWrapper:
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                self.model.fit(X, y)
                return self 

            def predict(self, X):
                return np.where(self.model.predict(X) >= 0, 1, -1)

            def decision_function(self, X):
                return self.model.predict(X)

        wrapper = KRWrapper(kr_clf)
        mean_score, min_score, max_score = cv_performance(wrapper, X_train, y_train, metric="auroc", k=5)
        results_gamma.append([gamma, f"{mean_score:.4f} ({min_score:.4f}, {max_score:.4f})"])

    table_gamma = pd.DataFrame(results_gamma, columns=["Gamma", "Mean (Min, Max) CV Performance"])
    print(table_gamma.to_string(index=False))
    table_gamma.to_csv("section4_2b_results.csv", index=False)

    C_range = [0.01, 0.1, 1.0, 10, 100]
    gamma_range = [0.01, 0.1, 1, 10]

    best_C, best_gamma = select_param_RBF(X_train, y_train, C_range, gamma_range, metric="auroc", k=5)

    final_clf = KernelRidge(alpha=1/(2*best_C), kernel="rbf", gamma=best_gamma)
    final_clf.fit(X_train, y_train)

    class KRWrapper:
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            self.model.fit(X, y)
            return self 

        def predict(self, X):
            return np.where(self.model.predict(X) >= 0, 1, -1)

        def decision_function(self, X):
            return self.model.predict(X)


    final_wrapper = KRWrapper(final_clf)

    results_test = []
    for metric in metrics:
        median, lower, upper = performance(final_wrapper, X_test, y_test, metric=metric, bootstrap=True)
        results_test.append([metric, f"{median:.4f}", f"[{lower:.4f}, {upper:.4f}]"])

    table_test = pd.DataFrame(results_test, columns=["Performance Measure", "Median", "95% Confidence Interval"])
    print(table_test.to_string(index=False))
    table_test.to_csv("section4_2c_results.csv", index=False)



if __name__ == "__main__":
    main()
