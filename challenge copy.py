import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import helper
import yaml
import random
from sklearn.metrics import confusion_matrix


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = pd.DataFrame(X, columns=self.feature_names)
        X_new["std_all"] = np.std(X, axis=1)
        X_new["range_all"] = np.max(X, axis=1) - np.min(X, axis=1)
        if set(["max_HR", "max_SysABP"]).issubset(X_new.columns):
            X_new["shock_index"] = X_new["max_HR"] / (X_new["max_SysABP"] + 1e-5)
        if set(["max_SysABP", "max_DiasABP"]).issubset(X_new.columns):
            X_new["pulse_pressure"] = X_new["max_SysABP"] - X_new["max_DiasABP"]
        if set(["max_BUN", "max_Creatinine"]).issubset(X_new.columns):
            X_new["bun_cr_ratio"] = X_new["max_BUN"] / (X_new["max_Creatinine"] + 1e-5)
        X_new = X_new.replace([np.inf, -np.inf], np.nan)
        return X_new


def combined_score(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0).astype(int)
    y_pred[y_pred == 0] = -1
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    return 0.6 * auc + 0.4 * f1


def main():

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    seed = config.get("seed", 0)
    np.random.seed(seed)
    random.seed(seed)

    uniqname = "ziyuxu"

    # Only use challenge data for 30-day mortality prediction
    X_train_full, y_train_full, X_challenge, feature_names = helper.get_challenge_data()

    print(f"Train(full): {X_train_full.shape}, Challenge: {X_challenge.shape}")
    print(f"Training on 30-day mortality data with {len(y_train_full)} samples")

    categorical_features = []
    if "ICUType" in feature_names:
        categorical_features = ["ICUType"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )


    pipe = Pipeline([
        ("fe", FeatureEngineer(feature_names=feature_names)),
        ("pre", preprocessor),
        ("poly", PolynomialFeatures(interaction_only=True, include_bias=False)),
        ("clf", LogisticRegression(
            solver="liblinear",
            fit_intercept=False,
            random_state=seed,
            max_iter=3000
        ))
    ])

    param_grid = [
        {
            "poly__degree": [1],
            "clf__penalty": ["l1"],
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "clf__class_weight": ["balanced", None, {-1: 1, 1: 2}, {-1: 1, 1: 3}, {-1: 1, 1: 5}, {-1: 1, 1: 10}],
        },
        {
            "poly__degree": [1],
            "clf__penalty": ["l2"],
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "clf__class_weight": ["balanced", None, {-1: 1, 1: 2}, {-1: 1, 1: 3}, {-1: 1, 1: 5}, {-1: 1, 1: 10}],
        }
    ]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True
    )

    print("\n=== Starting Grid Search ===")
    print(f"Total parameter combinations: {len(param_grid[0]['clf__C']) * len(param_grid[0]['clf__class_weight']) * 2}")

    grid.fit(X_train_full, y_train_full)
    best_clf = grid.best_estimator_

    print(f"\n=== Best Parameters Found ===")
    print(f"Best C: {grid.best_params_['clf__C']}")
    print(f"Best penalty: {grid.best_params_['clf__penalty']}")
    print(f"Best class_weight: {grid.best_params_['clf__class_weight']}")
    print(f"Best CV AUROC score: {grid.best_score_:.4f}")

    # Detailed cross-validation evaluation
    print(f"\n=== Detailed Cross-Validation Results ===")
    cv_auroc = cross_val_score(grid.best_estimator_, X_train_full, y_train_full, cv=5, scoring='roc_auc')
    cv_f1 = cross_val_score(grid.best_estimator_, X_train_full, y_train_full, cv=5, scoring='f1')

    print(f"5-Fold CV AUROC: {cv_auroc.mean():.4f} (+/- {cv_auroc.std() * 2:.4f})")
    print(f"5-Fold CV F1: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    print(f"Individual AUROC scores: {[f'{score:.4f}' for score in cv_auroc]}")
    print(f"Individual F1 scores: {[f'{score:.4f}' for score in cv_f1]}")

    y_pred_chal = best_clf.predict(X_challenge).astype(int)
    y_score_chal = best_clf.decision_function(X_challenge)

    # Training set performance evaluation
    y_train_pred = best_clf.predict(X_train_full)
    y_train_score = best_clf.decision_function(X_train_full)

    train_auroc = roc_auc_score(y_train_full, y_train_score)
    train_f1 = f1_score(y_train_full, y_train_pred)

    print(f"\n=== Training Set Performance ===")
    print(f"Training AUROC: {train_auroc:.4f}")
    print(f"Training F1: {train_f1:.4f}")

    # Confusion Matrix on Training Full
    cm_train = confusion_matrix(y_train_full, y_train_pred, labels=[-1, 1])
    print(f"\n=== Confusion Matrix on Training Set ===")
    print(f"                 Predicted")
    print(f"Actual    Survived  Died")
    print(f"Survived    {cm_train[0][0]:6d} {cm_train[0][1]:5d}")
    print(f"Died        {cm_train[1][0]:6d} {cm_train[1][1]:5d}")
    print(f"Total samples: {cm_train.sum()}")

    from helper import save_challenge_predictions
    save_challenge_predictions(y_pred_chal, y_score_chal, uniqname)
    print(f"\n=== Final Output ===")
    print(f"Saved predictions to {uniqname}.csv")
    print(f"Challenge set predictions: {len(y_pred_chal)} samples")



if __name__ == "__main__":
    main()
