"""
EECS 445 Fall 2025
Problem 5: Challenge Classifier (Advanced Feature Engineering + Validation Metrics)
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix, make_scorer
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import helper
import project1
import warnings
warnings.filterwarnings('ignore')

SEED = 4452025
np.random.seed(SEED)


# ==============================
# Utility Functions
# ==============================

def sweep_best_threshold(y_true, scores):
    """Find threshold that maximizes F1-score."""
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_f1, best_t = -1, 0
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


class AdvancedPreprocessor:
    """Simplified advanced preprocessing with feature selection"""

    def __init__(self, feature_selection_k=60):
        self.feature_selection_k = feature_selection_k

    def fit_transform(self, X, y):
        # Simple imputation + scaling
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        X_imp = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imp)

        # Feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(self.feature_selection_k, X.shape[1]))
        X_sel = selector.fit_transform(X_scaled, y)

        self.imputer = imputer
        self.scaler = scaler
        self.selector = selector
        self.selected_features = np.array(X.columns)[selector.get_support()]

        return X_sel

    def transform(self, X):
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return self.selector.transform(X_scaled)


def custom_f1_scorer(y_true, y_score):
    """Custom F1 scorer with optimal threshold search"""
    thresholds = np.linspace(y_score.min(), y_score.max(), 100)
    best_f1 = 0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        best_f1 = max(best_f1, f1)
    return best_f1


def train_best_model(X_train, y_train):
    """Train logistic regression variants with CV, return best"""
    preprocessor = AdvancedPreprocessor(feature_selection_k=60)
    X_proc = preprocessor.fit_transform(X_train, y_train)

    models = {
        'lr_l1': LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=SEED),
        'lr_l2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000, random_state=SEED),
        'lr_elastic': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                         max_iter=2000, random_state=SEED)
    }

    param_grids = {
        'lr_l1': {'C': [0.01, 0.1, 1, 10]},
        'lr_l2': {'C': [0.01, 0.1, 1, 10]},
        'lr_elastic': {'C': [0.1, 1, 10], 'l1_ratio': [0.3, 0.5, 0.7]}
    }

    best_model, best_score, best_name = None, -1, ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        print(f"🔍 Training {name}...")
        grid = GridSearchCV(model, param_grids[name], scoring="roc_auc", cv=cv, n_jobs=-1)
        grid.fit(X_proc, y_train)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

        print(f"{name} best params: {grid.best_params_}, AUROC={grid.best_score_:.4f}")

    print(f"\n✅ Best model: {best_name}, AUROC={best_score:.4f}")
    return best_model, preprocessor


# ==============================
# Main
# ==============================

def main():
    # Load challenge data
    X_train, y_train, X_challenge, feature_names = helper.get_challenge_data()
    print(f"Train set: {X_train.shape}, Challenge set: {X_challenge.shape}")

    # Convert to DataFrame for preprocessing
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_challenge_df = pd.DataFrame(X_challenge, columns=feature_names)

    # Split validation set for metrics
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_df, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    # Train model
    best_model, preprocessor = train_best_model(X_tr, y_tr)

    # Validation evaluation
    X_val_proc = preprocessor.transform(X_val)
    y_score_val = best_model.predict_proba(X_val_proc)[:, 1]
    thr, f1_val = sweep_best_threshold((y_val == 1).astype(int), y_score_val)
    auroc_val = roc_auc_score((y_val == 1).astype(int), y_score_val)

    y_pred_val = (y_score_val >= thr).astype(int)
    cm = confusion_matrix((y_val == 1).astype(int), y_pred_val)

    print(f"\n=== Validation Results ===")
    print(f"Best threshold={thr:.4f}, F1={f1_val:.4f}, AUROC={auroc_val:.4f}")
    print("\nConfusion Matrix (2x2):")
    print(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

    # Final training on full set
    X_full_proc = preprocessor.fit_transform(X_train_df, y_train)
    best_model.fit(X_full_proc, y_train)

    # Predict on challenge
    X_challenge_proc = preprocessor.transform(X_challenge_df)
    y_score = best_model.decision_function(X_challenge_proc)
    y_label = best_model.predict(X_challenge_proc).astype(int)

    helper.save_challenge_predictions(y_label, y_score, uniqname="lyjie")
    print("\n✅ Saved challenge predictions to lyjie.csv")


if __name__ == "__main__":
    main()
