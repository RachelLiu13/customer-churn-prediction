"""Churn models: a logistic-regression baseline and a gradient-boosted model, same preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import NUMERIC, categorical_columns

SEED = 42


def preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_columns(df)),
    ])


def make_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Logistic regression": Pipeline([
            ("prep", preprocessor(df)),
            ("clf", LogisticRegression(max_iter=2000, C=0.5)),
        ]),
        "XGBoost": Pipeline([
            ("prep", preprocessor(df)),
            ("clf", XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                                  eval_metric="logloss")),
        ]),
    }


def split(df: pd.DataFrame, test_size: float = 0.25):
    """Stratified hold-out split; the test set is touched once, at the end."""
    X, y = df.drop(columns="churn"), df["churn"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def evaluate(models: dict, X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """5-fold CV on the training set, then a single score on the hold-out set."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, test_scores = [], {}
    for name, model in models.items():
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        test_scores[name] = p
        rows.append({"model": name,
                     "cv_roc_auc": roc_auc_score(y_train, oof), "cv_pr_auc": average_precision_score(y_train, oof),
                     "test_roc_auc": roc_auc_score(y_test, p), "test_pr_auc": average_precision_score(y_test, p)})
    return pd.DataFrame(rows).set_index("model"), test_scores


def top_drivers(model: Pipeline, n: int = 10) -> pd.Series:
    """Largest logistic-regression coefficients (on standardized / one-hot features)."""
    names = model.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names)
    coef.index = coef.index.str.replace("num__", "").str.replace("cat__", "")
    return coef.reindex(coef.abs().sort_values(ascending=False).index).head(n)
