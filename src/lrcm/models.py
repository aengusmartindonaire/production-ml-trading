"""
Model constructors and tuning helpers (linear, tree-based, etc.).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor


def build_xgb_model(params: Dict[str, Any]) -> XGBRegressor:
    """
    Create an XGBRegressor using a params dict.

    This replaces manually instantiating xgb_champion in the notebook.
    """
    base = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    base.update(params)
    return XGBRegressor(**base)


def xgb_objective(trial, X, y, n_splits: int = 5) -> float:
    """
    Optuna objective function for tuning your XGB model.
    This is the same logic as in your notebook, but now
    takes X and y explicitly instead of using globals.
    """
    # Hyperparameter search space (same ranges you used)
    n_estimators = trial.suggest_int("n_estimators", 200, 1000)
    max_depth = trial.suggest_int("max_depth", 2, 9)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.10)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bt = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1.0, 100.0)

    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bt,
        reg_lambda=reg_lambda,
    )

    model = build_xgb_model(params)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    # Optuna will maximize this
    return float(scores.mean())

