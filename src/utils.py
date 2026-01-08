from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metrics used in the notebook + production."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae}


def build_results_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Notebook-style results table: actual, predicted, diff, diff_pct."""
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    df["diff"] = df["actual"] - df["predicted"]
    denom = df["actual"].replace(0, np.nan)
    df["diff_pct"] = (df["diff"] / denom) * 100.0
    return df


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    param_grids: Dict[str, Dict[str, Any]],
    *,
    scoring: str = "r2",
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 0,
    refit: bool = True,
) -> Tuple[pd.DataFrame, str, Any, Dict[str, Any]]:
    """
    Evaluate multiple models using GridSearchCV, returning a leaderboard + winner.

    Returns
    -------
    leaderboard_df : pd.DataFrame
        Columns: model, r2, rmse, mae, best_params
    best_model_name : str
    best_estimator : fitted estimator
    best_params : dict
    """
    rows = []

    best_name: Optional[str] = None
    best_estimator = None
    best_params: Dict[str, Any] = {}
    best_score = -1e18  # for r2; higher is better

    for name, model in models.items():
        grid = param_grids.get(name, {})
        if grid is None:
            grid = {}

        gs = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=refit,
        )

        gs.fit(X_train, y_train)

        fitted = gs.best_estimator_ if hasattr(gs, "best_estimator_") else model
        y_pred = fitted.predict(X_test)

        metrics = regression_metrics(y_test, y_pred)
        row = {
            "model": name,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "best_params": gs.best_params_ if hasattr(gs, "best_params_") else {},
        }
        rows.append(row)

        # Pick winner by r2 (same as notebook)
        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_name = name
            best_estimator = fitted
            best_params = row["best_params"]

    leaderboard_df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)

    if best_name is None or best_estimator is None:
        raise RuntimeError("No model evaluated successfully (best_estimator missing).")

    return leaderboard_df, best_name, best_estimator, best_params
