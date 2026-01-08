"""
model_trainer.py
================

Notebook-close + production-grade model trainer for regression (Shield Insurance premium prediction).

What this module does
---------------------
1) Runs DataTransformation (your production pipeline) to generate:
   - train_arr: numpy array (n_train, n_features + 1), last column is target
   - test_arr:  numpy array (n_test,  n_features + 1), last column is target
   - preprocessor.pkl: sklearn ColumnTransformer used to create feature matrix

2) Trains and evaluates multiple regression models (notebook parity):
   - LinearRegression (baseline)
   - Ridge (baseline)
   - Lasso (baseline)
   - XGBoostRegressor with RandomizedSearchCV (primary model)

3) Produces notebook-style artifacts:
   - model_leaderboard.csv
   - model.pkl (winner)
   - results_predictions.csv (actual, predicted, diff, diff_pct)
   - residual_hist.png, residual_scatter.png
   - residual_hist_seaborn.png (optional)
   - feature_importance.csv
   - SHAP outputs (robust, best effort; never crashes training)
       * shap_values.npz
       * shap_summary_dot.png
       * shap_summary_bar.png
       * shap_dependence_top_feature.png (optional)

Why your run failed
-------------------
Your pipeline succeeded until SHAP TreeExplainer crashed due to an XGBoost base_score parsing issue:
  ValueError: could not convert string to float: '[1.9488639E4]'

This file fixes it by:
- Trying TreeExplainer first (fastest)
- If it fails, falling back to shap.Explainer(model.predict, X_background)
- If both fail, logs and skips SHAP (training still succeeds)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from src.exception import CustomException
from src.logger import logging

# Uses your production transformation pipeline
from src.components.data_transformation import DataTransformation, DataTransformationConfig


# =============================================================================
# CONFIG
# =============================================================================
@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration controlling training + artifacts.
    Designed for easy tuning without touching core logic.
    """
    artifacts_dir: Path = Path("artifacts")

    # Winner artifacts
    sklearn_model_path: Path = Path("artifacts/model.pkl")

    # Metadata outputs
    metrics_path: Path = Path("artifacts/model_metrics.txt")
    winner_meta_path: Path = Path("artifacts/model_winner.txt")
    leaderboard_path: Path = Path("artifacts/model_leaderboard.csv")

    # Notebook-style predictions export
    results_df_path: Path = Path("artifacts/results_predictions.csv")

    # Interpretability artifacts
    feature_importance_path: Path = Path("artifacts/feature_importance.csv")

    # Diagnostics artifacts
    residual_hist_path: Path = Path("artifacts/residual_hist.png")
    residual_scatter_path: Path = Path("artifacts/residual_scatter.png")
    seaborn_residual_hist_path: Path = Path("artifacts/residual_hist_seaborn.png")

    # SHAP artifacts
    shap_values_path: Path = Path("artifacts/shap_values.npz")
    shap_summary_dot_path: Path = Path("artifacts/shap_summary_dot.png")
    shap_summary_bar_path: Path = Path("artifacts/shap_summary_bar.png")
    shap_dependence_path: Path = Path("artifacts/shap_dependence_top_feature.png")

    # Training controls
    random_state: int = 42
    n_jobs: int = -1
    val_size: float = 0.2

    # XGBoost tuning controls
    xgb_n_iter: int = 20
    xgb_cv: int = 3
    xgb_scoring: str = "r2"
    xgb_early_stopping_rounds: int = 100  # best-effort

    # SHAP controls
    enable_shap: bool = True
    shap_max_background: int = 500   # subsample for background
    shap_max_explain: int = 1000     # subsample for explanation


# =============================================================================
# TRAINER
# =============================================================================
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig | None = None):
        self.config = config or ModelTrainerConfig()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _ensure_artifacts_dir(self) -> None:
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _write_text(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def _split_xy(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        arr is expected to be shape (n, p+1), last col is y.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError("Expected numpy array input.")
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Transformed array must be 2D with at least 2 columns (X + y).")
        return arr[:, :-1], arr[:, -1]

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Notebook-style regression metrics.
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"r2": r2, "rmse": rmse, "mae": mae}

    def _try_get_feature_names(self, preprocessor_path: str) -> Optional[np.ndarray]:
        """
        Extract feature names from ColumnTransformer if supported.
        Helpful for feature importance + SHAP feature labeling.
        """
        try:
            pre = joblib.load(preprocessor_path)
            if hasattr(pre, "get_feature_names_out"):
                return pre.get_feature_names_out()
            return None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Notebook-style exports
    # -------------------------------------------------------------------------
    def _save_results_df(self, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
        """
        Saves a notebook-style prediction table.
        """
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
        df["diff"] = df["actual"] - df["predicted"]
        denom = df["actual"].replace(0, np.nan)
        df["diff_pct"] = (df["diff"] / denom) * 100.0
        df.to_csv(path, index=False)
        logging.info("Saved notebook-style results_df: %s", path)

    def _save_feature_importance(self, model: Any, feature_names: Optional[np.ndarray], out_path: Path) -> None:
        """
        Saves coefficients or feature_importances_ if present.
        """
        try:
            if hasattr(model, "coef_"):
                imp = np.asarray(model.coef_).reshape(-1)
                kind = "coef_"
            elif hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_).reshape(-1)
                kind = "feature_importances_"
            else:
                logging.info("Model exposes no coef_ or feature_importances_; skipping.")
                return

            if feature_names is not None and len(feature_names) == len(imp):
                df = pd.DataFrame({"feature": feature_names, "importance": imp})
            else:
                df = pd.DataFrame({"feature_index": np.arange(len(imp)), "importance": imp})

            df["abs_importance"] = df["importance"].abs()
            df = df.sort_values("abs_importance", ascending=False).drop(columns=["abs_importance"])
            df.to_csv(out_path, index=False)
            logging.info("Saved %s to: %s", kind, out_path)
        except Exception as e:
            logging.warning("Failed to save feature importance: %s", str(e))

    def _save_residual_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Saves:
        - residual distribution histogram
        - residuals vs predicted scatter
        - optional seaborn histplot (+kde)
        """
        residuals = y_true - y_pred

        # Matplotlib (always attempt)
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=40)
            plt.title("Residual Distribution (y_true - y_pred)")
            plt.xlabel("Residual")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(self.config.residual_hist_path)
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.35)
            plt.axhline(0.0, linestyle="--")
            plt.title("Residuals vs Predicted")
            plt.xlabel("Predicted")
            plt.ylabel("Residual")
            plt.tight_layout()
            plt.savefig(self.config.residual_scatter_path)
            plt.close()

            logging.info("Saved residual plots: %s | %s",
                         self.config.residual_hist_path, self.config.residual_scatter_path)
        except Exception as e:
            logging.warning("Matplotlib residual plots skipped: %s", str(e))

        # Optional seaborn histplot
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True, bins=40)
            plt.title("Residual Distribution (Seaborn histplot + KDE)")
            plt.xlabel("Residual")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.config.seaborn_residual_hist_path)
            plt.close()

            logging.info("Saved seaborn residual histplot: %s", self.config.seaborn_residual_hist_path)
        except Exception:
            logging.info("Seaborn not available (or failed). Skipping seaborn histplot.")

    # -------------------------------------------------------------------------
    # XGBoost tuning (RandomizedSearchCV)
    # -------------------------------------------------------------------------
    def _train_xgboost_with_random_search(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Optional[Any]:
        """
        XGBoost training that mirrors typical notebook tuning via RandomizedSearchCV.
        Early stopping is best-effort (some versions do not support in this setup).
        """
        try:
            import xgboost as xgb
            from xgboost import XGBRegressor
        except ImportError:
            logging.warning("xgboost not installed -> skipping XGBoost.")
            return None

        # Note: we DO NOT set base_score explicitly; XGBoost will infer it.
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        param_dist = {
            "n_estimators": [300, 600, 900, 1200, 2000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "min_child_weight": [1, 2, 5],
        }

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=self.config.xgb_n_iter,
            scoring=self.config.xgb_scoring,
            cv=self.config.xgb_cv,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=0,
        )

        logging.info("Running RandomizedSearchCV for XGBoost (n_iter=%s, cv=%s).",
                     self.config.xgb_n_iter, self.config.xgb_cv)
        search.fit(X_train_full, y_train_full)

        best_model = search.best_estimator_
        logging.info("XGBoost best params: %s", search.best_params_)
        logging.info("XGBoost best CV score: %.6f", float(search.best_score_))

        # Best-effort early stopping refinement
        logging.info("Attempting early-stopping refinement (best model).")
        try:
            cb = [xgb.callback.EarlyStopping(rounds=self.config.xgb_early_stopping_rounds, save_best=True)]
            best_model.fit(
                X_train_full,
                y_train_full,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=cb,
            )
            logging.info("Refit with callback early stopping.")
        except Exception:
            try:
                best_model.fit(
                    X_train_full,
                    y_train_full,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=self.config.xgb_early_stopping_rounds,
                )
                logging.info("Refit with early_stopping_rounds.")
            except Exception:
                logging.warning("Early stopping not supported; keeping CV-best model.")

        return best_model

    # -------------------------------------------------------------------------
    # SHAP (Robust: never fails training)
    # -------------------------------------------------------------------------
    def _run_shap_analysis(
        self,
        model: Any,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        feature_names: Optional[np.ndarray],
    ) -> None:
        """
        Robust SHAP that NEVER crashes the pipeline.

        Strategy:
          1) Try shap.TreeExplainer(model, data=background)  [fast, best for XGB]
             - This is where your error occurred due to base_score parsing in SHAP.
          2) If TreeExplainer fails, fallback to:
                shap.Explainer(model.predict, background)
             - More general and avoids the internal XGB model loader.
          3) If fallback fails, log and skip SHAP.

        This guarantees training artifacts still get saved.
        """
        if not self.config.enable_shap:
            logging.info("SHAP disabled by config.")
            return

        # Only run SHAP if winner is XGBoost-like
        cls_name = model.__class__.__name__.lower()
        if "xgb" not in cls_name and "xgboost" not in cls_name:
            logging.info("SHAP skipped: model is not XGBoost-like (%s).", model.__class__.__name__)
            return

        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("shap not installed -> skipping SHAP.")
            return

        rng = np.random.default_rng(self.config.random_state)

        # Subsample background
        X_bg = X_background
        if X_bg.shape[0] > self.config.shap_max_background:
            idx = rng.choice(X_bg.shape[0], size=self.config.shap_max_background, replace=False)
            X_bg = X_bg[idx]

        # Subsample explain
        X_ex = X_explain
        if X_ex.shape[0] > self.config.shap_max_explain:
            idx = rng.choice(X_ex.shape[0], size=self.config.shap_max_explain, replace=False)
            X_ex = X_ex[idx]

        if feature_names is None:
            feature_names = np.array([f"f{i}" for i in range(X_ex.shape[1])])

        # ---------------------------
        # Attempt 1: TreeExplainer
        # ---------------------------
        try:
            logging.info("Running SHAP TreeExplainer: background=%s, explain=%s", X_bg.shape, X_ex.shape)
            explainer = shap.TreeExplainer(model, data=X_bg)
            shap_values = explainer.shap_values(X_ex)
            expected_value = getattr(explainer, "expected_value", None)

            np.savez_compressed(
                self.config.shap_values_path,
                shap_values=shap_values,
                X_explain=X_ex,
                expected_value=np.array(expected_value) if expected_value is not None else None,
            )
            logging.info("Saved SHAP values: %s", self.config.shap_values_path)

            # Summary dot
            plt.figure(figsize=(12, 7))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(self.config.shap_summary_dot_path, dpi=160)
            plt.close()
            logging.info("Saved SHAP dot plot: %s", self.config.shap_summary_dot_path)

            # Summary bar
            plt.figure(figsize=(12, 7))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(self.config.shap_summary_bar_path, dpi=160)
            plt.close()
            logging.info("Saved SHAP bar plot: %s", self.config.shap_summary_bar_path)

            # Optional dependence plot (best-effort)
            try:
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                top_idx = int(np.argmax(mean_abs))
                top_feature = feature_names[top_idx]

                plt.figure(figsize=(12, 7))
                shap.dependence_plot(
                    ind=top_idx,
                    shap_values=shap_values,
                    features=X_ex,
                    feature_names=feature_names,
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(self.config.shap_dependence_path, dpi=160)
                plt.close()
                logging.info("Saved SHAP dependence plot (top=%s): %s", top_feature, self.config.shap_dependence_path)
            except Exception:
                logging.info("Dependence plot skipped (optional).")

            return  # ✅ SHAP succeeded via TreeExplainer

        except Exception as e:
            # This is where your base_score parsing error occurs.
            logging.warning("TreeExplainer failed (will fallback). Reason: %s", str(e))

        # ---------------------------
        # Attempt 2: model-agnostic Explainer fallback
        # ---------------------------
        try:
            logging.info("Running SHAP fallback via shap.Explainer(model.predict, background).")

            # Using model.predict avoids SHAP's XGB model loader entirely.
            explainer = shap.Explainer(model.predict, X_bg)
            explanation = explainer(X_ex)

            shap_values = explanation.values
            base_values = getattr(explanation, "base_values", None)

            np.savez_compressed(
                self.config.shap_values_path,
                shap_values=shap_values,
                X_explain=X_ex,
                expected_value=np.array(base_values) if base_values is not None else None,
            )
            logging.info("Saved SHAP values (fallback): %s", self.config.shap_values_path)

            # Summary dot
            plt.figure(figsize=(12, 7))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(self.config.shap_summary_dot_path, dpi=160)
            plt.close()
            logging.info("Saved SHAP dot plot (fallback): %s", self.config.shap_summary_dot_path)

            # Summary bar
            plt.figure(figsize=(12, 7))
            shap.summary_plot(shap_values, X_ex, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(self.config.shap_summary_bar_path, dpi=160)
            plt.close()
            logging.info("Saved SHAP bar plot (fallback): %s", self.config.shap_summary_bar_path)

            return  # ✅ SHAP succeeded via fallback

        except Exception as e:
            # Never crash training because SHAP failed
            logging.warning("SHAP fallback failed; skipping SHAP. Reason: %s", str(e))
            return

    # -------------------------------------------------------------------------
    # Main training pipeline
    # -------------------------------------------------------------------------
    def initiate_model_trainer(
        self,
        train_arr: np.ndarray,
        test_arr: np.ndarray,
        preprocessor_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        logging.info("Starting model training pipeline.")
        try:
            self._ensure_artifacts_dir()

            X_train_full, y_train_full = self._split_xy(train_arr)
            X_test, y_test = self._split_xy(test_arr)

            logging.info("Train(full): X=%s y=%s", X_train_full.shape, y_train_full.shape)
            logging.info("Test:       X=%s y=%s", X_test.shape, y_test.shape)

            # Internal split (used for early stopping attempt)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=self.config.val_size,
                random_state=self.config.random_state,
            )
            logging.info("Internal split -> Train=%s | Val=%s", X_train.shape, X_val.shape)

            feature_names = self._try_get_feature_names(preprocessor_path) if preprocessor_path else None

            leaderboard_rows: List[Dict[str, Any]] = []

            # Track best by R2
            best_name: Optional[str] = None
            best_model: Optional[Any] = None
            best_metrics: Dict[str, float] = {"r2": -1e18, "rmse": float("inf"), "mae": float("inf")}

            # ----------------------------
            # Baselines (notebook style)
            # ----------------------------
            logging.info("Training baseline: LinearRegression")
            lr = LinearRegression()
            lr.fit(X_train_full, y_train_full)
            preds = lr.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "LinearRegression", **metrics})
            best_name, best_model, best_metrics = "LinearRegression", lr, metrics

            logging.info("Training baseline: Ridge")
            ridge = Ridge(random_state=self.config.random_state)
            ridge.fit(X_train_full, y_train_full)
            preds = ridge.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "Ridge", **metrics})
            if metrics["r2"] > best_metrics["r2"]:
                best_name, best_model, best_metrics = "Ridge", ridge, metrics

            logging.info("Training baseline: Lasso")
            lasso = Lasso(random_state=self.config.random_state, max_iter=50000)
            lasso.fit(X_train_full, y_train_full)
            preds = lasso.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "Lasso", **metrics})
            if metrics["r2"] > best_metrics["r2"]:
                best_name, best_model, best_metrics = "Lasso", lasso, metrics

            # ----------------------------
            # Primary: XGBoost tuned
            # ----------------------------
            xgb_model = self._train_xgboost_with_random_search(X_train_full, y_train_full, X_val, y_val)
            if xgb_model is not None:
                preds = xgb_model.predict(X_test)
                metrics = self._evaluate(y_test, preds)
                leaderboard_rows.append({"model": "XGBoost(RandomSearch)", **metrics})
                if metrics["r2"] > best_metrics["r2"]:
                    best_name, best_model, best_metrics = "XGBoost(RandomSearch)", xgb_model, metrics

            if best_model is None or best_name is None:
                raise RuntimeError("No model trained successfully. Check logs.")

            # Save leaderboard
            df_lb = pd.DataFrame(leaderboard_rows).sort_values(by="r2", ascending=False)
            df_lb.to_csv(self.config.leaderboard_path, index=False)
            logging.info("Saved leaderboard: %s", self.config.leaderboard_path)

            # Save winner model
            joblib.dump(best_model, self.config.sklearn_model_path)
            logging.info("Saved winner model: %s", self.config.sklearn_model_path)

            # Notebook-style results df
            best_preds = best_model.predict(X_test)
            self._save_results_df(y_test, best_preds, self.config.results_df_path)

            # Residual diagnostics
            self._save_residual_diagnostics(y_test, best_preds)

            # Feature importance
            self._save_feature_importance(best_model, feature_names, self.config.feature_importance_path)

            # ✅ Robust SHAP (will never crash training)
            self._run_shap_analysis(
                model=best_model,
                X_background=X_train_full,
                X_explain=X_test,
                feature_names=feature_names,
            )

            # Summary text
            summary = (
                f"Best model: {best_name}\n"
                f"Saved to: {self.config.sklearn_model_path}\n"
                f"TEST R2: {best_metrics['r2']:.6f}\n"
                f"TEST RMSE: {best_metrics['rmse']:.6f}\n"
                f"TEST MAE: {best_metrics['mae']:.6f}\n\n"
                f"Artifacts:\n"
                f"- Leaderboard: {self.config.leaderboard_path}\n"
                f"- Results DF: {self.config.results_df_path}\n"
                f"- Feature importance: {self.config.feature_importance_path}\n"
                f"- Residual hist: {self.config.residual_hist_path}\n"
                f"- Residual scatter: {self.config.residual_scatter_path}\n"
                f"- SHAP values: {self.config.shap_values_path}\n"
                f"- SHAP dot: {self.config.shap_summary_dot_path}\n"
                f"- SHAP bar: {self.config.shap_summary_bar_path}\n"
            )
            self._write_text(self.config.metrics_path, summary)
            self._write_text(self.config.winner_meta_path, f"{best_name} | {self.config.sklearn_model_path}\n")

            logging.info("Winner: %s | R2=%.4f", best_name, best_metrics["r2"])
            return best_name, best_metrics

        except Exception as e:
            logging.exception("Model training pipeline failed.")
            raise CustomException(e, sys)


# =============================================================================
# SCRIPT ENTRYPOINT
# =============================================================================
def _run_as_script() -> None:
    """
    End-to-end script:
      - Runs DataTransformation
      - Runs ModelTrainer
    """
    logging.info("Running model_trainer as a script.")
    try:
        tcfg = DataTransformationConfig()
        if not tcfg.train_data_path.exists() or not tcfg.test_data_path.exists():
            raise FileNotFoundError(
                "Missing artifacts/train.csv or artifacts/test.csv.\n"
                "Run ingestion first to generate them."
            )

        transformer = DataTransformation(tcfg)
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path=str(tcfg.train_data_path),
            test_path=str(tcfg.test_data_path),
        )
        logging.info("Preprocessor used: %s", preprocessor_path)

        trainer = ModelTrainer(ModelTrainerConfig())
        best_name, best_metrics = trainer.initiate_model_trainer(
            train_arr=train_arr,
            test_arr=test_arr,
            preprocessor_path=preprocessor_path,
        )

        print("✅ Model training complete")
        print("Best model:", best_name)
        print("Metrics:", best_metrics)
        print("Artifacts folder:", str(trainer.config.artifacts_dir.resolve()))

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    _run_as_script()
