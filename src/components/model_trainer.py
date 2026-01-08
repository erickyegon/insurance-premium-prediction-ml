"""
model_trainer.py
================

Purpose
-------
Production-grade trainer for regression (insurance premiums). Integrates with transformation pipeline.
Trains baselines (Linear, Ridge, Lasso) and tuned XGBoost; evaluates; saves artifacts including interpretability.

What this module does
---------------------
1. Runs transformation to get arrays/preprocessor.
2. Trains/evaluates models (notebook parity).
3. Saves: model, leaderboard, predictions, residuals plots, importance, SHAP.
4. Fixes SHAP issues with robust fallbacks.

Visualization Improvements (Modern Principles)
----------------------------------------------
- Seaborn theme: Whitegrid, muted palette for clarity/colorblindness.
- Clarity: Titles, labels, legends, grids, alpha for scatters.
- Visibility: KDE in hists, reg lines in scatters, high DPI (300).
- Professionalism: Tight layouts, annotations, consistent styling.
- Enhanced: Added reg line to residual scatter; better SHAP sizing.

Design Goals (Advanced / Production)
------------------------------------
- Config-driven: Toggles, paths, hyperparameters.
- Robust: SHAP never crashes training; graceful lib skips.
- Explainability: Feature importance, SHAP for insights.
- Metrics: R2, RMSE, MAE (notebook-style).
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

# Transformation integration
from src.components.data_transformation import DataTransformation, DataTransformationConfig

# Optional imports for visualizations and SHAP
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy.stats import gaussian_kde, linregress
except ImportError:
    gaussian_kde = None
    linregress = None


# =============================================================================
# CONFIG
# =============================================================================
@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Config for training/artifacts. Frozen for safety.
    
    Attributes:
        artifacts_dir: Root outputs.
        sklearn_model_path: Winner model.
        metrics_path/winner_meta_path/leaderboard_path: Metadata.
        results_df_path: Predictions CSV.
        feature_importance_path: Importance CSV.
        residual_hist_path/residual_scatter_path: Diagnostics (seaborn hist removed as integrated).
        shap_values_path/shap_summary_dot_path/shap_summary_bar_path/shap_dependence_path: Interpretability.
        random_state: Reproducibility.
        n_jobs: Parallelism.
        val_size: Internal val split.
        xgb_n_iter/xgb_cv/xgb_scoring/xgb_early_stopping_rounds: XGBoost tuning.
        enable_shap: Toggle SHAP.
        shap_max_background/shap_max_explain: Subsampling for efficiency.
    """
    artifacts_dir: Path = Path("artifacts")
    sklearn_model_path: Path = Path("artifacts/model.pkl")
    metrics_path: Path = Path("artifacts/model_metrics.txt")
    winner_meta_path: Path = Path("artifacts/model_winner.txt")
    leaderboard_path: Path = Path("artifacts/model_leaderboard.csv")
    results_df_path: Path = Path("artifacts/results_predictions.csv")
    feature_importance_path: Path = Path("artifacts/feature_importance.csv")
    residual_hist_path: Path = Path("artifacts/residual_hist.png")
    residual_scatter_path: Path = Path("artifacts/residual_scatter.png")
    shap_values_path: Path = Path("artifacts/shap_values.npz")
    shap_summary_dot_path: Path = Path("artifacts/shap_summary_dot.png")
    shap_summary_bar_path: Path = Path("artifacts/shap_summary_bar.png")
    shap_dependence_path: Path = Path("artifacts/shap_dependence_top_feature.png")
    random_state: int = 42
    n_jobs: int = -1
    val_size: float = 0.2
    xgb_n_iter: int = 20
    xgb_cv: int = 3
    xgb_scoring: str = "r2"
    xgb_early_stopping_rounds: int = 100
    enable_shap: bool = True
    shap_max_background: int = 500
    shap_max_explain: int = 1000


# =============================================================================
# TRAINER
# =============================================================================
class ModelTrainer:
    """
    Orchestrates training, evaluation, artifacts.
    """
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
        Splits array into X (all but last col) and y (last col).
        """
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Array must be 2D with >=2 cols (X + y).")
        return arr[:, :-1], arr[:, -1]

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes regression metrics: R2, RMSE, MAE.
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"r2": r2, "rmse": rmse, "mae": mae}

    def _try_get_feature_names(self, preprocessor_path: str) -> Optional[np.ndarray]:
        """
        Gets feature names from preprocessor for labeling.
        """
        try:
            pre = joblib.load(preprocessor_path)
            if hasattr(pre, "get_feature_names_out"):
                return pre.get_feature_names_out()
            return None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Artifacts
    # -------------------------------------------------------------------------
    def _save_results_df(self, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
        """
        Saves predictions with diffs for analysis.
        """
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
        df["diff"] = df["actual"] - df["predicted"]
        denom = df["actual"].replace(0, np.nan)
        df["diff_pct"] = (df["diff"] / denom) * 100.0
        df.to_csv(path, index=False)
        logging.info("Saved results_df: %s", path)

    def _save_feature_importance(self, model: Any, feature_names: Optional[np.ndarray], out_path: Path) -> None:
        """
        Saves coef_ or feature_importances_ if available.
        Sorts by absolute importance.
        """
        try:
            if hasattr(model, "coef_"):
                imp = np.asarray(model.coef_).reshape(-1)
                kind = "coef_"
            elif hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_).reshape(-1)
                kind = "feature_importances_"
            else:
                logging.info("No coef_/importances_; skipping.")
                return

            if feature_names is not None and len(feature_names) == len(imp):
                df = pd.DataFrame({"feature": feature_names, "importance": imp})
            else:
                df = pd.DataFrame({"feature_index": np.arange(len(imp)), "importance": imp})

            df["abs_importance"] = df["importance"].abs()
            df = df.sort_values("abs_importance", ascending=False).drop(columns=["abs_importance"])
            df.to_csv(out_path, index=False)
            logging.info("Saved %s: %s", kind, out_path)
        except Exception as e:
            logging.warning("Feature importance save failed: %s", str(e))

    def _save_residual_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Saves enhanced residual plots.
        
        - Hist: With KDE for distribution if available.
        - Scatter: With reg line, zero reference.
        Uses sns if available for professionalism; fallback to plt.
        Gracefully handles if libs or sub-libs (scipy) missing.
        """
        if plt is None:
            logging.warning("matplotlib not available; skipping residual plots.")
            return

        residuals = y_true - y_pred

        # Set theme if sns available
        if sns is not None:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

        try:
            # Hist plot
            fig, ax = plt.subplots(figsize=(10, 6))
            if sns is not None:
                sns.histplot(residuals, kde=True, bins=40, ax=ax)
            else:
                ax.hist(residuals, bins=40, density=True)
                if gaussian_kde is not None:
                    try:
                        kde = gaussian_kde(residuals)
                        x = np.linspace(residuals.min(), residuals.max(), 100)
                        ax.plot(x, kde(x), 'r--')
                    except Exception:
                        pass  # Skip KDE if fails
            ax.set_title("Residual Distribution (Actual - Predicted)", fontsize=14)
            ax.set_xlabel("Residual Value", fontsize=12)
            ax.set_ylabel("Density" if sns is None else "Count", fontsize=12)
            plt.tight_layout()
            plt.savefig(self.config.residual_hist_path, dpi=300)
            plt.close()

            # Scatter plot with reg line
            fig, ax = plt.subplots(figsize=(10, 6))
            if sns is not None:
                sns.regplot(x=y_pred, y=residuals, scatter_kws={"alpha": 0.35}, line_kws={"color": "red"}, ax=ax)
            else:
                ax.scatter(y_pred, residuals, alpha=0.35)
                if linregress is not None:
                    try:
                        valid_mask = ~np.isnan(y_pred) & ~np.isnan(residuals)
                        slope, intercept, _, _, _ = linregress(y_pred[valid_mask], residuals[valid_mask])
                        line = slope * y_pred + intercept
                        ax.plot(y_pred, line, 'r--')
                    except Exception:
                        pass  # Skip reg line if fails
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
            ax.set_title("Residuals vs Predicted Values", fontsize=14)
            ax.set_xlabel("Predicted Value", fontsize=12)
            ax.set_ylabel("Residual Value", fontsize=12)
            plt.tight_layout()
            plt.savefig(self.config.residual_scatter_path, dpi=300)
            plt.close()

            logging.info("Saved residuals: %s | %s",
                         self.config.residual_hist_path, self.config.residual_scatter_path)
        except Exception as e:
            logging.warning("Residual plots generation failed: %s", str(e))

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------
    def _train_xgboost_with_random_search(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Optional[Any]:
        """
        Tunes XGBoost with RandomizedSearchCV; refines with early stopping.
        """
        try:
            from xgboost import XGBRegressor, callback
        except ImportError:
            logging.warning("xgboost skipped.")
            return None

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
            base,
            param_distributions=param_dist,
            n_iter=self.config.xgb_n_iter,
            scoring=self.config.xgb_scoring,
            cv=self.config.xgb_cv,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=0,
        )

        logging.info("RandomizedSearchCV for XGBoost (n_iter=%s, cv=%s).",
                     self.config.xgb_n_iter, self.config.xgb_cv)
        search.fit(X_train_full, y_train_full)

        best_model = search.best_estimator_
        logging.info("Best params: %s", search.best_params_)
        logging.info("Best CV score: %.6f", float(search.best_score_))

        # Early stopping
        logging.info("Early-stopping refinement.")
        try:
            cb = [callback.EarlyStopping(rounds=self.config.xgb_early_stopping_rounds, save_best=True)]
            best_model.fit(
                X_train_full, y_train_full,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=cb,
            )
        except Exception:
            try:
                best_model.fit(
                    X_train_full, y_train_full,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=self.config.xgb_early_stopping_rounds,
                )
            except Exception:
                logging.warning("Early stopping skipped; using CV-best.")

        return best_model

    # -------------------------------------------------------------------------
    # SHAP
    # -------------------------------------------------------------------------
    def _run_shap_analysis(
        self,
        model: Any,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        feature_names: Optional[np.ndarray],
    ) -> None:
        """
        Robust SHAP: Tries TreeExplainer, falls back to general Explainer.
        Subsamples for efficiency. Saves values/plots.
        Never crashes pipeline.
        """
        if not self.config.enable_shap:
            logging.info("SHAP disabled.")
            return

        cls_name = model.__class__.__name__.lower()
        if "xgb" not in cls_name:
            logging.info("SHAP skipped: not XGBoost (%s).", model.__class__.__name__)
            return

        try:
            import shap
            # No need for plt here, but if needed, check plt is not None
        except ImportError:
            logging.warning("shap not available -> skipping SHAP.")
            return

        # Set theme if available
        if sns is not None:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

        rng = np.random.default_rng(self.config.random_state)

        # Subsample
        X_bg = X_background
        if X_bg.shape[0] > self.config.shap_max_background:
            idx = rng.choice(X_bg.shape[0], self.config.shap_max_background, replace=False)
            X_bg = X_bg[idx]

        X_ex = X_explain
        if X_ex.shape[0] > self.config.shap_max_explain:
            idx = rng.choice(X_ex.shape[0], self.config.shap_max_explain, replace=False)
            X_ex = X_ex[idx]

        feature_names = feature_names if feature_names is not None else [f"f{i}" for i in range(X_ex.shape[1])]

        # Try TreeExplainer
        try:
            logging.info("TreeExplainer: bg=%s, ex=%s", X_bg.shape, X_ex.shape)
            explainer = shap.TreeExplainer(model, data=X_bg)
            shap_values = explainer.shap_values(X_ex)
            expected_value = getattr(explainer, "expected_value", None)

            np.savez_compressed(
                self.config.shap_values_path,
                shap_values=shap_values,
                X_explain=X_ex,
                expected_value=np.array(expected_value) if expected_value is not None else None,
            )
            logging.info("Saved SHAP: %s", self.config.shap_values_path)

            if plt is not None:
                # Dot plot
                fig = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
                plt.title("SHAP Summary (Dot Plot)", fontsize=14)
                plt.tight_layout()
                plt.savefig(self.config.shap_summary_dot_path, dpi=300)
                plt.close()

                # Bar plot
                fig = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_ex, feature_names=feature_names, plot_type="bar", show=False)
                plt.title("SHAP Summary (Bar Plot)", fontsize=14)
                plt.tight_layout()
                plt.savefig(self.config.shap_summary_bar_path, dpi=300)
                plt.close()

                # Dependence
                try:
                    mean_abs = np.mean(np.abs(shap_values), axis=0)
                    top_idx = int(np.argmax(mean_abs))
                    top_feature = feature_names[top_idx]
                    fig = plt.figure(figsize=(10, 6))
                    shap.dependence_plot(top_idx, shap_values, X_ex, feature_names=feature_names, show=False)
                    plt.title(f"SHAP Dependence: {top_feature}", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(self.config.shap_dependence_path, dpi=300)
                    plt.close()
                    logging.info("Saved dependence (%s): %s", top_feature, self.config.shap_dependence_path)
                except Exception:
                    logging.info("Dependence skipped.")

            return

        except Exception as e:
            logging.warning("TreeExplainer failed; fallback. %s", str(e))

        # Fallback Explainer
        try:
            logging.info("Fallback Explainer.")
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
            logging.info("Saved SHAP (fallback): %s", self.config.shap_values_path)

            if plt is not None:
                # Dot plot
                fig = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
                plt.title("SHAP Summary (Dot Plot - Fallback)", fontsize=14)
                plt.tight_layout()
                plt.savefig(self.config.shap_summary_dot_path, dpi=300)
                plt.close()

                # Bar plot
                fig = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_ex, feature_names=feature_names, plot_type="bar", show=False)
                plt.title("SHAP Summary (Bar Plot - Fallback)", fontsize=14)
                plt.tight_layout()
                plt.savefig(self.config.shap_summary_bar_path, dpi=300)
                plt.close()

        except Exception as e:
            logging.warning("SHAP fallback failed; skipping. %s", str(e))

    # -------------------------------------------------------------------------
    # Main
    # -------------------------------------------------------------------------
    def initiate_model_trainer(
        self,
        train_arr: np.ndarray,
        test_arr: np.ndarray,
        preprocessor_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Trains models, selects best by R2, saves artifacts.
        
        Returns: (best_name, best_metrics)
        """
        logging.info("Starting training.")
        try:
            self._ensure_artifacts_dir()

            X_train_full, y_train_full = self._split_xy(train_arr)
            X_test, y_test = self._split_xy(test_arr)

            logging.info("Train: X=%s y=%s", X_train_full.shape, y_train_full.shape)
            logging.info("Test: X=%s y=%s", X_test.shape, y_test.shape)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=self.config.val_size, random_state=self.config.random_state
            )
            logging.info("Internal: Train=%s Val=%s", X_train.shape, X_val.shape)

            feature_names = self._try_get_feature_names(preprocessor_path) if preprocessor_path else None

            leaderboard_rows: List[Dict[str, Any]] = []
            best_name: Optional[str] = None
            best_model: Optional[Any] = None
            best_metrics: Dict[str, float] = {"r2": -1e18, "rmse": float("inf"), "mae": float("inf")}

            # Baselines
            logging.info("Baseline: LinearRegression")
            lr = LinearRegression()
            lr.fit(X_train_full, y_train_full)
            preds = lr.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "LinearRegression", **metrics})
            best_name, best_model, best_metrics = "LinearRegression", lr, metrics

            logging.info("Baseline: Ridge")
            ridge = Ridge(random_state=self.config.random_state)
            ridge.fit(X_train_full, y_train_full)
            preds = ridge.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "Ridge", **metrics})
            if metrics["r2"] > best_metrics["r2"]:
                best_name, best_model, best_metrics = "Ridge", ridge, metrics

            logging.info("Baseline: Lasso")
            lasso = Lasso(random_state=self.config.random_state, max_iter=50000)
            lasso.fit(X_train_full, y_train_full)
            preds = lasso.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "Lasso", **metrics})
            if metrics["r2"] > best_metrics["r2"]:
                best_name, best_model, best_metrics = "Lasso", lasso, metrics

            # XGBoost
            xgb_model = self._train_xgboost_with_random_search(X_train_full, y_train_full, X_val, y_val)
            if xgb_model:
                preds = xgb_model.predict(X_test)
                metrics = self._evaluate(y_test, preds)
                leaderboard_rows.append({"model": "XGBoost(RandomSearch)", **metrics})
                if metrics["r2"] > best_metrics["r2"]:
                    best_name, best_model, best_metrics = "XGBoost(RandomSearch)", xgb_model, metrics

            if best_model is None or best_name is None:
                raise RuntimeError("No model trained.")

            # Leaderboard
            df_lb = pd.DataFrame(leaderboard_rows).sort_values("r2", ascending=False)
            df_lb.to_csv(self.config.leaderboard_path, index=False)
            logging.info("Saved leaderboard: %s", self.config.leaderboard_path)

            # Winner
            joblib.dump(best_model, self.config.sklearn_model_path)
            logging.info("Saved model: %s", self.config.sklearn_model_path)

            best_preds = best_model.predict(X_test)
            self._save_results_df(y_test, best_preds, self.config.results_df_path)
            self._save_residual_diagnostics(y_test, best_preds)
            self._save_feature_importance(best_model, feature_names, self.config.feature_importance_path)
            self._run_shap_analysis(best_model, X_train_full, X_test, feature_names)

            # Summary
            summary = (
                f"Best model: {best_name}\n"
                f"Saved to: {self.config.sklearn_model_path}\n"
                f"TEST R2: {best_metrics['r2']:.6f}\n"
                f"TEST RMSE: {best_metrics['rmse']:.6f}\n"
                f"TEST MAE: {best_metrics['mae']:.6f}\n\n"
                f"Artifacts:\n"
                f"- Leaderboard: {self.config.leaderboard_path}\n"
                f"- Results: {self.config.results_df_path}\n"
                f"- Importance: {self.config.feature_importance_path}\n"
                f"- Hist: {self.config.residual_hist_path}\n"
                f"- Scatter: {self.config.residual_scatter_path}\n"
                f"- SHAP: {self.config.shap_values_path}\n"
                f"- Dot: {self.config.shap_summary_dot_path}\n"
                f"- Bar: {self.config.shap_summary_bar_path}\n"
            )
            self._write_text(self.config.metrics_path, summary)
            self._write_text(self.config.winner_meta_path, f"{best_name} | {self.config.sklearn_model_path}\n")

            logging.info("Winner: %s | R2=%.4f", best_name, best_metrics["r2"])
            return best_name, best_metrics

        except Exception as e:
            logging.exception("Training failed.")
            raise CustomException(e, sys)


# =============================================================================
# SCRIPT
# =============================================================================
def _run_as_script() -> None:
    """
    End-to-end: Transformation + Training.
    """
    logging.info("Running as script.")
    try:
        tcfg = DataTransformationConfig()
        if not tcfg.train_data_path.exists() or not tcfg.test_data_path.exists():
            raise FileNotFoundError("Run ingestion.")

        transformer = DataTransformation(tcfg)
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            str(tcfg.train_data_path), str(tcfg.test_data_path)
        )
        logging.info("Preprocessor: %s", preprocessor_path)

        trainer = ModelTrainer(ModelTrainerConfig())
        best_name, best_metrics = trainer.initiate_model_trainer(
            train_arr, test_arr, preprocessor_path
        )

        print("✅ Complete")
        print("Best:", best_name)
        print("Metrics:", best_metrics)
        print("Artifacts:", str(trainer.config.artifacts_dir.resolve()))

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    _run_as_script()