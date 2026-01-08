"""
model_trainer.py
================

Purpose
-------
Production-grade trainer for regression (insurance premiums) with comprehensive evaluation.
Integrates with transformation pipeline and provides detailed model understanding through
extensive visualizations and metrics.

What this module does
---------------------
1. Runs transformation to get arrays/preprocessor.
2. Trains/evaluates models (Linear, Ridge, Lasso, XGBoost).
3. Performs comprehensive model evaluation with multiple metrics.
4. Generates extensive visualizations for model understanding:
   - Residual diagnostics (histogram, scatter)
   - Actual vs Predicted plots
   - Learning curves
   - Cross-validation score distributions
   - Model comparison charts
   - Feature importance analysis
   - SHAP interpretability plots
5. Saves all artifacts for reproducibility and analysis.

Enhanced Evaluation Features
----------------------------
- Extended metrics: R2, RMSE, MAE, MAPE, Explained Variance
- Actual vs Predicted scatter with perfect prediction line
- Learning curves to diagnose bias/variance
- CV score distributions for model stability assessment
- Side-by-side model performance comparison
- Prediction error distribution analysis
- Comprehensive model diagnostics

Visualization Improvements (Modern Principles)
----------------------------------------------
- Seaborn theme: Whitegrid, muted palette for clarity/colorblindness.
- Clarity: Titles, labels, legends, grids, alpha for scatters.
- Visibility: KDE in hists, reg lines in scatters, high DPI (300).
- Professionalism: Tight layouts, annotations, consistent styling.
- Organization: All plots saved to dedicated folders for easy access.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from src.exception import CustomException
from src.logger import logging

# Transformation integration
from src.components.data_transformation import DataTransformation, DataTransformationConfig

# Optional imports for visualizations and SHAP
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None
    gridspec = None

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
    Configuration for model training, evaluation, and artifacts generation.
    Frozen dataclass ensures configuration immutability for reproducibility.
    
    Attributes:
        artifacts_dir: Root directory for all outputs
        
        Model Artifacts:
            sklearn_model_path: Serialized winner model (joblib format)
            
        Metrics & Metadata:
            metrics_path: Comprehensive metrics summary (text file)
            winner_meta_path: Winner model identifier
            leaderboard_path: All models ranked by performance (CSV)
            extended_metrics_path: Detailed evaluation metrics (CSV)
            
        Results & Predictions:
            results_df_path: Predictions with actuals and residuals (CSV)
            
        Feature Analysis:
            feature_importance_path: Feature coefficients/importances (CSV)
            
        Diagnostic Visualizations:
            residual_hist_path: Residual distribution with KDE
            residual_scatter_path: Residuals vs predicted with regression line
            actual_vs_predicted_path: Scatter plot showing prediction accuracy
            error_distribution_path: Comprehensive error analysis
            
        Model Understanding:
            learning_curves_path: Training/validation scores vs dataset size
            cv_scores_path: Cross-validation score distributions
            model_comparison_path: Side-by-side model performance
            
        Interpretability (SHAP):
            shap_values_path: Computed SHAP values (compressed numpy)
            shap_summary_dot_path: Feature importance by impact
            shap_summary_bar_path: Mean absolute SHAP values
            shap_dependence_path: Top feature interaction plot
            
        Training Parameters:
            random_state: Random seed for reproducibility (42)
            n_jobs: CPU parallelism (-1 = all cores)
            val_size: Validation split ratio for early stopping (0.2 = 20%)
            
        XGBoost Hyperparameter Tuning:
            xgb_n_iter: RandomizedSearchCV iterations (20)
            xgb_cv: Cross-validation folds (3)
            xgb_scoring: Optimization metric ('r2')
            xgb_early_stopping_rounds: Patience for early stopping (100)
            
        SHAP Configuration:
            enable_shap: Toggle SHAP analysis (True)
            shap_max_background: Background samples for explainer (500)
            shap_max_explain: Samples to explain (1000)
            
        Evaluation Settings:
            enable_learning_curves: Generate learning curves (True)
            learning_curve_cv: CV folds for learning curves (5)
            learning_curve_train_sizes: Sample sizes to evaluate (10 points)
    """
    # Directory structure
    artifacts_dir: Path = Path("artifacts")
    
    # Core model artifacts
    sklearn_model_path: Path = Path("artifacts/model.pkl")
    
    # Metrics and metadata files
    metrics_path: Path = Path("artifacts/model_metrics.txt")
    winner_meta_path: Path = Path("artifacts/model_winner.txt")
    leaderboard_path: Path = Path("artifacts/model_leaderboard.csv")
    extended_metrics_path: Path = Path("artifacts/extended_metrics.csv")
    
    # Predictions and results
    results_df_path: Path = Path("artifacts/results_predictions.csv")
    
    # Feature analysis
    feature_importance_path: Path = Path("artifacts/feature_importance.csv")
    
    # Diagnostic plots
    residual_hist_path: Path = Path("artifacts/plots/residual_hist.png")
    residual_scatter_path: Path = Path("artifacts/plots/residual_scatter.png")
    actual_vs_predicted_path: Path = Path("artifacts/plots/actual_vs_predicted.png")
    error_distribution_path: Path = Path("artifacts/plots/error_distribution.png")
    
    # Model understanding plots
    learning_curves_path: Path = Path("artifacts/plots/learning_curves.png")
    cv_scores_path: Path = Path("artifacts/plots/cv_scores_distribution.png")
    model_comparison_path: Path = Path("artifacts/plots/model_comparison.png")
    
    # SHAP interpretability
    shap_values_path: Path = Path("artifacts/shap/shap_values.npz")
    shap_summary_dot_path: Path = Path("artifacts/shap/shap_summary_dot.png")
    shap_summary_bar_path: Path = Path("artifacts/shap/shap_summary_bar.png")
    shap_dependence_path: Path = Path("artifacts/shap/shap_dependence_top_feature.png")
    
    # Training parameters
    random_state: int = 42
    n_jobs: int = -1
    val_size: float = 0.2
    
    # XGBoost tuning parameters
    xgb_n_iter: int = 20
    xgb_cv: int = 3
    xgb_scoring: str = "r2"
    xgb_early_stopping_rounds: int = 100
    
    # SHAP parameters
    enable_shap: bool = True
    shap_max_background: int = 500
    shap_max_explain: int = 1000
    
    # Evaluation parameters
    enable_learning_curves: bool = True
    learning_curve_cv: int = 5
    learning_curve_train_sizes: int = 10


# =============================================================================
# TRAINER
# =============================================================================
class ModelTrainer:
    """
    Orchestrates comprehensive model training, evaluation, and artifact generation.
    
    This class handles the complete ML pipeline from model training through evaluation
    and visualization. It trains multiple baseline models and XGBoost, selects the best
    performer, and generates extensive diagnostic outputs for model understanding.
    
    Key Responsibilities:
        1. Train multiple regression models (Linear, Ridge, Lasso, XGBoost)
        2. Evaluate models using comprehensive metrics
        3. Generate diagnostic visualizations
        4. Perform interpretability analysis (SHAP)
        5. Save all artifacts for reproducibility
    """
    
    def __init__(self, config: ModelTrainerConfig | None = None):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration. If None, uses default ModelTrainerConfig.
        """
        self.config = config or ModelTrainerConfig()
        # Store all trained models for comparison
        self.trained_models: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    def _ensure_artifacts_dir(self) -> None:
        """
        Create artifacts directory structure if it doesn't exist.
        Creates: artifacts/, artifacts/plots/, artifacts/shap/
        """
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.config.artifacts_dir / "plots").mkdir(exist_ok=True)
        (self.config.artifacts_dir / "shap").mkdir(exist_ok=True)

    def _write_text(self, path: Path, content: str) -> None:
        """
        Write text content to file with UTF-8 encoding.
        
        Args:
            path: Output file path
            content: Text content to write
        """
        path.write_text(content, encoding="utf-8")

    def _split_xy(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split array into features (X) and target (y).
        Assumes target is the last column.
        
        Args:
            arr: 2D numpy array with features and target
            
        Returns:
            Tuple of (X, y) where X is all columns except last, y is last column
            
        Raises:
            ValueError: If array is not 2D or has fewer than 2 columns
        """
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Array must be 2D with >=2 cols (X + y).")
        return arr[:, :-1], arr[:, -1]

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive regression metrics for model evaluation.
        
        Metrics computed:
            - R2 Score: Coefficient of determination (1.0 = perfect, 0.0 = baseline)
            - RMSE: Root Mean Squared Error (lower is better, same units as target)
            - MAE: Mean Absolute Error (lower is better, robust to outliers)
            - MAPE: Mean Absolute Percentage Error (% error, scale-independent)
            - Explained Variance: Proportion of variance explained by model
            
        Args:
            y_true: Actual target values
            y_pred: Model predictions
            
        Returns:
            Dictionary with all computed metrics
        """
        # Core metrics
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        # Additional metrics for comprehensive evaluation
        # MAPE: Percentage error (useful for interpretability)
        # Avoid division by zero by replacing 0s with small value
        mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100)
        
        # Explained variance: Similar to R2 but doesn't penalize systematic offset
        explained_var = float(explained_variance_score(y_true, y_pred))
        
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "explained_variance": explained_var
        }

    def _try_get_feature_names(self, preprocessor_path: str) -> Optional[np.ndarray]:
        """
        Extract feature names from saved preprocessor for labeling plots.
        
        The preprocessor's get_feature_names_out() returns transformed feature names
        including one-hot encoded categorical variables.
        
        Args:
            preprocessor_path: Path to saved preprocessor pickle
            
        Returns:
            Array of feature names if available, None otherwise
        """
        try:
            pre = joblib.load(preprocessor_path)
            if hasattr(pre, "get_feature_names_out"):
                return pre.get_feature_names_out()
            return None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Artifact Saving Methods
    # -------------------------------------------------------------------------
    def _save_results_df(self, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
        """
        Save detailed predictions DataFrame with residuals and percentage errors.
        
        Creates a CSV with:
            - actual: True target values
            - predicted: Model predictions
            - diff: Residual (actual - predicted)
            - diff_pct: Percentage error (useful for insurance premiums)
            
        This file enables detailed error analysis and model debugging.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            path: Output CSV path
        """
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
        df["diff"] = df["actual"] - df["predicted"]
        # Percentage difference (replace 0s to avoid division by zero)
        denom = df["actual"].replace(0, np.nan)
        df["diff_pct"] = (df["diff"] / denom) * 100.0
        df.to_csv(path, index=False)
        logging.info("Saved results_df: %s", path)

    def _save_feature_importance(self, model: Any, feature_names: Optional[np.ndarray], out_path: Path) -> None:
        """
        Extract and save feature importance/coefficients from trained model.
        
        For linear models: Uses coef_ (regression coefficients)
        For tree models: Uses feature_importances_ (Gini importance)
        
        Output is sorted by absolute importance to highlight most impactful features.
        
        Args:
            model: Trained sklearn/xgboost model
            feature_names: Names of features for labeling
            out_path: Output CSV path
        """
        try:
            # Extract importance based on model type
            if hasattr(model, "coef_"):
                imp = np.asarray(model.coef_).reshape(-1)
                kind = "coef_"
            elif hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_).reshape(-1)
                kind = "feature_importances_"
            else:
                logging.info("No coef_/importances_; skipping.")
                return

            # Create DataFrame with feature names or indices
            if feature_names is not None and len(feature_names) == len(imp):
                df = pd.DataFrame({"feature": feature_names, "importance": imp})
            else:
                df = pd.DataFrame({"feature_index": np.arange(len(imp)), "importance": imp})

            # Sort by absolute importance (magnitude matters more than direction)
            df["abs_importance"] = df["importance"].abs()
            df = df.sort_values("abs_importance", ascending=False).drop(columns=["abs_importance"])
            df.to_csv(out_path, index=False)
            logging.info("Saved %s: %s", kind, out_path)
        except Exception as e:
            logging.warning("Feature importance save failed: %s", str(e))

    # -------------------------------------------------------------------------
    # Enhanced Visualization Methods
    # -------------------------------------------------------------------------
    def _save_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Generate actual vs predicted scatter plot with perfect prediction line.
        
        This plot is crucial for understanding:
            - Overall prediction accuracy (points close to diagonal = good)
            - Systematic bias (points consistently above/below diagonal)
            - Heteroscedasticity (variance changes with prediction magnitude)
            - Outliers and prediction errors
            
        Features:
            - 45-degree reference line (perfect predictions)
            - Semi-transparent points to show density
            - Correlation coefficient annotation
            - Professional styling with seaborn
        """
        if plt is None:
            logging.warning("matplotlib not available; skipping actual vs predicted.")
            return

        if sns is not None:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot with transparency to show overlapping points
            ax.scatter(y_true, y_pred, alpha=0.4, edgecolors='k', linewidth=0.5, s=50)
            
            # Perfect prediction line (45-degree line)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate and display correlation coefficient
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel("Actual Values", fontsize=12)
            ax.set_ylabel("Predicted Values", fontsize=12)
            ax.set_title("Actual vs Predicted Values", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.config.actual_vs_predicted_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Saved actual vs predicted: %s", self.config.actual_vs_predicted_path)
        except Exception as e:
            logging.warning("Actual vs predicted plot failed: %s", str(e))

    def _save_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Generate comprehensive error distribution analysis with multiple subplots.
        
        This 2x2 grid shows:
            1. Residual histogram with KDE - shows if errors are normally distributed
            2. Residual scatter - shows if errors are homoscedastic (constant variance)
            3. Absolute error distribution - shows magnitude of errors regardless of direction
            4. Q-Q plot - checks if residuals follow normal distribution
            
        These plots help diagnose:
            - Non-normality of errors (Q-Q plot deviation)
            - Heteroscedasticity (residual scatter pattern)
            - Systematic bias (histogram centering)
            - Error magnitude patterns (absolute error distribution)
        """
        if plt is None or gridspec is None:
            logging.warning("matplotlib not available; skipping error distribution.")
            return

        if sns is not None:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

        try:
            residuals = y_true - y_pred
            abs_errors = np.abs(residuals)
            
            fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # 1. Residual histogram with KDE - shows error distribution shape
            ax1 = fig.add_subplot(gs[0, 0])
            if sns is not None:
                sns.histplot(residuals, kde=True, bins=40, ax=ax1)
            else:
                ax1.hist(residuals, bins=40, density=True, alpha=0.7)
            ax1.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax1.set_title("Residual Distribution", fontsize=12, fontweight='bold')
            ax1.set_xlabel("Residual (Actual - Predicted)")
            ax1.set_ylabel("Density")
            ax1.legend()
            
            # 2. Residuals vs Predicted - checks for heteroscedasticity
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(y_pred, residuals, alpha=0.4, edgecolors='k', linewidth=0.5)
            ax2.axhline(0, color='r', linestyle='--', linewidth=2)
            # Add smoothed trend line to detect patterns
            if linregress is not None:
                try:
                    slope, intercept, _, _, _ = linregress(y_pred, residuals)
                    line = slope * y_pred + intercept
                    ax2.plot(y_pred, line, 'b-', linewidth=2, alpha=0.7, label='Trend')
                    ax2.legend()
                except Exception:
                    pass
            ax2.set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Predicted Values")
            ax2.set_ylabel("Residuals")
            ax2.grid(True, alpha=0.3)
            
            # 3. Absolute error distribution - shows error magnitude patterns
            ax3 = fig.add_subplot(gs[1, 0])
            if sns is not None:
                sns.histplot(abs_errors, kde=True, bins=40, ax=ax3)
            else:
                ax3.hist(abs_errors, bins=40, density=True, alpha=0.7)
            mean_abs_error = np.mean(abs_errors)
            ax3.axvline(mean_abs_error, color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_abs_error:.2f}')
            ax3.set_title("Absolute Error Distribution", fontsize=12, fontweight='bold')
            ax3.set_xlabel("Absolute Error")
            ax3.set_ylabel("Density")
            ax3.legend()
            
            # 4. Q-Q plot for normality check
            ax4 = fig.add_subplot(gs[1, 1])
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle("Comprehensive Error Analysis", fontsize=14, fontweight='bold', y=0.995)
            plt.savefig(self.config.error_distribution_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Saved error distribution: %s", self.config.error_distribution_path)
        except Exception as e:
            logging.warning("Error distribution plot failed: %s", str(e))

    def _save_residual_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Generate enhanced residual diagnostic plots (histogram and scatter).
        
        Residual plots are fundamental for regression diagnostics:
            - Histogram: Should be approximately normal and centered at 0
            - Scatter: Should show random pattern with no trends (homoscedastic)
            
        Features:
            - KDE overlay for smooth distribution visualization
            - Regression line on scatter to detect systematic patterns
            - Zero reference line for easy interpretation
            - High-resolution output (300 DPI)
        """
        if plt is None:
            logging.warning("matplotlib not available; skipping residual plots.")
            return

        residuals = y_true - y_pred

        if sns is not None:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

        try:
            # Histogram with KDE
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
                        pass
            ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_title("Residual Distribution (Actual - Predicted)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Residual Value", fontsize=12)
            ax.set_ylabel("Density" if sns is None else "Count", fontsize=12)
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.config.residual_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Scatter plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6))
            if sns is not None:
                sns.regplot(x=y_pred, y=residuals, scatter_kws={"alpha": 0.35}, 
                           line_kws={"color": "red"}, ax=ax)
            else:
                ax.scatter(y_pred, residuals, alpha=0.35)
                if linregress is not None:
                    try:
                        valid_mask = ~np.isnan(y_pred) & ~np.isnan(residuals)
                        slope, intercept, _, _, _ = linregress(y_pred[valid_mask], residuals[valid_mask])
                        line = slope * y_pred + intercept
                        ax.plot(y_pred, line, 'r--')
                    except Exception:
                        pass
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
            ax.set_title("Residuals vs Predicted Values", fontsize=14, fontweight='bold')
            ax.set_xlabel("Predicted Value", fontsize=12)
            ax.set_ylabel("Residual Value", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.config.residual_scatter_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info("Saved residuals: %s | %s",
                         self.config.residual_hist_path, self.config.residual_scatter_path)
        except Exception as e:
            logging.warning("Residual plots generation failed: %s", str(e))

    def _save_learning_curves(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> None:
        """
        Generate learning curves to diagnose bias and variance.
        
        Learning curves show:
            - Training score vs dataset size (should be high, stable)
            - Validation score vs dataset size (should increase and converge)
            
        Interpretation:
            - High bias (underfitting): Both scores low and close together
            - High variance (overfitting): Large gap between train and validation
            - Good fit: Both scores high and converging
            - More data helps: Validation score still improving at max size
            
        This helps answer: "Would more data help?" and "Is model too simple/complex?"
        
        Args:
            model: Trained model to evaluate
            X: Training features
            y: Training target
            model_name: Name for plot title
        """
        if not self.config.enable_learning_curves or plt is None:
            return

        try:
            logging.info(f"Generating learning curves for {model_name}...")
            
            # Generate learning curve data at different training set sizes
            train_sizes = np.linspace(0.1, 1.0, self.config.learning_curve_train_sizes)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=self.config.learning_curve_cv,
                scoring='r2',
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state
            )
            
            # Calculate mean and std for train and validation scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            if sns is not None:
                sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot training scores with confidence interval
            ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color='blue')
            
            # Plot validation scores with confidence interval
            ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                           alpha=0.2, color='red')
            
            ax.set_xlabel('Training Set Size', fontsize=12)
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_title(f'Learning Curves: {model_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add interpretation text
            gap = train_mean[-1] - val_mean[-1]
            if gap > 0.1:
                interpretation = "High variance: Consider regularization or more data"
            elif val_mean[-1] < 0.7:
                interpretation = "High bias: Consider more features or complex model"
            else:
                interpretation = "Good fit: Model generalizes well"
            
            ax.text(0.02, 0.02, interpretation, transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(self.config.learning_curves_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Saved learning curves: %s", self.config.learning_curves_path)
        except Exception as e:
            logging.warning(f"Learning curves failed for {model_name}: {str(e)}")

    def _save_cv_scores_distribution(self, models_cv_scores: Dict[str, np.ndarray]) -> None:
        """
        Visualize cross-validation score distributions for all models.
        
        This box plot shows:
            - Median performance (line in box)
            - Variability (box height = IQR)
            - Outliers (points beyond whiskers)
            - Consistency across folds
            
        Models with:
            - Higher median = better average performance
            - Smaller boxes = more consistent/stable
            - No outliers = reliable across different data splits
            
        This helps assess model stability and reliability, not just point performance.
        
        Args:
            models_cv_scores: Dict mapping model names to array of CV scores
        """
        if plt is None or not models_cv_scores:
            return

        try:
            if sns is not None:
                sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data for box plot
            model_names = list(models_cv_scores.keys())
            scores_list = [models_cv_scores[name] for name in model_names]
            
            # Create box plot
            bp = ax.boxplot(scores_list, labels=model_names, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_title('Cross-Validation Score Distribution by Model', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.config.cv_scores_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Saved CV scores distribution: %s", self.config.cv_scores_path)
        except Exception as e:
            logging.warning(f"CV scores distribution plot failed: {str(e)}")

    def _save_model_comparison(self, leaderboard_df: pd.DataFrame) -> None:
        """
        Create side-by-side comparison of all models across multiple metrics.
        
        This grouped bar chart shows:
            - R² score (higher is better, max 1.0)
            - RMSE (lower is better, in target units)
            - MAE (lower is better, robust to outliers)
            
        Visualizing multiple metrics simultaneously helps because:
            - R² shows overall fit quality
            - RMSE penalizes large errors more
            - MAE is less sensitive to outliers
            
        A model that performs well across all metrics is more reliable.
        
        Args:
            leaderboard_df: DataFrame with model names and metrics
        """
        if plt is None:
            return

        try:
            if sns is not None:
                sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
            
            # Select metrics to compare
            metrics_to_plot = ['r2', 'rmse', 'mae']
            available_metrics = [m for m in metrics_to_plot if m in leaderboard_df.columns]
            
            if not available_metrics:
                return
            
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 5))
            if len(available_metrics) == 1:
                axes = [axes]
            
            # Plot each metric
            for idx, metric in enumerate(available_metrics):
                ax = axes[idx]
                
                # Sort by metric (R2 descending, errors ascending)
                if metric == 'r2':
                    sorted_df = leaderboard_df.sort_values(metric, ascending=False)
                else:
                    sorted_df = leaderboard_df.sort_values(metric, ascending=True)
                
                bars = ax.bar(range(len(sorted_df)), sorted_df[metric], 
                             color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_df))))
                
                # Annotate bars with values
                for i, (bar, value) in enumerate(zip(bars, sorted_df[metric])):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_xticks(range(len(sorted_df)))
                ax.set_xticklabels(sorted_df['model'], rotation=45, ha='right')
                ax.set_ylabel(metric.upper(), fontsize=11)
                ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(self.config.model_comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Saved model comparison: %s", self.config.model_comparison_path)
        except Exception as e:
            logging.warning(f"Model comparison plot failed: {str(e)}")

    # -------------------------------------------------------------------------
    # XGBoost Training
    # -------------------------------------------------------------------------
    def _train_xgboost_with_random_search(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Optional[Any]:
        """
        Train XGBoost with hyperparameter tuning via RandomizedSearchCV.
        
        Process:
            1. RandomizedSearchCV explores hyperparameter space efficiently
            2. Best parameters selected based on CV performance
            3. Final model refined with early stopping on validation set
            
        Hyperparameters tuned:
            - n_estimators: Number of boosting rounds
            - learning_rate: Step size shrinkage (lower = more conservative)
            - max_depth: Tree depth (higher = more complex, risk overfitting)
            - subsample: Row sampling ratio (< 1.0 adds randomness)
            - colsample_bytree: Column sampling ratio
            - reg_alpha/lambda: L1/L2 regularization
            - min_child_weight: Minimum samples in leaf (larger = more conservative)
            
        Early stopping prevents overfitting by monitoring validation performance.
        
        Args:
            X_train_full: Full training features
            y_train_full: Full training target
            X_val: Validation features for early stopping
            y_val: Validation target
            
        Returns:
            Trained XGBRegressor or None if xgboost unavailable
        """
        try:
            from xgboost import XGBRegressor, callback
        except ImportError:
            logging.warning("xgboost not installed; skipping XGBoost model.")
            return None

        # Initialize base estimator with sensible defaults
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        # Define hyperparameter search space
        # These ranges balance exploration with computational efficiency
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

        # RandomizedSearchCV samples random combinations
        # More efficient than GridSearchCV for large parameter spaces
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

        logging.info("Starting RandomizedSearchCV for XGBoost (n_iter=%s, cv=%s)",
                     self.config.xgb_n_iter, self.config.xgb_cv)
        search.fit(X_train_full, y_train_full)

        best_model = search.best_estimator_
        logging.info("Best XGBoost parameters found: %s", search.best_params_)
        logging.info("Best CV score (R²): %.6f", float(search.best_score_))

        # Refine model with early stopping to prevent overfitting
        # Monitors validation set and stops if no improvement
        logging.info("Refining model with early stopping (patience=%s rounds)",
                     self.config.xgb_early_stopping_rounds)
        try:
            # Try modern callback API first
            cb = [callback.EarlyStopping(rounds=self.config.xgb_early_stopping_rounds, save_best=True)]
            best_model.fit(
                X_train_full, y_train_full,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=cb,
            )
        except Exception:
            # Fallback to older API if callback fails
            try:
                best_model.fit(
                    X_train_full, y_train_full,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=self.config.xgb_early_stopping_rounds,
                )
            except Exception:
                logging.warning("Early stopping failed; using CV-best model.")

        return best_model

    # -------------------------------------------------------------------------
    # SHAP Analysis
    # -------------------------------------------------------------------------
    def _run_shap_analysis(
        self,
        model: Any,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        feature_names: Optional[np.ndarray],
    ) -> None:
        """
        Perform robust SHAP (SHapley Additive exPlanations) analysis for model interpretability.
        
        SHAP values explain individual predictions by showing:
            - How much each feature contributed to a specific prediction
            - Direction of impact (positive/negative)
            - Magnitude of effect
            
        Why SHAP?
            - Model-agnostic interpretability
            - Consistent and locally accurate
            - Based on solid game theory foundations
            - Shows feature interactions
            
        This implementation:
            1. Tries TreeExplainer first (fast, exact for tree models)
            2. Falls back to general Explainer if needed
            3. Subsamples data for efficiency
            4. Never crashes the pipeline (graceful degradation)
            
        Generates:
            - Dot plot: Shows all features and their impacts
            - Bar plot: Average absolute impact per feature
            - Dependence plot: Interaction effects for top feature
            
        Args:
            model: Trained model to explain
            X_background: Reference dataset for SHAP explainer
            X_explain: Samples to generate explanations for
            feature_names: Feature labels for plots
        """
        if not self.config.enable_shap:
            logging.info("SHAP analysis disabled in config.")
            return

        # SHAP works best with tree-based models (XGBoost, Random Forest, etc.)
        cls_name = model.__class__.__name__.lower()
        if "xgb" not in cls_name:
            logging.info("SHAP analysis skipped: model type not XGBoost (%s)", 
                        model.__class__.__name__)
            return

        try:
            import shap
        except ImportError:
            logging.warning("shap library not installed; skipping SHAP analysis.")
            return

        # Apply consistent styling
        if sns is not None:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

        rng = np.random.default_rng(self.config.random_state)

        # Subsample background data for efficiency
        # Background data provides reference distribution for SHAP
        X_bg = X_background
        if X_bg.shape[0] > self.config.shap_max_background:
            idx = rng.choice(X_bg.shape[0], self.config.shap_max_background, replace=False)
            X_bg = X_bg[idx]
            logging.info("Subsampled background to %d samples", len(X_bg))

        # Subsample explanation data
        X_ex = X_explain
        if X_ex.shape[0] > self.config.shap_max_explain:
            idx = rng.choice(X_ex.shape[0], self.config.shap_max_explain, replace=False)
            X_ex = X_ex[idx]
            logging.info("Subsampled explanation data to %d samples", len(X_ex))

        # Use provided feature names or generate defaults
        feature_names = feature_names if feature_names is not None else [f"f{i}" for i in range(X_ex.shape[1])]

        # Strategy 1: Try TreeExplainer (fast and exact for tree models)
        try:
            logging.info("Attempting TreeExplainer (bg=%s, explain=%s)", X_bg.shape, X_ex.shape)
            explainer = shap.TreeExplainer(model, data=X_bg)
            shap_values = explainer.shap_values(X_ex)
            expected_value = getattr(explainer, "expected_value", None)

            # Save SHAP values for future analysis
            np.savez_compressed(
                self.config.shap_values_path,
                shap_values=shap_values,
                X_explain=X_ex,
                expected_value=np.array(expected_value) if expected_value is not None else None,
            )
            logging.info("Saved SHAP values: %s", self.config.shap_values_path)

            if plt is not None:
                # Generate summary plots
                self._generate_shap_plots(shap_values, X_ex, feature_names)
            
            return

        except Exception as e:
            logging.warning("TreeExplainer failed, attempting fallback: %s", str(e))

        # Strategy 2: Fallback to general Explainer (slower but more robust)
        try:
            logging.info("Using fallback general Explainer")
            explainer = shap.Explainer(model.predict, X_bg)
            explanation = explainer(X_ex)
            shap_values = explanation.values
            base_values = getattr(explanation, "base_values", None)

            # Save SHAP values
            np.savez_compressed(
                self.config.shap_values_path,
                shap_values=shap_values,
                X_explain=X_ex,
                expected_value=np.array(base_values) if base_values is not None else None,
            )
            logging.info("Saved SHAP values (fallback): %s", self.config.shap_values_path)

            if plt is not None:
                self._generate_shap_plots(shap_values, X_ex, feature_names, fallback=True)

        except Exception as e:
            logging.warning("SHAP analysis completely failed; skipping. Error: %s", str(e))

    def _generate_shap_plots(
        self,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        feature_names: List[str],
        fallback: bool = False
    ) -> None:
        """
        Generate and save SHAP visualization plots.
        
        Creates three key plots:
            1. Dot plot: Shows feature impacts across all predictions
            2. Bar plot: Average absolute SHAP value per feature
            3. Dependence plot: Detailed view of top feature's effects
            
        Args:
            shap_values: Computed SHAP values
            X_explain: Samples being explained
            feature_names: Feature labels
            fallback: Whether using fallback explainer (for labeling)
        """
        import shap
        
        suffix = " (Fallback)" if fallback else ""
        
        try:
            # 1. Summary dot plot - most informative SHAP visualization
            # Shows distribution of impacts for each feature
            # Color represents feature value (red=high, blue=low)
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary (Dot Plot){suffix}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.config.shap_summary_dot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved SHAP dot plot: %s", self.config.shap_summary_dot_path)

            # 2. Summary bar plot - shows overall feature importance
            # Simpler view showing average magnitude of impact
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, 
                            plot_type="bar", show=False)
            plt.title(f"SHAP Summary (Bar Plot){suffix}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.config.shap_summary_bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved SHAP bar plot: %s", self.config.shap_summary_bar_path)

            # 3. Dependence plot for most important feature
            # Shows how feature value relates to SHAP value (impact)
            # Color shows interaction with another feature
            try:
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                top_idx = int(np.argmax(mean_abs))
                top_feature = feature_names[top_idx]
                
                fig = plt.figure(figsize=(10, 6))
                shap.dependence_plot(top_idx, shap_values, X_explain, 
                                   feature_names=feature_names, show=False)
                plt.title(f"SHAP Dependence: {top_feature}{suffix}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.config.shap_dependence_path, dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("Saved SHAP dependence plot for '%s': %s", 
                           top_feature, self.config.shap_dependence_path)
            except Exception as e:
                logging.info("SHAP dependence plot skipped: %s", str(e))

        except Exception as e:
            logging.warning("SHAP plot generation failed: %s", str(e))

    # -------------------------------------------------------------------------
    # Main Training Method
    # -------------------------------------------------------------------------
    def initiate_model_trainer(
        self,
        train_arr: np.ndarray,
        test_arr: np.ndarray,
        preprocessor_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Main training pipeline: trains models, evaluates, selects best, generates artifacts.
        
        Pipeline steps:
            1. Setup: Create directories, split data
            2. Train baseline models: LinearRegression, Ridge, Lasso
            3. Train XGBoost with hyperparameter tuning
            4. Evaluate all models on test set
            5. Select best model by R² score
            6. Generate comprehensive evaluation artifacts:
               - Prediction results
               - Residual diagnostics
               - Feature importance
               - Learning curves
               - CV score distributions
               - Model comparison charts
               - SHAP interpretability
            7. Save winner model and metadata
            
        Args:
            train_arr: Training data (features + target in last column)
            test_arr: Test data (features + target in last column)
            preprocessor_path: Path to saved preprocessor (for feature names)
            
        Returns:
            Tuple of (best_model_name, best_model_metrics_dict)
            
        Raises:
            CustomException: If training fails
        """
        logging.info("=" * 80)
        logging.info("STARTING MODEL TRAINING PIPELINE")
        logging.info("=" * 80)
        
        try:
            # Step 1: Setup
            self._ensure_artifacts_dir()

            # Split data into features and target
            X_train_full, y_train_full = self._split_xy(train_arr)
            X_test, y_test = self._split_xy(test_arr)

            logging.info("Dataset sizes:")
            logging.info("  Train: X=%s, y=%s", X_train_full.shape, y_train_full.shape)
            logging.info("  Test:  X=%s, y=%s", X_test.shape, y_test.shape)

            # Create internal validation split for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, 
                test_size=self.config.val_size, 
                random_state=self.config.random_state
            )
            logging.info("  Internal split: Train=%s, Val=%s", X_train.shape, X_val.shape)

            # Extract feature names for interpretability
            feature_names = self._try_get_feature_names(preprocessor_path) if preprocessor_path else None
            if feature_names is not None:
                logging.info("Loaded %d feature names from preprocessor", len(feature_names))

            # Storage for results
            leaderboard_rows: List[Dict[str, Any]] = []
            models_cv_scores: Dict[str, np.ndarray] = {}
            best_name: Optional[str] = None
            best_model: Optional[Any] = None
            best_metrics: Dict[str, float] = {"r2": -1e18, "rmse": float("inf"), "mae": float("inf")}

            logging.info("\n" + "=" * 80)
            logging.info("TRAINING BASELINE MODELS")
            logging.info("=" * 80)

            # Step 2: Train Linear Regression (simplest baseline)
            logging.info("\n[1/4] Training LinearRegression...")
            lr = LinearRegression()
            lr.fit(X_train_full, y_train_full)
            preds = lr.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "LinearRegression", **metrics})
            self.trained_models["LinearRegression"] = lr
            logging.info("  Results: R²=%.4f, RMSE=%.4f, MAE=%.4f", 
                        metrics['r2'], metrics['rmse'], metrics['mae'])
            
            # Track best model
            best_name, best_model, best_metrics = "LinearRegression", lr, metrics
            
            # Cross-validation for stability assessment
            cv_scores = cross_val_score(lr, X_train_full, y_train_full, cv=5, 
                                       scoring='r2', n_jobs=self.config.n_jobs)
            models_cv_scores["LinearRegression"] = cv_scores
            logging.info("  CV R² scores: mean=%.4f, std=%.4f", cv_scores.mean(), cv_scores.std())

            # Step 3: Train Ridge (L2 regularization)
            logging.info("\n[2/4] Training Ridge...")
            ridge = Ridge(random_state=self.config.random_state)
            ridge.fit(X_train_full, y_train_full)
            preds = ridge.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "Ridge", **metrics})
            self.trained_models["Ridge"] = ridge
            logging.info("  Results: R²=%.4f, RMSE=%.4f, MAE=%.4f", 
                        metrics['r2'], metrics['rmse'], metrics['mae'])
            
            if metrics["r2"] > best_metrics["r2"]:
                best_name, best_model, best_metrics = "Ridge", ridge, metrics
                logging.info("  ✓ New best model!")
            
            cv_scores = cross_val_score(ridge, X_train_full, y_train_full, cv=5,
                                       scoring='r2', n_jobs=self.config.n_jobs)
            models_cv_scores["Ridge"] = cv_scores
            logging.info("  CV R² scores: mean=%.4f, std=%.4f", cv_scores.mean(), cv_scores.std())

            # Step 4: Train Lasso (L1 regularization, feature selection)
            logging.info("\n[3/4] Training Lasso...")
            lasso = Lasso(random_state=self.config.random_state, max_iter=50000)
            lasso.fit(X_train_full, y_train_full)
            preds = lasso.predict(X_test)
            metrics = self._evaluate(y_test, preds)
            leaderboard_rows.append({"model": "Lasso", **metrics})
            self.trained_models["Lasso"] = lasso
            logging.info("  Results: R²=%.4f, RMSE=%.4f, MAE=%.4f", 
                        metrics['r2'], metrics['rmse'], metrics['mae'])
            
            if metrics["r2"] > best_metrics["r2"]:
                best_name, best_model, best_metrics = "Lasso", lasso, metrics
                logging.info("  ✓ New best model!")
            
            cv_scores = cross_val_score(lasso, X_train_full, y_train_full, cv=5,
                                       scoring='r2', n_jobs=self.config.n_jobs)
            models_cv_scores["Lasso"] = cv_scores
            logging.info("  CV R² scores: mean=%.4f, std=%.4f", cv_scores.mean(), cv_scores.std())

            # Step 5: Train XGBoost (gradient boosting with tuning)
            logging.info("\n[4/4] Training XGBoost (with hyperparameter tuning)...")
            xgb_model = self._train_xgboost_with_random_search(
                X_train_full, y_train_full, X_val, y_val
            )
            
            if xgb_model:
                preds = xgb_model.predict(X_test)
                metrics = self._evaluate(y_test, preds)
                leaderboard_rows.append({"model": "XGBoost(RandomSearch)", **metrics})
                self.trained_models["XGBoost(RandomSearch)"] = xgb_model
                logging.info("  Results: R²=%.4f, RMSE=%.4f, MAE=%.4f", 
                            metrics['r2'], metrics['rmse'], metrics['mae'])
                
                if metrics["r2"] > best_metrics["r2"]:
                    best_name, best_model, best_metrics = "XGBoost(RandomSearch)", xgb_model, metrics
                    logging.info("  ✓ New best model!")
                
                cv_scores = cross_val_score(xgb_model, X_train_full, y_train_full, cv=3,
                                           scoring='r2', n_jobs=self.config.n_jobs)
                models_cv_scores["XGBoost(RandomSearch)"] = cv_scores
                logging.info("  CV R² scores: mean=%.4f, std=%.4f", cv_scores.mean(), cv_scores.std())

            # Ensure we have a winner
            if best_model is None or best_name is None:
                raise RuntimeError("No model was successfully trained.")

            logging.info("\n" + "=" * 80)
            logging.info("MODEL SELECTION COMPLETE")
            logging.info("=" * 80)
            logging.info("Winner: %s", best_name)
            logging.info("Test R²: %.6f", best_metrics['r2'])
            logging.info("Test RMSE: %.6f", best_metrics['rmse'])
            logging.info("Test MAE: %.6f", best_metrics['mae'])

            # Step 6: Generate and save all artifacts
            logging.info("\n" + "=" * 80)
            logging.info("GENERATING EVALUATION ARTIFACTS")
            logging.info("=" * 80)

            # Save leaderboard
            df_lb = pd.DataFrame(leaderboard_rows).sort_values("r2", ascending=False)
            df_lb.to_csv(self.config.leaderboard_path, index=False)
            logging.info("✓ Saved leaderboard: %s", self.config.leaderboard_path)

            # Save extended metrics
            df_lb.to_csv(self.config.extended_metrics_path, index=False)
            logging.info("✓ Saved extended metrics: %s", self.config.extended_metrics_path)

            # Save winner model
            joblib.dump(best_model, self.config.sklearn_model_path)
            logging.info("✓ Saved model: %s", self.config.sklearn_model_path)

            # Generate predictions for winner model
            best_preds = best_model.predict(X_test)

            # Save detailed results
            self._save_results_df(y_test, best_preds, self.config.results_df_path)

            # Generate all visualizations
            logging.info("\nGenerating visualizations...")
            self._save_actual_vs_predicted(y_test, best_preds)
            self._save_residual_diagnostics(y_test, best_preds)
            self._save_error_distribution(y_test, best_preds)
            self._save_learning_curves(best_model, X_train_full, y_train_full, best_name)
            self._save_cv_scores_distribution(models_cv_scores)
            self._save_model_comparison(df_lb)

            # Save feature importance
            self._save_feature_importance(best_model, feature_names, self.config.feature_importance_path)

            # Run SHAP analysis for interpretability
            logging.info("\nRunning SHAP interpretability analysis...")
            self._run_shap_analysis(best_model, X_train_full, X_test, feature_names)

            # Step 7: Create summary report
            summary = self._create_summary_report(best_name, best_metrics, df_lb)
            self._write_text(self.config.metrics_path, summary)
            self._write_text(self.config.winner_meta_path, 
                           f"{best_name} | {self.config.sklearn_model_path}\n")

            logging.info("\n" + "=" * 80)
            logging.info("TRAINING PIPELINE COMPLETE")
            logging.info("=" * 80)
            logging.info("All artifacts saved to: %s", str(self.config.artifacts_dir.resolve()))

            return best_name, best_metrics

        except Exception as e:
            logging.exception("Model training pipeline failed")
            raise CustomException(e, sys)

    def _create_summary_report(
        self,
        best_name: str,
        best_metrics: Dict[str, float],
        leaderboard_df: pd.DataFrame
    ) -> str:
        """
        Create comprehensive text summary of training results.
        
        Args:
            best_name: Name of winning model
            best_metrics: Metrics dictionary for winner
            leaderboard_df: DataFrame with all model results
            
        Returns:
            Formatted summary string
        """
        summary = f"""
================================================================================
                         MODEL TRAINING SUMMARY
================================================================================

WINNER MODEL
------------
Model:              {best_name}
Saved to:           {self.config.sklearn_model_path}

TEST SET PERFORMANCE
-------------------
R² Score:           {best_metrics['r2']:.6f}
RMSE:               {best_metrics['rmse']:.6f}
MAE:                {best_metrics['mae']:.6f}
MAPE:               {best_metrics.get('mape', 0.0):.2f}%
Explained Variance: {best_metrics.get('explained_variance', 0.0):.6f}

ALL MODELS LEADERBOARD (sorted by R²)
-------------------------------------
{leaderboard_df.to_string(index=False)}

ARTIFACTS GENERATED
-------------------
Core Outputs:
  - Model:              {self.config.sklearn_model_path}
  - Leaderboard:        {self.config.leaderboard_path}
  - Extended Metrics:   {self.config.extended_metrics_path}
  - Predictions:        {self.config.results_df_path}
  - Feature Importance: {self.config.feature_importance_path}

Diagnostic Plots:
  - Residual Histogram:        {self.config.residual_hist_path}
  - Residual Scatter:          {self.config.residual_scatter_path}
  - Actual vs Predicted:       {self.config.actual_vs_predicted_path}
  - Error Distribution:        {self.config.error_distribution_path}

Model Understanding:
  - Learning Curves:           {self.config.learning_curves_path}
  - CV Score Distribution:     {self.config.cv_scores_path}
  - Model Comparison:          {self.config.model_comparison_path}

Interpretability (SHAP):
  - SHAP Values:               {self.config.shap_values_path}
  - SHAP Dot Plot:             {self.config.shap_summary_dot_path}
  - SHAP Bar Plot:             {self.config.shap_summary_bar_path}
  - SHAP Dependence:           {self.config.shap_dependence_path}

================================================================================
Training completed successfully. Review plots and metrics for model insights.
================================================================================
"""
        return summary


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
def _run_as_script() -> None:
    """
    End-to-end execution: data transformation + model training.
    
    This function demonstrates the complete pipeline:
        1. Load ingested and split data
        2. Apply transformations (scaling, encoding)
        3. Train and evaluate models
        4. Generate comprehensive diagnostics
        5. Save all artifacts
    """
    logging.info("=" * 80)
    logging.info("RUNNING MODEL TRAINING SCRIPT")
    logging.info("=" * 80)
    
    try:
        # Step 1: Check for transformed data
        tcfg = DataTransformationConfig()
        if not tcfg.train_data_path.exists() or not tcfg.test_data_path.exists():
            raise FileNotFoundError(
                "Transformed data not found. Please run data ingestion first."
            )

        # Step 2: Apply transformations
        logging.info("\nApplying data transformations...")
        transformer = DataTransformation(tcfg)
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            str(tcfg.train_data_path), str(tcfg.test_data_path)
        )
        logging.info("Preprocessor saved to: %s", preprocessor_path)

        # Step 3: Train models
        logging.info("\nInitiating model training...")
        trainer = ModelTrainer(ModelTrainerConfig())
        best_name, best_metrics = trainer.initiate_model_trainer(
            train_arr, test_arr, preprocessor_path
        )

        # Step 4: Report results
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best Model: {best_name}")
        print(f"Test R²:    {best_metrics['r2']:.6f}")
        print(f"Test RMSE:  {best_metrics['rmse']:.6f}")
        print(f"Test MAE:   {best_metrics['mae']:.6f}")
        print(f"\nArtifacts location: {trainer.config.artifacts_dir.resolve()}")
        print("=" * 80)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    _run_as_script()