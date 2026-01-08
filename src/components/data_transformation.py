"""
data_transformation.py
======================

Purpose
-------
A comprehensive data transformation pipeline that combines:
1. Data cleaning (standardize names, handle missing/duplicates, optional outliers).
2. **Enhanced EDA** with comprehensive visualizations and statistical analysis.
3. Feature engineering (schema-aware, leakage-safe derived features).
4. Optional multicollinearity checks (VIF on numerics only, with controlled dropping).
5. End-to-end preprocessing pipeline (impute, scale, one-hot).
6. Fit on train, transform train/test, append target.
7. Save artifacts (preprocessor, transformed arrays, comprehensive EDA files, VIF report).

Enhanced EDA Features
---------------------
- Distribution analysis: Histograms with KDE, Q-Q plots, statistical tests
- Categorical analysis: Count plots with percentages
- Target analysis: Distribution, relationships with features
- Bivariate analysis: Feature vs target visualizations
- Outlier detection: Multiple visualization techniques
- Correlation analysis: Enhanced heatmaps with clustering
- Statistical summaries: Skewness, kurtosis, normality tests
- Comprehensive reports: HTML-style summary documents

Design Goals (Advanced / Production)
------------------------------------
- Robust to schema drift: Infers columns if not specified, validates presence.
- Reproducible: Fits on train only; saves preprocessor for inference.
- Leakage prevention: Explicit checks in FE; no target-derived features.
- Logging & errors: CustomException + detailed logging for debugging.
- Config-driven: Toggles for cleaning/EDA/FE/VIF; thresholds; expected schema.
- Trainer-friendly: Outputs .npy arrays with X + y concatenated.
- Advanced: Added IQR outlier option; comprehensive EDA suite.

Visualization Improvements (Modern Principles)
----------------------------------------------
- Seaborn theme: Whitegrid, muted palette for clarity/colorblindness.
- Clarity: Descriptive titles/labels, rotated ticks, annotations, tight layouts.
- Visibility: Larger figures, KDE in histograms, colormaps in heatmaps.
- Professionalism: High DPI (300), consistent styling across all plots.
- Organization: Structured folders (distributions/, relationships/, outliers/).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional VIF
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    variance_inflation_factor = None

# Optional statistical tests
try:
    from scipy import stats
    from scipy.stats import shapiro, normaltest, skew, kurtosis
except ImportError:
    stats = None
    shapiro = None
    normaltest = None
    skew = None
    kurtosis = None

# Optional plotting (EDA)
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

from src.exception import CustomException
from src.logger import logging


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Comprehensive configuration for transformation pipeline with enhanced EDA.
    Frozen dataclass ensures immutability for reproducibility.
    
    Core Artifacts:
        artifacts_dir: Root directory for all outputs
        train_data_path/test_data_path: Input CSV files from ingestion
        preprocessor_obj_file_path: Saved sklearn pipeline
        transformed_train/test_file_path: Numpy arrays with X + y
        
    EDA Organization:
        eda_artifacts_dir: Root EDA directory
        eda_distributions_dir: Distribution plots (histograms, Q-Q)
        eda_relationships_dir: Feature vs target plots
        eda_outliers_dir: Outlier detection visualizations
        eda_correlations_dir: Correlation matrices and heatmaps
        eda_summary_report_path: Comprehensive HTML-style text report
        
    Feature Management:
        feature_manifest_path: JSON of selected features (for auditing)
        vif_report_path: CSV of VIF scores (multicollinearity)
        
    Data Schema:
        target_column: Name of target variable
        numerical_columns: Override numeric feature detection
        categorical_columns: Override categorical feature detection
        require_target_in_test: Whether test set must have target
        
    Cleaning Parameters:
        enable_cleaning: Toggle all cleaning steps
        standardize_column_names: Normalize column names (lower, underscores)
        missing_value_strategy: "drop" (aggressive) or "keep" (for imputers)
        drop_duplicates: Remove duplicate rows
        outlier_strategy: None | "clip" (winsorize) | "filter" (drop) | "iqr"
        outlier_lower_q/upper_q: Quantiles for clip/filter
        outlier_multiplier: IQR multiplier (1.5 = mild, 3.0 = extreme)
        outlier_numeric_cols: Specific columns; None infers all numeric
        
    EDA Parameters:
        enable_eda: Master toggle for all EDA
        eda_save_distributions: Histogram + KDE for numeric features
        eda_save_qq_plots: Normality assessment plots
        eda_save_categorical_analysis: Count plots for categorical features
        eda_save_target_analysis: Target distribution and relationships
        eda_save_bivariate: Feature vs target scatter/box plots
        eda_save_boxplots: Traditional boxplots for outlier detection
        eda_save_correlation: Correlation matrix CSV
        eda_save_heatmap: Visual correlation heatmap
        eda_save_statistical_summary: Extended stats (skew, kurtosis, tests)
        eda_max_plots_per_type: Limit plots to prevent overload
        eda_categorical_top_n: Show top N categories in plots
        
    Feature Engineering:
        enable_feature_engineering: Toggle FE
        disallow_target_leakage: Prevent target use in features
        
    VIF (Multicollinearity):
        enable_vif: Toggle VIF computation
        vif_threshold: Drop features with VIF > threshold
        vif_max_features_to_drop: Limit drops for safety
        vif_exempt_features: Protected features
        
    Pipeline:
        use_only_categorical: Baseline mode (ignore numerics)
        force_dense_output: Convert sparse to dense arrays
    """
    # Core paths
    artifacts_dir: Path = Path("artifacts")
    train_data_path: Path = Path("artifacts/train.csv")
    test_data_path: Path = Path("artifacts/test.csv")
    preprocessor_obj_file_path: Path = Path("artifacts/preprocessor.pkl")
    transformed_train_file_path: Path = Path("artifacts/train_transformed.npy")
    transformed_test_file_path: Path = Path("artifacts/test_transformed.npy")
    
    # Enhanced EDA structure
    eda_artifacts_dir: Path = Path("artifacts/eda")
    eda_distributions_dir: Path = Path("artifacts/eda/distributions")
    eda_relationships_dir: Path = Path("artifacts/eda/relationships")
    eda_outliers_dir: Path = Path("artifacts/eda/outliers")
    eda_correlations_dir: Path = Path("artifacts/eda/correlations")
    eda_summary_report_path: Path = Path("artifacts/eda/transformation_summary.txt")
    
    # Feature artifacts
    feature_manifest_path: Path = Path("artifacts/feature_manifest.json")
    vif_report_path: Path = Path("artifacts/vif_report.csv")
    
    # Schema
    target_column: str = "annual_premium_amount"
    require_target_in_test: bool = True
    categorical_columns: Optional[Tuple[str, ...]] = (
        "gender", "region", "marital_status", "physical_activity",
        "stress_level", "bmi_category", "smoking_status",
        "employment_status", "medical_history", "insurance_plan",
    )
    numerical_columns: Optional[Tuple[str, ...]] = (
        "age", "number_of_dependants", "income_lakhs",
    )
    
    # Cleaning
    enable_cleaning: bool = True
    standardize_column_names: bool = True
    missing_value_strategy: str = "keep"
    drop_duplicates: bool = True
    outlier_strategy: Optional[str] = None
    outlier_lower_q: float = 0.01
    outlier_upper_q: float = 0.99
    outlier_multiplier: float = 1.5
    outlier_numeric_cols: Optional[Tuple[str, ...]] = None
    
    # Enhanced EDA toggles
    enable_eda: bool = True
    eda_save_distributions: bool = True
    eda_save_qq_plots: bool = True
    eda_save_categorical_analysis: bool = True
    eda_save_target_analysis: bool = True
    eda_save_bivariate: bool = True
    eda_save_boxplots: bool = True
    eda_save_correlation: bool = True
    eda_save_heatmap: bool = True
    eda_save_statistical_summary: bool = True
    eda_max_plots_per_type: int = 20
    eda_categorical_top_n: int = 15
    
    # Feature engineering
    enable_feature_engineering: bool = True
    disallow_target_leakage: bool = True
    
    # VIF
    enable_vif: bool = True
    vif_threshold: float = 10.0
    vif_max_features_to_drop: int = 5
    vif_exempt_features: Tuple[str, ...] = ()
    
    # Pipeline
    use_only_categorical: bool = False
    force_dense_output: bool = True


# ---------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------
class DataCleaning:
    """
    Modular data cleaning utilities for preprocessing.
    
    Philosophy: Minimal intervention to preserve information.
    Heavy operations (outliers, dropping) are optional and configurable.
    """
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for consistency across datasets.
        
        Operations:
            - Strip whitespace
            - Replace spaces and hyphens with underscores
            - Convert to lowercase
            
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.lower()
        )
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        Handle missing values based on strategy.
        
        Strategies:
            - "drop": Remove rows with any NA (risks data loss, use cautiously)
            - "keep": Retain NA for downstream imputers (preferred for ML pipelines)
            
        Note: "keep" strategy allows sklearn imputers to handle missingness
        with appropriate strategies (median for numeric, mode for categorical).
        
        Args:
            df: Input DataFrame
            strategy: "drop" or "keep"
            
        Returns:
            DataFrame after missing value handling
        """
        if strategy not in ("drop", "keep"):
            raise ValueError("missing_value_strategy must be 'drop' or 'keep'")

        df = df.copy()
        if strategy == "drop":
            before = len(df)
            df = df.dropna()
            dropped = before - len(df)
            if dropped > 0:
                logging.info("Missing-value strategy=drop: dropped %s rows (%.2f%%).",
                           dropped, 100 * dropped / before)
        else:
            na_count = df.isna().sum().sum()
            logging.info("Missing-value strategy=keep: %s NA values will be handled by imputers.",
                        na_count)
        return df

    @staticmethod
    def handle_duplicates(df: pd.DataFrame, drop_duplicates: bool) -> pd.DataFrame:
        """
        Optionally remove duplicate rows to prevent data leakage and overfitting.
        
        Duplicates can:
            - Artificially inflate performance metrics
            - Cause overfitting if duplicates split across train/test
            - Indicate data quality issues
            
        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicates
            
        Returns:
            DataFrame after duplicate handling
        """
        df = df.copy()
        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            dropped = before - len(df)
            if dropped > 0:
                logging.info("Dropped %s duplicate rows (%.2f%%).", dropped, 100 * dropped / before)
        return df

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        strategy: Optional[str],
        lower_q: float,
        upper_q: float,
        multiplier: float,
        numeric_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Handle outliers in numeric columns using various strategies.
        
        Strategies:
            None: Skip outlier handling
            "clip": Winsorize to quantiles (preserves all data, reduces extremes)
            "filter": Drop rows outside quantiles (data loss, but removes true outliers)
            "iqr": Tukey's method using Q1 - multiplier*IQR, Q3 + multiplier*IQR
                   - multiplier=1.5: mild outliers (typical)
                   - multiplier=3.0: extreme outliers only
                   
        Considerations:
            - Outliers may be informative (high-risk customers, rare events)
            - Robust scalers (RobustScaler) and tree models handle outliers well
            - Use "clip" to preserve data while reducing extreme influence
            - Use "filter" only when outliers are clearly erroneous
            
        Args:
            df: Input DataFrame
            strategy: Outlier handling strategy
            lower_q/upper_q: Quantiles for clip/filter (e.g., 0.01, 0.99)
            multiplier: IQR multiplier for "iqr" strategy
            numeric_cols: Columns to process; None infers all numeric
            exclude_cols: Columns to skip (e.g., target)
            
        Returns:
            DataFrame after outlier handling
        """
        if strategy is None:
            return df

        if strategy not in ("clip", "filter", "iqr"):
            raise ValueError("outlier_strategy must be None, 'clip', 'filter', or 'iqr'")

        df = df.copy()
        exclude_cols = exclude_cols or []
        
        # Infer numeric columns if not specified
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        if not numeric_cols:
            logging.info("Outlier handling skipped: no numeric columns selected.")
            return df

        # Compute bounds for each column
        bounds = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
                
            if strategy == "iqr":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo = q1 - multiplier * iqr
                hi = q3 + multiplier * iqr
            else:
                lo = series.quantile(lower_q)
                hi = series.quantile(upper_q)
            bounds[col] = (lo, hi)
            logging.info("Outlier bounds for %s: [%.2f, %.2f]", col, lo, hi)

        if not bounds:
            logging.info("Outlier handling skipped: numeric columns had all-NA.")
            return df

        # Apply strategy
        if strategy == "clip":
            for col, (lo, hi) in bounds.items():
                original = df[col].copy()
                df[col] = df[col].clip(lower=lo, upper=hi)
                clipped = (df[col] != original).sum()
                logging.info("  %s: clipped %s values (%.2f%%)", col, clipped,
                           100 * clipped / len(df))
            logging.info("Outlier strategy=clip applied to %s columns.", len(bounds))
            return df

        # "filter" or "iqr" - drop rows
        before = len(df)
        mask = np.ones(len(df), dtype=bool)
        for col, (lo, hi) in bounds.items():
            col_mask = df[col].between(lo, hi) | df[col].isna()
            excluded = (~col_mask).sum()
            logging.info("  %s: excluding %s rows (%.2f%%)", col, excluded,
                        100 * excluded / len(df))
            mask &= col_mask
            
        df = df.loc[mask].copy()
        dropped = before - len(df)
        logging.info("Outlier strategy=%s dropped %s rows total (%.2f%%).",
                    strategy, dropped, 100 * dropped / before)
        return df


# ---------------------------------------------------------------------
# Enhanced Data Exploration (EDA)
# ---------------------------------------------------------------------
class DataExploration:
    """
    Comprehensive EDA suite with modern visualizations and statistical analysis.
    
    This class generates publication-quality plots and detailed statistical
    summaries to understand data characteristics before modeling.
    
    Key Outputs:
        - Distribution analysis (histograms, Q-Q plots, statistical tests)
        - Categorical analysis (count plots with percentages)
        - Target analysis (distribution and relationships)
        - Bivariate analysis (feature vs target visualizations)
        - Outlier detection (multiple visualization techniques)
        - Correlation analysis (enhanced heatmaps)
        - Comprehensive text summary report
    """
    
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize EDA engine with configuration.
        
        Args:
            config: Transformation configuration with EDA settings
        """
        self.config = config
        
        # Create directory structure
        for dir_path in [
            self.config.eda_artifacts_dir,
            self.config.eda_distributions_dir,
            self.config.eda_relationships_dir,
            self.config.eda_outliers_dir,
            self.config.eda_correlations_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set professional theme
        if sns:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    def run(self, df: pd.DataFrame, df_name: str = "dataset") -> Dict[str, Any]:
        """
        Execute comprehensive EDA suite.
        
        Generates all enabled visualizations and statistical analyses.
        
        Args:
            df: DataFrame to analyze
            df_name: Name for labeling outputs (e.g., "train", "test")
            
        Returns:
            Dictionary with summary statistics and metadata
        """
        logging.info("\n" + "=" * 80)
        logging.info("STARTING ENHANCED EDA: %s", df_name.upper())
        logging.info("=" * 80)

        summary_stats = {}

        # 1. Basic statistics
        stats_df = self._save_basic_statistics(df, df_name)
        summary_stats['basic_stats'] = stats_df

        # 2. Missing data analysis
        missing_df = self._save_missing_analysis(df, df_name)
        summary_stats['missing_data'] = missing_df

        # Check if visualization libraries available
        if plt is None:
            logging.warning("matplotlib not available; skipping visualizations.")
            return summary_stats

        # 3. Distribution analysis
        if self.config.eda_save_distributions:
            self._save_distribution_analysis(df, df_name)

        # 4. Q-Q plots for normality
        if self.config.eda_save_qq_plots:
            self._save_qq_plots(df, df_name)

        # 5. Statistical summary
        if self.config.eda_save_statistical_summary:
            stats_summary = self._save_statistical_summary(df, df_name)
            summary_stats['statistical_summary'] = stats_summary

        # 6. Categorical analysis
        if self.config.eda_save_categorical_analysis:
            self._save_categorical_analysis(df, df_name)

        # 7. Target analysis
        if self.config.eda_save_target_analysis and self.config.target_column in df.columns:
            self._save_target_analysis(df, df_name)

        # 8. Bivariate analysis
        if self.config.eda_save_bivariate and self.config.target_column in df.columns:
            self._save_bivariate_analysis(df, df_name)

        # 9. Outlier detection
        if self.config.eda_save_boxplots:
            self._save_outlier_detection(df, df_name)

        # 10. Correlation analysis
        if self.config.eda_save_correlation:
            corr_df = self._save_correlation_analysis(df, df_name)
            summary_stats['correlation'] = corr_df

        # 11. Generate comprehensive summary report
        self._generate_summary_report(df, df_name, summary_stats)

        logging.info("\n" + "=" * 80)
        logging.info("EDA COMPLETE: %s", df_name.upper())
        logging.info("=" * 80)

        return summary_stats

    def _save_basic_statistics(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Generate and save basic statistical summary.
        
        Includes count, mean, std, min, quartiles, max for numeric features.
        
        Args:
            df: Input DataFrame
            df_name: Name for output file
            
        Returns:
            Statistics DataFrame
        """
        stats = df.describe(include="all").transpose()
        out_path = self.config.eda_artifacts_dir / f"{df_name}_stats_summary.csv"
        stats.to_csv(out_path)
        logging.info("✓ Saved basic statistics: %s", out_path)
        return stats

    def _save_missing_analysis(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Analyze and visualize missing data patterns.
        
        Creates:
            - CSV with missing counts and percentages
            - Bar plot showing missingness by feature
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
            
        Returns:
            Missing data DataFrame
        """
        miss = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(3),
            "dtype": df.dtypes.astype(str),
        }).sort_values("missing_pct", ascending=False)
        
        out_path = self.config.eda_artifacts_dir / f"{df_name}_missingness.csv"
        miss.to_csv(out_path)
        logging.info("✓ Saved missingness report: %s", out_path)

        # Visualize if there's any missing data
        if miss['missing_count'].sum() > 0 and plt is not None:
            missing_features = miss[miss['missing_count'] > 0].head(20)
            if not missing_features.empty:
                fig, ax = plt.subplots(figsize=(12, max(6, len(missing_features) * 0.3)))
                if sns:
                    sns.barplot(data=missing_features.reset_index(), 
                              y='index', x='missing_pct', ax=ax, palette='viridis')
                else:
                    ax.barh(range(len(missing_features)), missing_features['missing_pct'])
                    ax.set_yticks(range(len(missing_features)))
                    ax.set_yticklabels(missing_features.index)
                
                ax.set_xlabel('Missing Percentage (%)', fontsize=12)
                ax.set_ylabel('Feature', fontsize=12)
                ax.set_title(f'{df_name.capitalize()}: Missing Data Analysis', 
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.config.eda_artifacts_dir / f"{df_name}_missing_data.png",
                          dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("✓ Saved missing data visualization")

        return miss

    def _save_distribution_analysis(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Generate distribution plots for numeric features.
        
        Creates histograms with KDE overlay showing:
            - Data distribution shape (normal, skewed, bimodal, etc.)
            - Central tendency and spread
            - Potential outliers
            
        Each plot includes:
            - Histogram bars
            - KDE smooth curve
            - Mean and median lines
            - Statistical annotations
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            logging.info("No numeric columns for distribution analysis.")
            return

        # Limit number of plots
        numeric_cols = numeric_cols[:self.config.eda_max_plots_per_type]
        logging.info("Generating distribution plots for %s numeric features...", len(numeric_cols))

        for col in numeric_cols:
            try:
                series = df[col].dropna()
                if series.empty or len(series) < 2:
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))
                
                if sns:
                    sns.histplot(series, kde=True, bins=30, ax=ax, color='steelblue')
                else:
                    ax.hist(series, bins=30, density=True, alpha=0.7, color='steelblue')

                # Add mean and median lines
                mean_val = series.mean()
                median_val = series.median()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

                # Add statistics box
                stats_text = f'Std: {series.std():.2f}\nMin: {series.min():.2f}\nMax: {series.max():.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'Distribution: {col}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                out_path = self.config.eda_distributions_dir / f"{df_name}_{col}_distribution.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logging.warning("Distribution plot failed for %s: %s", col, str(e))

        logging.info("✓ Saved distribution plots to %s", self.config.eda_distributions_dir)

    def _save_qq_plots(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Generate Q-Q plots to assess normality of numeric features.
        
        Q-Q (Quantile-Quantile) plots compare data distribution to normal distribution:
            - Points on diagonal = normally distributed
            - Curve above diagonal = right-skewed
            - Curve below diagonal = left-skewed
            - S-shape = heavy tails
            
        Useful for:
            - Identifying transformation needs (log, sqrt, etc.)
            - Validating assumptions for parametric tests
            - Understanding data shape
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
        """
        if stats is None:
            logging.info("scipy.stats not available; skipping Q-Q plots.")
            return

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return

        numeric_cols = numeric_cols[:self.config.eda_max_plots_per_type]
        logging.info("Generating Q-Q plots for %s numeric features...", len(numeric_cols))

        for col in numeric_cols:
            try:
                series = df[col].dropna()
                if series.empty or len(series) < 3:
                    continue

                fig, ax = plt.subplots(figsize=(8, 8))
                stats.probplot(series, dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot: {col} (Normality Check)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                out_path = self.config.eda_distributions_dir / f"{df_name}_{col}_qq_plot.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logging.warning("Q-Q plot failed for %s: %s", col, str(e))

        logging.info("✓ Saved Q-Q plots to %s", self.config.eda_distributions_dir)

    def _save_statistical_summary(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Compute extended statistical metrics for numeric features.
        
        Metrics include:
            - Skewness: Measure of asymmetry (0 = symmetric, + = right tail, - = left tail)
            - Kurtosis: Measure of tail heaviness (3 = normal, >3 = heavy tails)
            - Normality tests: Shapiro-Wilk and D'Agostino-Pearson tests
            
        These metrics help identify:
            - Features needing transformation
            - Potential modeling issues
            - Data quality problems
            
        Args:
            df: Input DataFrame
            df_name: Name for output file
            
        Returns:
            Statistical summary DataFrame
        """
        if skew is None or kurtosis is None:
            logging.info("scipy.stats not available; skipping statistical summary.")
            return pd.DataFrame()

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()

        logging.info("Computing extended statistical summary...")
        
        stats_list = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty or len(series) < 3:
                continue

            stat_dict = {
                'feature': col,
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skewness': skew(series),
                'kurtosis': kurtosis(series),
            }

            # Normality tests (only if enough samples)
            if len(series) >= 20:
                try:
                    if shapiro is not None:
                        _, p_shapiro = shapiro(series[:5000])  # Limit to 5000 for performance
                        stat_dict['shapiro_p_value'] = p_shapiro
                        stat_dict['is_normal_shapiro'] = p_shapiro > 0.05
                except Exception:
                    pass

                try:
                    if normaltest is not None:
                        _, p_normal = normaltest(series)
                        stat_dict['normaltest_p_value'] = p_normal
                        stat_dict['is_normal_normaltest'] = p_normal > 0.05
                except Exception:
                    pass

            stats_list.append(stat_dict)

        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            out_path = self.config.eda_artifacts_dir / f"{df_name}_statistical_summary.csv"
            stats_df.to_csv(out_path, index=False)
            logging.info("✓ Saved statistical summary: %s", out_path)
            return stats_df

        return pd.DataFrame()

    def _save_categorical_analysis(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Generate count plots for categorical features.
        
        Creates bar plots showing:
            - Frequency of each category
            - Percentage distribution
            - Top N categories (if many categories exist)
            
        Helps identify:
            - Class imbalance
            - Rare categories
            - Dominant patterns
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
        """
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        
        # Remove target if present
        if self.config.target_column in cat_cols:
            cat_cols.remove(self.config.target_column)

        if not cat_cols:
            logging.info("No categorical columns for analysis.")
            return

        cat_cols = cat_cols[:self.config.eda_max_plots_per_type]
        logging.info("Generating categorical analysis for %s features...", len(cat_cols))

        for col in cat_cols:
            try:
                # Get value counts
                value_counts = df[col].value_counts().head(self.config.eda_categorical_top_n)
                if value_counts.empty:
                    continue

                fig, ax = plt.subplots(figsize=(12, max(6, len(value_counts) * 0.4)))
                
                if sns:
                    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')
                else:
                    ax.barh(range(len(value_counts)), value_counts.values)
                    ax.set_yticks(range(len(value_counts)))
                    ax.set_yticklabels(value_counts.index)

                # Add percentage labels
                total = df[col].count()
                for i, (idx, val) in enumerate(value_counts.items()):
                    pct = 100 * val / total
                    ax.text(val, i, f' {val} ({pct:.1f}%)', va='center', fontsize=10)

                ax.set_xlabel('Count', fontsize=12)
                ax.set_ylabel(col, fontsize=12)
                title = f'{df_name.capitalize()}: {col} Distribution'
                if len(df[col].unique()) > self.config.eda_categorical_top_n:
                    title += f' (Top {self.config.eda_categorical_top_n})'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                out_path = self.config.eda_distributions_dir / f"{df_name}_{col}_counts.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logging.warning("Categorical plot failed for %s: %s", col, str(e))

        logging.info("✓ Saved categorical analysis to %s", self.config.eda_distributions_dir)

    def _save_target_analysis(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Analyze target variable distribution and characteristics.
        
        Creates comprehensive target analysis:
            - Distribution histogram with statistics
            - Log-scale distribution (if target is positive-valued)
            - Box plot for outlier detection
            
        Understanding target distribution is crucial for:
            - Model selection (regression vs classification)
            - Loss function choice
            - Transformation needs
            - Evaluation metric selection
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
        """
        target = self.config.target_column
        if target not in df.columns:
            return

        logging.info("Analyzing target variable: %s", target)
        
        target_series = df[target].dropna()
        if target_series.empty:
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if sns:
            sns.histplot(target_series, kde=True, bins=40, ax=ax1, color='coral')
        else:
            ax1.hist(target_series, bins=40, density=True, alpha=0.7, color='coral')

        mean_val = target_series.mean()
        median_val = target_series.median()
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax1.set_xlabel(target, fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Target Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Log-scale distribution (if positive)
        ax2 = fig.add_subplot(gs[0, 1])
        if (target_series > 0).all():
            log_target = np.log1p(target_series)
            if sns:
                sns.histplot(log_target, kde=True, bins=40, ax=ax2, color='teal')
            else:
                ax2.hist(log_target, bins=40, density=True, alpha=0.7, color='teal')
            ax2.set_xlabel(f'log({target})', fontsize=11)
            ax2.set_ylabel('Density', fontsize=11)
            ax2.set_title('Target Distribution (Log Scale)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Log scale not applicable\n(negative values present)',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            ax2.set_title('Target Distribution (Log Scale)', fontsize=12, fontweight='bold')

        # 3. Box plot
        ax3 = fig.add_subplot(gs[0, 2])
        if sns:
            sns.boxplot(y=target_series, ax=ax3, color='lightblue')
        else:
            ax3.boxplot(target_series)
        ax3.set_ylabel(target, fontsize=11)
        ax3.set_title('Target Box Plot (Outliers)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{df_name.capitalize()}: Target Analysis ({target})', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(self.config.eda_artifacts_dir / f"{df_name}_target_analysis.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("✓ Saved target analysis")

    def _save_bivariate_analysis(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Analyze relationships between features and target.
        
        Creates:
            - Scatter plots: Numeric features vs target (shows correlation)
            - Box plots: Categorical features vs target (shows group differences)
            
        These plots reveal:
            - Linear/non-linear relationships
            - Group effects
            - Interaction patterns
            - Predictive power
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
        """
        target = self.config.target_column
        if target not in df.columns:
            return

        logging.info("Generating bivariate analysis (features vs target)...")

        # Numeric vs target (scatter plots)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target]
        numeric_cols = numeric_cols[:self.config.eda_max_plots_per_type]

        for col in numeric_cols:
            try:
                plot_df = df[[col, target]].dropna()
                if plot_df.empty or len(plot_df) < 2:
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))
                
                if sns:
                    sns.scatterplot(data=plot_df, x=col, y=target, alpha=0.5, ax=ax)
                    sns.regplot(data=plot_df, x=col, y=target, scatter=False, 
                              color='red', ax=ax, line_kws={'linewidth': 2})
                else:
                    ax.scatter(plot_df[col], plot_df[target], alpha=0.5)

                # Calculate correlation
                corr = plot_df[col].corr(plot_df[target])
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=ax.transAxes, fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel(target, fontsize=12)
                ax.set_title(f'{col} vs {target}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                out_path = self.config.eda_relationships_dir / f"{df_name}_{col}_vs_target.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logging.warning("Scatter plot failed for %s: %s", col, str(e))

        # Categorical vs target (box plots)
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c != target]
        cat_cols = cat_cols[:self.config.eda_max_plots_per_type]

        for col in cat_cols:
            try:
                plot_df = df[[col, target]].dropna()
                if plot_df.empty or len(plot_df[col].unique()) < 2:
                    continue

                # Limit categories for readability
                top_cats = plot_df[col].value_counts().head(15).index
                plot_df = plot_df[plot_df[col].isin(top_cats)]

                fig, ax = plt.subplots(figsize=(12, 6))
                
                if sns:
                    sns.boxplot(data=plot_df, x=col, y=target, ax=ax, palette='Set2')
                else:
                    plot_df.boxplot(column=target, by=col, ax=ax)

                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel(target, fontsize=12)
                ax.set_title(f'{col} vs {target}', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                out_path = self.config.eda_relationships_dir / f"{df_name}_{col}_vs_target.png"
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logging.warning("Box plot failed for %s: %s", col, str(e))

        logging.info("✓ Saved bivariate analysis to %s", self.config.eda_relationships_dir)

    def _save_outlier_detection(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Generate comprehensive outlier detection visualizations.
        
        Creates:
            - Multi-feature box plot panel
            - Individual box plots for detailed view
            
        Box plots show:
            - Median (line in box)
            - IQR (box height = Q1 to Q3)
            - Whiskers (1.5 * IQR)
            - Outliers (points beyond whiskers)
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return

        logging.info("Generating outlier detection plots...")

        # Multi-feature panel
        plot_cols = numeric_cols[:min(len(numeric_cols), 20)]
        if plot_cols:
            fig, ax = plt.subplots(figsize=(max(12, len(plot_cols) * 0.8), 8))
            if sns:
                sns.boxplot(data=df[plot_cols], ax=ax, palette="Set3")
            else:
                df[plot_cols].boxplot(ax=ax)
            
            ax.set_title(f'{df_name.capitalize()}: Outlier Detection (Box Plots)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Values', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.config.eda_outliers_dir / f"{df_name}_outliers_panel.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

        logging.info("✓ Saved outlier detection plots to %s", self.config.eda_outliers_dir)

    def _save_correlation_analysis(self, df: pd.DataFrame, df_name: str) -> Optional[pd.DataFrame]:
        """
        Perform correlation analysis with enhanced visualization.
        
        Creates:
            - Correlation matrix CSV
            - Annotated heatmap with hierarchical clustering
            - High correlation pairs report
            
        Correlation analysis reveals:
            - Multicollinearity (high feature-feature correlation)
            - Predictive relationships (feature-target correlation)
            - Redundant features
            
        Args:
            df: Input DataFrame
            df_name: Name for output files
            
        Returns:
            Correlation DataFrame
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols or len(numeric_cols) < 2:
            logging.info("Insufficient numeric columns for correlation analysis.")
            return None

        logging.info("Performing correlation analysis...")

        # Compute correlation matrix
        corr = df[numeric_cols].corr(numeric_only=True)
        
        # Save CSV
        out_path = self.config.eda_correlations_dir / f"{df_name}_correlation.csv"
        corr.to_csv(out_path)
        logging.info("✓ Saved correlation matrix: %s", out_path)

        # Save high correlations
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_val = corr.iloc[i, j]
                if abs(corr_val) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'feature_1': corr.columns[i],
                        'feature_2': corr.columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', 
                                                                     key=abs, ascending=False)
            high_corr_path = self.config.eda_correlations_dir / f"{df_name}_high_correlations.csv"
            high_corr_df.to_csv(high_corr_path, index=False)
            logging.info("✓ Found %s high correlation pairs (|r| > 0.7)", len(high_corr_pairs))

        # Generate heatmap
        if self.config.eda_save_heatmap:
            fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.6), 
                                           max(8, len(numeric_cols) * 0.5)))
            
            if sns:
                # Use hierarchical clustering for better visualization
                try:
                    sns.clustermap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                                 linewidths=0.5, figsize=(max(12, len(numeric_cols) * 0.7),
                                                         max(10, len(numeric_cols) * 0.6)),
                                 cbar_kws={"shrink": 0.8})
                    plt.suptitle(f'{df_name.capitalize()}: Correlation Heatmap (Clustered)',
                               fontsize=14, fontweight='bold', y=0.98)
                    plt.savefig(self.config.eda_correlations_dir / f"{df_name}_correlation_clustered.png",
                              dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception:
                    # Fallback to regular heatmap
                    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                              linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
            else:
                im = ax.imshow(corr, cmap="coolwarm", aspect='auto')
                plt.colorbar(im, ax=ax)
                # Add annotations
                for i in range(len(numeric_cols)):
                    for j in range(len(numeric_cols)):
                        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                               ha="center", va="center", fontsize=8)
            
            if not sns or True:  # For regular heatmap
                ax.set_xticks(range(len(numeric_cols)))
                ax.set_yticks(range(len(numeric_cols)))
                ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=10)
                ax.set_yticklabels(numeric_cols, rotation=0, fontsize=10)
                ax.set_title(f'{df_name.capitalize()}: Correlation Heatmap',
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.config.eda_correlations_dir / f"{df_name}_correlation_heatmap.png",
                          dpi=300, bbox_inches='tight')
                plt.close()

            logging.info("✓ Saved correlation heatmap")

        return corr

    def _generate_summary_report(self, df: pd.DataFrame, df_name: str, 
                                 summary_stats: Dict[str, Any]) -> None:
        """
        Generate comprehensive text summary report.
        
        Creates a detailed report including:
            - Dataset overview
            - Missing data summary
            - Statistical insights
            - Correlation highlights
            - Recommendations
            
        Args:
            df: Input DataFrame
            df_name: Dataset name
            summary_stats: Dictionary with computed statistics
        """
        report_lines = [
            "=" * 80,
            f"DATA TRANSFORMATION - EDA SUMMARY: {df_name.upper()}",
            "=" * 80,
            "",
            "DATASET OVERVIEW",
            "-" * 80,
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "",
        ]

        # Column types
        dtypes_count = df.dtypes.value_counts()
        report_lines.append("Column Types:")
        for dtype, count in dtypes_count.items():
            report_lines.append(f"  {dtype}: {count} columns")
        report_lines.append("")

        # Missing data
        if 'missing_data' in summary_stats:
            miss_df = summary_stats['missing_data']
            total_missing = miss_df['missing_count'].sum()
            if total_missing > 0:
                report_lines.extend([
                    "MISSING DATA",
                    "-" * 80,
                    f"Total missing values: {total_missing:,}",
                    f"Percentage of dataset: {100 * total_missing / (df.shape[0] * df.shape[1]):.2f}%",
                    "",
                    "Features with missing data (top 10):",
                ])
                for _, row in miss_df.head(10).iterrows():
                    if row['missing_count'] > 0:
                        report_lines.append(f"  {row.name}: {row['missing_count']:,} ({row['missing_pct']:.2f}%)")
                report_lines.append("")

        # Statistical summary
        if 'statistical_summary' in summary_stats and not summary_stats['statistical_summary'].empty:
            stats_df = summary_stats['statistical_summary']
            report_lines.extend([
                "STATISTICAL INSIGHTS",
                "-" * 80,
            ])
            
            # Skewed features
            if 'skewness' in stats_df.columns:
                skewed = stats_df[abs(stats_df['skewness']) > 1]
                if not skewed.empty:
                    report_lines.append(f"Highly skewed features ({len(skewed)}):")
                    for _, row in skewed.head(5).iterrows():
                        report_lines.append(f"  {row['feature']}: skewness = {row['skewness']:.2f}")
                    report_lines.append("")

            # Non-normal features
            if 'is_normal_shapiro' in stats_df.columns:
                non_normal = stats_df[stats_df['is_normal_shapiro'] == False]
                if not non_normal.empty:
                    report_lines.append(f"Non-normal features (Shapiro test, p<0.05): {len(non_normal)}")
                    report_lines.append("")

        # Correlation insights
        if 'correlation' in summary_stats and summary_stats['correlation'] is not None:
            corr = summary_stats['correlation']
            target = self.config.target_column
            
            report_lines.extend([
                "CORRELATION INSIGHTS",
                "-" * 80,
            ])
            
            if target in corr.columns:
                target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
                report_lines.append(f"Top correlations with target ({target}):")
                for feat, corr_val in target_corr.head(5).items():
                    report_lines.append(f"  {feat}: {corr_val:.3f}")
                report_lines.append("")

        # Recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 80,
        ])
        
        recommendations = []
        
        # Check for missing data
        if 'missing_data' in summary_stats:
            miss_df = summary_stats['missing_data']
            high_missing = miss_df[miss_df['missing_pct'] > 40]
            if not high_missing.empty:
                recommendations.append(
                    f"• Consider dropping {len(high_missing)} features with >40% missing data"
                )
        
        # Check for skewness
        if 'statistical_summary' in summary_stats and not summary_stats['statistical_summary'].empty:
            stats_df = summary_stats['statistical_summary']
            if 'skewness' in stats_df.columns:
                highly_skewed = stats_df[abs(stats_df['skewness']) > 2]
                if not highly_skewed.empty:
                    recommendations.append(
                        f"• Apply log/sqrt transformation to {len(highly_skewed)} highly skewed features"
                    )
        
        # Check for outliers (if numeric columns exist)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            recommendations.append(
                "• Review outlier detection plots before deciding on outlier handling strategy"
            )
        
        if recommendations:
            report_lines.extend(recommendations)
        else:
            report_lines.append("• Data appears clean and ready for modeling")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])

        # Save report
        report_path = self.config.eda_artifacts_dir / f"{df_name}_eda_summary.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logging.info("✓ Saved comprehensive EDA summary: %s", report_path)


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------
class FeatureEngineer:
    """
    Schema-aware feature engineering with leakage prevention.
    
    Creates interpretable features that enhance model performance without
    using target information. All features are conditional on column presence
    to handle schema drift gracefully.
    
    Feature Categories:
        - Binary flags: Simple yes/no indicators
        - Ratios: Relationships between features
        - Transformations: Non-linear representations (log, polynomial)
        - Interactions: Combined effects of multiple features
    """
    
    def __init__(self, target_col: str, disallow_target_leakage: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            target_col: Name of target column to exclude from features
            disallow_target_leakage: Whether to prevent target use in features
        """
        self.target_col = target_col
        self.disallow_target_leakage = disallow_target_leakage

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features if source columns exist.
        
        Features created:
            1. has_dependents: Binary flag (0/1) indicating presence of dependents
            2. income_per_dependent: Income divided by dependents (handles zero)
            3. log_income_lakhs: Log-transformed income (reduces skewness)
            4. age_squared: Non-linear age effect
            5. age_income_interaction: Combined effect of age and income
            
        All features are safe from target leakage and handle missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()

        # Leakage prevention guard
        if self.disallow_target_leakage and self.target_col in df.columns:
            logging.info("Target leakage prevention active: no target-derived features created.")

        def has_cols(cols: List[str]) -> bool:
            """Check if all required columns exist."""
            return all(c in df.columns for c in cols)

        features_added = 0

        # 1) Binary flag: has_dependents
        # Useful for: Identifying lifestyle differences between parents and non-parents
        if "number_of_dependants" in df.columns:
            df["has_dependents"] = (
                pd.to_numeric(df["number_of_dependants"], errors="coerce")
                .fillna(0) > 0
            ).astype(int)
            features_added += 1
            logging.info("  Added feature: has_dependents")

        # 2) Ratio: income_per_dependent
        # Useful for: Measuring financial burden per dependent
        # Handles zero division by replacing 0 dependents with NaN
        if has_cols(["income_lakhs", "number_of_dependants"]):
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            deps = pd.to_numeric(df["number_of_dependants"], errors="coerce").replace(0, np.nan)
            df["income_per_dependent"] = (income / deps).replace([np.inf, -np.inf], np.nan)
            features_added += 1
            logging.info("  Added feature: income_per_dependent")

        # 3) Transformation: log_income_lakhs
        # Useful for: Reducing right-skewness in income distribution
        # Uses log1p to handle zeros gracefully
        if "income_lakhs" in df.columns:
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            df["log_income_lakhs"] = np.log1p(income.clip(lower=0))
            features_added += 1
            logging.info("  Added feature: log_income_lakhs")

        # 4) Polynomial: age_squared
        # Useful for: Capturing non-linear age effects (e.g., U-shaped relationships)
        if "age" in df.columns:
            age = pd.to_numeric(df["age"], errors="coerce")
            df["age_squared"] = age ** 2
            features_added += 1
            logging.info("  Added feature: age_squared")

        # 5) Interaction: age_income_interaction
        # Useful for: Capturing combined effects (e.g., high income matters more for older people)
        if has_cols(["age", "income_lakhs"]):
            age = pd.to_numeric(df["age"], errors="coerce")
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            df["age_income_interaction"] = age * income
            features_added += 1
            logging.info("  Added feature: age_income_interaction")

        logging.info("Feature engineering completed: %s new features created.", features_added)
        return df


# ---------------------------------------------------------------------
# Data Transformation (Main Orchestrator)
# ---------------------------------------------------------------------
class DataTransformation:
    """
    Main orchestrator for end-to-end data transformation pipeline.
    
    Coordinates:
        1. Data loading and schema standardization
        2. Cleaning (missing values, duplicates, outliers)
        3. Feature engineering
        4. EDA (comprehensive visualization and analysis)
        5. Multicollinearity detection (VIF)
        6. Preprocessing (imputation, scaling, encoding)
        7. Artifact saving (arrays, preprocessor, reports)
        
    Design principles:
        - Fit on train, transform on test (no data leakage)
        - Config-driven with sensible defaults
        - Graceful degradation (features work even if columns missing)
        - Comprehensive logging and error handling
        - Reproducible (saves all artifacts needed for inference)
    """
    
    def __init__(self, config: DataTransformationConfig | None = None):
        """
        Initialize transformation pipeline.
        
        Args:
            config: Transformation configuration; uses defaults if None
        """
        self.config = config or DataTransformationConfig()
        self.feature_engineer = FeatureEngineer(
            target_col=self.config.target_column,
            disallow_target_leakage=self.config.disallow_target_leakage,
        )
        self.eda = DataExploration(self.config)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    def _safe_onehot(self) -> OneHotEncoder:
        """
        Create OneHotEncoder compatible with different sklearn versions.
        
        sklearn version compatibility:
            - >=1.2: uses sparse_output parameter
            - <1.2: uses sparse parameter
            
        Returns:
            OneHotEncoder instance
        """
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _standardize_schema(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Standardize DataFrame schema early in pipeline.
        
        Ensures consistent column naming across datasets to prevent
        mismatches between train and test.
        
        Args:
            df: Input DataFrame
            df_name: Name for logging
            
        Returns:
            DataFrame with standardized schema
        """
        if self.config.standardize_column_names:
            logging.info("[%s] Standardizing column names.", df_name)
            df = DataCleaning.standardize_columns(df)

        if df.empty:
            raise ValueError(f"{df_name} dataset is empty after load/standardization.")
        
        logging.info("[%s] Shape after standardization: %s", df_name, df.shape)
        return df

    def _validate_target_presence(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Validate target column presence in datasets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Raises:
            ValueError: If target missing from train or (optionally) test
        """
        target = self.config.target_column
        
        if target not in train_df.columns:
            raise ValueError(
                f"Target column '{target}' missing from train dataset. "
                f"Available columns: {list(train_df.columns)}"
            )
        
        if self.config.require_target_in_test and target not in test_df.columns:
            raise ValueError(
                f"Target column '{target}' missing from test dataset but "
                f"require_target_in_test=True. Available columns: {list(test_df.columns)}"
            )
        
        logging.info("Target validation passed: '%s' found in datasets.", target)

    def _infer_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Infer or use configured numeric and categorical columns.
        
        Priority:
            1. Use config columns if specified
            2. Infer from dtypes otherwise
            
        Always excludes target from predictors.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (numeric_columns, categorical_columns)
        """
        # Numeric columns
        if self.config.numerical_columns is not None:
            numeric_cols = [c for c in self.config.numerical_columns if c in df.columns]
            logging.info("Using configured numeric columns: %s", len(numeric_cols))
        else:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            logging.info("Inferred numeric columns: %s", len(numeric_cols))

        # Categorical columns
        if self.config.categorical_columns is not None:
            cat_cols = [c for c in self.config.categorical_columns if c in df.columns]
            logging.info("Using configured categorical columns: %s", len(cat_cols))
        else:
            cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            logging.info("Inferred categorical columns: %s", len(cat_cols))

        # Remove target from predictors
        target = self.config.target_column
        numeric_cols = [c for c in numeric_cols if c != target]
        cat_cols = [c for c in cat_cols if c != target]

        return numeric_cols, cat_cols

    def _coerce_numeric(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Coerce columns to numeric type, converting invalid values to NaN.
        
        Handles cases where numeric columns contain string values due to
        data quality issues or mixed types.
        
        Args:
            df: Input DataFrame
            numeric_cols: Columns to coerce
            
        Returns:
            DataFrame with numeric columns coerced
        """
        df = df.copy()
        for col in numeric_cols:
            if col in df.columns:
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if original_dtype != df[col].dtype:
                    na_introduced = df[col].isna().sum()
                    logging.info("  Coerced %s to numeric (introduced %s NAs)", col, na_introduced)
        return df

    def _build_preprocessor(
        self, 
        numeric_features: List[str], 
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """
        Build sklearn preprocessing pipeline.
        
        Numeric pipeline:
            1. SimpleImputer(median): Handles missing values robustly
            2. StandardScaler: Normalizes to mean=0, std=1 for scale-sensitive models
            
        Categorical pipeline:
            1. SimpleImputer(most_frequent): Handles missing values
            2. OneHotEncoder: Creates binary columns for each category
            
        Args:
            numeric_features: Numeric column names
            categorical_features: Categorical column names
            
        Returns:
            Fitted ColumnTransformer
        """
        logging.info("\nBuilding preprocessing pipeline:")
        logging.info("  Numeric features: %s", len(numeric_features))
        logging.info("  Categorical features: %s", len(categorical_features))

        # Numeric preprocessing: median imputation + standardization
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical preprocessing: mode imputation + one-hot encoding
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", self._safe_onehot()),
            ]
        )

        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",  # Drop columns not in either list
        )
        
        return preprocessor

    # -------------------------------------------------------------------------
    # VIF (Multicollinearity Detection)
    # -------------------------------------------------------------------------
    def _calculate_vif(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF) for numeric features.
        
        VIF measures multicollinearity:
            - VIF = 1: No correlation with other features
            - VIF < 5: Acceptable
            - VIF 5-10: Moderate multicollinearity (caution)
            - VIF > 10: High multicollinearity (consider removing)
            
        High VIF can:
            - Destabilize linear models (unstable coefficients)
            - Inflate standard errors
            - Make interpretation difficult
            
        Args:
            df: DataFrame with numeric features
            numeric_cols: Column names to analyze
            
        Returns:
            DataFrame with feature names and VIF scores
        """
        if not self.config.enable_vif:
            return pd.DataFrame()

        if variance_inflation_factor is None:
            logging.warning("statsmodels not available; VIF computation skipped.")
            return pd.DataFrame()

        if not numeric_cols:
            logging.info("VIF skipped: no numeric columns to analyze.")
            return pd.DataFrame()

        # Prepare data: remove inf/nan
        X = df[numeric_cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
        
        if X.empty or X.shape[0] < 10:
            logging.warning("VIF skipped: insufficient non-missing numeric rows (need >=10).")
            return pd.DataFrame()

        # Compute VIF for each feature
        logging.info("Computing VIF for %s numeric features...", len(numeric_cols))
        vif_list = []
        for i, col in enumerate(numeric_cols):
            try:
                vif_value = variance_inflation_factor(X.values, i)
                vif_list.append({"feature": col, "VIF": vif_value})
            except Exception as e:
                logging.warning("VIF computation failed for %s: %s", col, str(e))
        
        if not vif_list:
            return pd.DataFrame()
        
        vif_df = pd.DataFrame(vif_list).sort_values("VIF", ascending=False).reset_index(drop=True)
        return vif_df

    def _prune_by_vif(self, train_df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """
        Iteratively remove features with high VIF.
        
        Algorithm:
            1. Compute VIF for all features
            2. Find feature with highest VIF > threshold
            3. Remove that feature
            4. Recompute VIF and repeat
            5. Stop when all VIF <= threshold or max drops reached
            
        Protection:
            - Exempt features: Never dropped (protect critical features)
            - Max drops: Limit removals for safety
            
        Args:
            train_df: Training DataFrame
            numeric_cols: Initial numeric columns
            
        Returns:
            List of numeric columns after pruning
        """
        if not self.config.enable_vif:
            return numeric_cols

        # Initial VIF calculation
        vif_df = self._calculate_vif(train_df, numeric_cols)
        if vif_df.empty:
            return numeric_cols

        # Save initial VIF report
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        vif_df.to_csv(self.config.vif_report_path, index=False)
        logging.info("✓ Saved initial VIF report: %s", self.config.vif_report_path)

        # Log high VIF features
        high_vif = vif_df[vif_df['VIF'] > self.config.vif_threshold]
        if not high_vif.empty:
            logging.info("Features with VIF > %.1f:", self.config.vif_threshold)
            for _, row in high_vif.iterrows():
                logging.info("  %s: VIF = %.2f", row['feature'], row['VIF'])

        # Iterative pruning
        remaining = numeric_cols[:]
        drops = 0

        while drops < self.config.vif_max_features_to_drop:
            vif_df = self._calculate_vif(train_df, remaining)
            if vif_df.empty:
                break

            # Find highest VIF
            top = vif_df.iloc[0]
            top_feat = str(top["feature"])
            top_vif = float(top["VIF"])

            # Stop if threshold satisfied
            if top_vif <= self.config.vif_threshold:
                logging.info("VIF pruning complete: all features have VIF <= %.1f", 
                           self.config.vif_threshold)
                break

            # Check if exempt
            if top_feat in self.config.vif_exempt_features:
                # Find next non-exempt feature
                non_exempt = vif_df[~vif_df["feature"].isin(self.config.vif_exempt_features)]
                if non_exempt.empty:
                    logging.warning("All high-VIF features are exempt; stopping VIF pruning.")
                    break
                    
                top = non_exempt.iloc[0]
                top_feat = str(top["feature"])
                top_vif = float(top["VIF"])

                if top_vif <= self.config.vif_threshold:
                    break

            # Drop feature
            logging.warning("Dropping feature due to high VIF: %s (VIF=%.2f)", top_feat, top_vif)
            remaining.remove(top_feat)
            drops += 1

        if drops > 0:
            logging.info("✓ VIF pruning dropped %s feature(s). Remaining: %s", drops, len(remaining))
            # Save final VIF report
            final_vif = self._calculate_vif(train_df, remaining)
            if not final_vif.empty:
                final_vif_path = self.config.vif_report_path.parent / "vif_report_final.csv"
                final_vif.to_csv(final_vif_path, index=False)
                logging.info("✓ Saved final VIF report: %s", final_vif_path)
        else:
            logging.info("No features dropped by VIF pruning.")

        return remaining

    # -------------------------------------------------------------------------
    # Main Transformation Pipeline
    # -------------------------------------------------------------------------
    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Execute complete transformation pipeline.
        
        Pipeline steps:
            1. Load train and test CSVs
            2. Standardize schema
            3. Validate target presence
            4. Clean data (missing, duplicates, outliers)
            5. Engineer features
            6. Infer feature types
            7. Perform EDA (comprehensive)
            8. Check multicollinearity (VIF)
            9. Build preprocessing pipeline
            10. Fit on train, transform both
            11. Save artifacts
            
        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV
            
        Returns:
            Tuple of:
                - train_arr: numpy array [X_train | y_train]
                - test_arr: numpy array [X_test | y_test] or [X_test] if no target
                - preprocessor_path: string path to saved preprocessor
                
        Raises:
            CustomException: If any step fails
        """
        logging.info("\n" + "=" * 80)
        logging.info("STARTING DATA TRANSFORMATION PIPELINE")
        logging.info("=" * 80)
        
        try:
            # Step 1: Load data
            logging.info("\nStep 1: Loading data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("  Train shape: %s", train_df.shape)
            logging.info("  Test shape: %s", test_df.shape)

            # Step 2: Standardize schema
            logging.info("\nStep 2: Standardizing schema...")
            train_df = self._standardize_schema(train_df, "train")
            test_df = self._standardize_schema(test_df, "test")

            # Step 3: Validate target
            logging.info("\nStep 3: Validating target column...")
            self._validate_target_presence(train_df, test_df)

            # Step 4: Cleaning
            if self.config.enable_cleaning:
                logging.info("\nStep 4: Cleaning data...")
                train_df = DataCleaning.handle_missing_values(
                    train_df, self.config.missing_value_strategy
                )
                test_df = DataCleaning.handle_missing_values(
                    test_df, self.config.missing_value_strategy
                )
                train_df = DataCleaning.handle_duplicates(train_df, self.config.drop_duplicates)
                test_df = DataCleaning.handle_duplicates(test_df, self.config.drop_duplicates)
            else:
                logging.info("\nStep 4: Cleaning disabled (skipped)")

            # Step 5: Feature engineering
            if self.config.enable_feature_engineering:
                logging.info("\nStep 5: Feature engineering...")
                train_df = self.feature_engineer.add_features(train_df)
                test_df = self.feature_engineer.add_features(test_df)
            else:
                logging.info("\nStep 5: Feature engineering disabled (skipped)")

            # Step 6: Infer columns
            logging.info("\nStep 6: Inferring feature types...")
            numeric_cols, cat_cols = self._infer_columns(train_df)

            if self.config.use_only_categorical:
                numeric_cols = []
                logging.info("  Using ONLY categorical predictors (baseline mode).")

            logging.info("  Numeric features: %s", len(numeric_cols))
            logging.info("  Categorical features: %s", len(cat_cols))

            # Step 7: Coerce numeric
            train_df = self._coerce_numeric(train_df, numeric_cols)
            test_df = self._coerce_numeric(test_df, numeric_cols)

            # Step 8: Outlier handling
            if self.config.enable_cleaning and self.config.outlier_strategy is not None:
                logging.info("\nStep 8: Handling outliers (strategy=%s)...", 
                           self.config.outlier_strategy)
                outlier_cols = (
                    list(self.config.outlier_numeric_cols)
                    if self.config.outlier_numeric_cols is not None
                    else numeric_cols
                )
                train_df = DataCleaning.handle_outliers(
                    train_df,
                    strategy=self.config.outlier_strategy,
                    lower_q=self.config.outlier_lower_q,
                    upper_q=self.config.outlier_upper_q,
                    multiplier=self.config.outlier_multiplier,
                    numeric_cols=outlier_cols,
                    exclude_cols=[self.config.target_column],
                )
                test_df = DataCleaning.handle_outliers(
                    test_df,
                    strategy=self.config.outlier_strategy,
                    lower_q=self.config.outlier_lower_q,
                    upper_q=self.config.outlier_upper_q,
                    multiplier=self.config.outlier_multiplier,
                    numeric_cols=outlier_cols,
                    exclude_cols=[self.config.target_column],
                )
            else:
                logging.info("\nStep 8: Outlier handling disabled (skipped)")

            # Step 9: EDA
            if self.config.enable_eda:
                logging.info("\nStep 9: Performing comprehensive EDA...")
                self.eda.run(train_df, df_name="train")
                self.eda.run(test_df, df_name="test")
            else:
                logging.info("\nStep 9: EDA disabled (skipped)")

            # Step 10: VIF multicollinearity check
            logging.info("\nStep 10: Checking multicollinearity (VIF)...")
            numeric_cols = self._prune_by_vif(train_df, numeric_cols)

            # Step 11: Validate predictors
            logging.info("\nStep 11: Validating predictor availability...")
            all_predictors = numeric_cols + cat_cols
            missing_predictors = [c for c in all_predictors if c not in train_df.columns]
            if missing_predictors:
                raise ValueError(f"Missing predictors in train: {missing_predictors}")

            logging.info("  Final predictor count: %s", len(all_predictors))
            logging.info("    Numeric: %s", numeric_cols)
            logging.info("    Categorical: %s", cat_cols)

            # Step 12: Split X and y
            logging.info("\nStep 12: Splitting features and target...")
            target = self.config.target_column
            X_train = train_df[all_predictors].copy()
            y_train = train_df[target].to_numpy()

            X_test = test_df[all_predictors].copy()
            y_test = test_df[target].to_numpy() if target in test_df.columns else None

            if y_test is None and self.config.require_target_in_test:
                raise ValueError(f"Target '{target}' missing in test but required.")

            logging.info("  X_train: %s", X_train.shape)
            logging.info("  y_train: %s", y_train.shape)
            logging.info("  X_test: %s", X_test.shape)
            if y_test is not None:
                logging.info("  y_test: %s", y_test.shape)

            # Step 13: Build and fit preprocessor
            logging.info("\nStep 13: Building preprocessing pipeline...")
            preprocessor = self._build_preprocessor(numeric_cols, cat_cols)
            
            logging.info("Fitting preprocessor on training data...")
            X_train_processed = preprocessor.fit_transform(X_train)
            logging.info("  Transformed train shape: %s", X_train_processed.shape)
            
            logging.info("Transforming test data...")
            X_test_processed = preprocessor.transform(X_test)
            logging.info("  Transformed test shape: %s", X_test_processed.shape)

            # Step 14: Densify if needed
            if self.config.force_dense_output:
                if hasattr(X_train_processed, "toarray"):
                    X_train_processed = X_train_processed.toarray()
                    logging.info("  Converted train to dense array")
                if hasattr(X_test_processed, "toarray"):
                    X_test_processed = X_test_processed.toarray()
                    logging.info("  Converted test to dense array")

            # Step 15: Concatenate X and y
            logging.info("\nStep 15: Concatenating features and target...")
            train_arr = np.c_[X_train_processed, y_train]
            test_arr = (
                np.c_[X_test_processed, y_test] 
                if y_test is not None 
                else X_test_processed
            )
            logging.info("  Final train array: %s", train_arr.shape)
            logging.info("  Final test array: %s", test_arr.shape)

            # Step 16: Save artifacts
            logging.info("\nStep 16: Saving artifacts...")
            self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(preprocessor, self.config.preprocessor_obj_file_path)
            logging.info("  ✓ Saved preprocessor: %s", self.config.preprocessor_obj_file_path)
            
            np.save(self.config.transformed_train_file_path, train_arr)
            logging.info("  ✓ Saved train array: %s", self.config.transformed_train_file_path)
            
            np.save(self.config.transformed_test_file_path, test_arr)
            logging.info("  ✓ Saved test array: %s", self.config.transformed_test_file_path)

            # Step 17: Generate summary report
            logging.info("\nStep 17: Generating transformation summary...")
            self._generate_transformation_summary(
                train_df, test_df, numeric_cols, cat_cols, 
                X_train_processed.shape[1]
            )

            logging.info("\n" + "=" * 80)
            logging.info("TRANSFORMATION PIPELINE COMPLETE")
            logging.info("=" * 80)
            logging.info("All artifacts saved to: %s", str(self.config.artifacts_dir.resolve()))

            return train_arr, test_arr, str(self.config.preprocessor_obj_file_path)

        except Exception as e:
            logging.exception("Data transformation pipeline failed")
            raise CustomException(e, sys)

    def _generate_transformation_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numeric_cols: List[str],
        cat_cols: List[str],
        final_feature_count: int
    ) -> None:
        """
        Generate comprehensive transformation summary report.
        
        Args:
            train_df: Training DataFrame (after transformation)
            test_df: Test DataFrame (after transformation)
            numeric_cols: Final numeric column list
            cat_cols: Final categorical column list
            final_feature_count: Total features after preprocessing
        """
        summary_lines = [
            "=" * 80,
            "DATA TRANSFORMATION - PIPELINE SUMMARY",
            "=" * 80,
            "",
            "INPUT DATA",
            "-" * 80,
            f"Train path: {self.config.train_data_path}",
            f"Test path: {self.config.test_data_path}",
            f"Train shape: {train_df.shape}",
            f"Test shape: {test_df.shape}",
            "",
            "CONFIGURATION",
            "-" * 80,
            f"Target column: {self.config.target_column}",
            f"Cleaning enabled: {self.config.enable_cleaning}",
            f"Feature engineering enabled: {self.config.enable_feature_engineering}",
            f"EDA enabled: {self.config.enable_eda}",
            f"VIF enabled: {self.config.enable_vif}",
            f"Missing value strategy: {self.config.missing_value_strategy}",
            f"Outlier strategy: {self.config.outlier_strategy or 'None'}",
            "",
            "FEATURE SELECTION",
            "-" * 80,
            f"Numeric features: {len(numeric_cols)}",
        ]
        
        if numeric_cols:
            summary_lines.append("  " + ", ".join(numeric_cols[:10]))
            if len(numeric_cols) > 10:
                summary_lines.append(f"  ... and {len(numeric_cols) - 10} more")
        
        summary_lines.extend([
            f"Categorical features: {len(cat_cols)}",
        ])
        
        if cat_cols:
            summary_lines.append("  " + ", ".join(cat_cols[:10]))
            if len(cat_cols) > 10:
                summary_lines.append(f"  ... and {len(cat_cols) - 10} more")
        
        summary_lines.extend([
            "",
            "PREPROCESSING PIPELINE",
            "-" * 80,
            "Numeric pipeline:",
            "  1. SimpleImputer(strategy='median')",
            "  2. StandardScaler()",
            "",
            "Categorical pipeline:",
            "  1. SimpleImputer(strategy='most_frequent')",
            "  2. OneHotEncoder(handle_unknown='ignore')",
            "",
            "OUTPUT",
            "-" * 80,
            f"Final feature count (after one-hot): {final_feature_count}",
            f"Train array: {self.config.transformed_train_file_path}",
            f"Test array: {self.config.transformed_test_file_path}",
            f"Preprocessor: {self.config.preprocessor_obj_file_path}",
            "",
            "EDA ARTIFACTS",
            "-" * 80,
            f"EDA directory: {self.config.eda_artifacts_dir}",
            f"  Distributions: {self.config.eda_distributions_dir}",
            f"  Relationships: {self.config.eda_relationships_dir}",
            f"  Outliers: {self.config.eda_outliers_dir}",
            f"  Correlations: {self.config.eda_correlations_dir}",
            "",
            "=" * 80,
            "TRANSFORMATION COMPLETE - READY FOR MODEL TRAINING",
            "=" * 80,
        ])

        summary_path = self.config.eda_summary_report_path
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        logging.info("✓ Saved transformation summary: %s", summary_path)


# ---------------------------------------------------------------------
# Script Entrypoint
# ---------------------------------------------------------------------
def _run_as_script() -> None:
    """
    Execute transformation pipeline as standalone script.
    
    Assumes data ingestion has already created train.csv and test.csv.
    Runs complete pipeline and reports results.
    """
    logging.info("\n" + "=" * 80)
    logging.info("RUNNING DATA TRANSFORMATION AS SCRIPT")
    logging.info("=" * 80)
    
    try:
        cfg = DataTransformationConfig()
        logging.info("Configuration loaded:")
        logging.info("  Train path: %s", cfg.train_data_path)
        logging.info("  Test path: %s", cfg.test_data_path)

        # Validate inputs exist
        if not cfg.train_data_path.exists() or not cfg.test_data_path.exists():
            raise FileNotFoundError(
                "Input files not found. Please run data ingestion first.\n"
                f"  Expected train: {cfg.train_data_path}\n"
                f"  Expected test: {cfg.test_data_path}"
            )

        # Run transformation
        transformer = DataTransformation(cfg)
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path=str(cfg.train_data_path),
            test_path=str(cfg.test_data_path),
        )

        # Report results
        print("\n" + "=" * 80)
        print("✅ TRANSFORMATION COMPLETE")
        print("=" * 80)
        print(f"Train array shape: {train_arr.shape}")
        print(f"Test array shape:  {test_arr.shape}")
        print(f"Preprocessor saved: {preprocessor_path}")
        print(f"\nArtifacts location: {cfg.artifacts_dir.resolve()}")
        print(f"EDA location: {cfg.eda_artifacts_dir.resolve()}")
        print("=" * 80)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    _run_as_script()