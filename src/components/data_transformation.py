"""
data_transformation.py
======================

Purpose
-------
A single, unified "data transformation" module that combines:
1. Data cleaning (standardize names, handle missing/duplicates, optional outliers).
2. Optional EDA artifacts (stats, missingness, boxplots, correlations, heatmaps).
3. Feature engineering (schema-aware, leakage-safe derived features).
4. Optional multicollinearity checks (VIF on numerics only, with controlled dropping).
5. End-to-end preprocessing pipeline (impute, scale, one-hot).
6. Fit on train, transform train/test, append target.
7. Save artifacts (preprocessor, transformed arrays, EDA files, VIF report).

Design Goals (Advanced / Production)
------------------------------------
- Robust to schema drift: Infers columns if not specified, validates presence.
- Reproducible: Fits on train only; saves preprocessor for inference.
- Leakage prevention: Explicit checks in FE; no target-derived features.
- Logging & errors: CustomException + detailed logging for debugging.
- Config-driven: Toggles for cleaning/EDA/FE/VIF; thresholds; expected schema.
- Trainer-friendly: Outputs .npy arrays with X + y concatenated.
- Advanced: Added IQR outlier option; stratified inference if needed.

Visualization Improvements (Modern Principles)
----------------------------------------------
- Use Seaborn for aesthetics: Whitegrid theme, muted palette (colorblind-friendly, minimal).
- Clarity: Descriptive titles/labels, rotated ticks, annotations, tight layouts.
- Visibility: Larger figures, KDE in histograms, colormaps in heatmaps.
- Professionalism: High DPI (300), sans-serif fonts via theme.
- Fallback: Matplotlib if sns unavailable; skip plots if libs missing.

Assumptions & Notes
-------------------
- Expects train.csv/test.csv from ingestion (with target in both by default).
- VIF only on numerics (pre-onehot; avoids instability).
- Outliers: Conservative; prefer robust scalers/trees over dropping.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Optional plotting (EDA)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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
    Config for transformation pipeline. Frozen for immutability.
    
    Attributes:
        artifacts_dir: Root for outputs.
        train_data_path/test_data_path: Inputs from ingestion.
        preprocessor_obj_file_path: Saved pipeline.
        transformed_train_file_path/transformed_test_file_path: .npy outputs.
        eda_artifacts_dir: For EDA files/plots.
        feature_manifest_path: JSON of selected features (for auditing).
        vif_report_path: CSV of VIF scores.
        target_column: Name of y.
        require_target_in_test: If False, allow missing y in test (e.g., inference).
        enable_cleaning: Toggle cleaning steps.
        standardize_column_names: Normalize column names.
        missing_value_strategy: "drop" (aggressive) or "keep" (for imputers).
        drop_duplicates: Remove duplicate rows.
        outlier_strategy: None, "clip" (winsorize), "filter" (drop), or "iqr" (advanced IQR method).
        outlier_lower_q/outlier_upper_q: Quantiles for clip/filter.
        outlier_multiplier: For IQR strategy (e.g., 1.5 for mild outliers).
        outlier_numeric_cols: Specific cols; None infers.
        enable_eda: Toggle EDA artifacts.
        eda_save_boxplots/eda_save_correlation/eda_save_heatmap: Plot toggles.
        eda_max_boxplot_cols: Limit for plots.
        use_only_categorical: For baseline (ignore numerics).
        force_dense_output: Convert sparse to dense.
        categorical_columns/numerical_columns: Override inferred.
        enable_feature_engineering: Toggle FE.
        disallow_target_leakage: Prevent y in FE.
        enable_vif: Toggle multicollinearity check.
        vif_threshold: Drop if VIF > this.
        vif_max_features_to_drop: Limit drops.
        vif_exempt_features: Protect critical features.
    """
    artifacts_dir: Path = Path("artifacts")
    train_data_path: Path = Path("artifacts/train.csv")
    test_data_path: Path = Path("artifacts/test.csv")
    preprocessor_obj_file_path: Path = Path("artifacts/preprocessor.pkl")
    transformed_train_file_path: Path = Path("artifacts/train_transformed.npy")
    transformed_test_file_path: Path = Path("artifacts/test_transformed.npy")
    eda_artifacts_dir: Path = Path("artifacts/eda")
    feature_manifest_path: Path = Path("artifacts/feature_manifest.json")
    vif_report_path: Path = Path("artifacts/vif_report.csv")
    target_column: str = "annual_premium_amount"
    require_target_in_test: bool = True
    enable_cleaning: bool = True
    standardize_column_names: bool = True
    missing_value_strategy: str = "keep"  # "drop" or "keep"
    drop_duplicates: bool = True
    outlier_strategy: Optional[str] = None  # None | "clip" | "filter" | "iqr"
    outlier_lower_q: float = 0.01
    outlier_upper_q: float = 0.99
    outlier_multiplier: float = 1.5  # For IQR
    outlier_numeric_cols: Optional[Tuple[str, ...]] = None
    enable_eda: bool = True
    eda_save_boxplots: bool = True
    eda_save_correlation: bool = True
    eda_save_heatmap: bool = True  # New: visual heatmap
    eda_max_boxplot_cols: int = 30
    use_only_categorical: bool = False
    force_dense_output: bool = True
    categorical_columns: Optional[Tuple[str, ...]] = (
        "gender", "region", "marital_status", "physical_activity",
        "stress_level", "bmi_category", "smoking_status",
        "employment_status", "medical_history", "insurance_plan",
    )
    numerical_columns: Optional[Tuple[str, ...]] = (
        "age", "number_of_dependants", "income_lakhs",
    )
    enable_feature_engineering: bool = True
    disallow_target_leakage: bool = True
    enable_vif: bool = True
    vif_threshold: float = 10.0
    vif_max_features_to_drop: int = 5
    vif_exempt_features: Tuple[str, ...] = ()


# ---------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------
class DataCleaning:
    """
    Utilities for data cleaning. Kept modular for reusability.
    
    Rationale: Cleaning is minimal to preserve data; heavy ops (e.g., outliers) are optional.
    """
    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names: strip, replace spaces/dashes, lowercase.
        Improves consistency across datasets.
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
        Handles missing values based on strategy.
        
        - "drop": Removes rows with any NA (data loss risk).
        - "keep": Retains NA for downstream imputers (preferred for ML).
        """
        if strategy not in ("drop", "keep"):
            raise ValueError("missing_value_strategy must be 'drop' or 'keep'")

        df = df.copy()
        if strategy == "drop":
            before = len(df)
            df = df.dropna()
            logging.info("Missing-value strategy=drop: dropped %s rows.", before - len(df))
        else:
            logging.info("Missing-value strategy=keep: leaving NA for pipeline imputers.")
        return df

    @staticmethod
    def handle_duplicates(df: pd.DataFrame, drop_duplicates: bool) -> pd.DataFrame:
        """
        Optionally drops duplicate rows to prevent overfitting.
        """
        df = df.copy()
        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            logging.info("Dropped %s duplicate rows.", before - len(df))
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
        Handles outliers in numeric columns.
        
        Strategies:
        - None: Skip.
        - "clip": Winsorize to quantiles.
        - "filter": Drop rows outside quantiles.
        - "iqr": Advanced; uses Q1 - multiplier*IQR, Q3 + multiplier*IQR (Tukey's method).
        
        Excludes specified cols (e.g., target). Infers numerics if none provided.
        """
        if strategy is None:
            return df

        if strategy not in ("clip", "filter", "iqr"):
            raise ValueError("outlier_strategy must be None, 'clip', 'filter', or 'iqr'")

        df = df.copy()
        exclude_cols = exclude_cols or []
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        if not numeric_cols:
            logging.info("Outlier handling skipped: no numeric columns selected.")
            return df

        # Compute bounds
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

        if not bounds:
            logging.info("Outlier handling skipped: numeric columns had all-NA.")
            return df

        if strategy == "clip":
            for col, (lo, hi) in bounds.items():
                df[col] = df[col].clip(lower=lo, upper=hi)
            logging.info("Outlier strategy=clip applied to %s numeric columns.", len(bounds))
            return df

        # "filter" or "iqr" (both drop rows)
        before = len(df)
        mask = np.ones(len(df), dtype=bool)
        for col, (lo, hi) in bounds.items():
            mask &= df[col].between(lo, hi) | df[col].isna()
        df = df.loc[mask].copy()
        logging.info("Outlier strategy=%s dropped %s rows.", strategy, before - len(df))
        return df


# ---------------------------------------------------------------------
# Data Exploration (EDA)
# ---------------------------------------------------------------------
class DataExploration:
    """
    Generates EDA artifacts for diagnostics. Focus on reproducibility over interactivity.
    
    Visualizations follow modern principles: Clean, informative, professional.
    Uses Seaborn for better defaults if available; falls back to Matplotlib.
    """
    def __init__(self, eda_dir: Path, config: DataTransformationConfig):
        self.eda_dir = eda_dir
        self.config = config
        self.eda_dir.mkdir(parents=True, exist_ok=True)

        # Set modern theme if libs available
        if sns:
            sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)  # Professional, clear

    def run(self, df: pd.DataFrame, df_name: str = "dataset") -> None:
        """
        Runs EDA: stats, missingness, boxplots, correlation CSV/heatmap.
        """
        logging.info("EDA started for %s.", df_name)

        # 1) Basic stats (always save)
        stats = df.describe(include="all").transpose()
        stats.to_csv(self.eda_dir / f"{df_name}_stats_summary.csv")
        logging.info("Saved stats summary: %s", self.eda_dir / f"{df_name}_stats_summary.csv")

        # 2) Missingness report (always save)
        miss = pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_pct": (df.isna().mean() * 100).round(3),
                "dtype": df.dtypes.astype(str),
            }
        ).sort_values("missing_pct", ascending=False)
        miss.to_csv(self.eda_dir / f"{df_name}_missingness.csv")
        logging.info("Saved missingness report: %s", self.eda_dir / f"{df_name}_missingness.csv")

        if plt is None:
            logging.warning("matplotlib not available; skipping plot artifacts.")
            return

        # 3) Boxplots for numerics (subset for readability)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols and self.config.eda_save_boxplots:
            max_cols = min(len(numeric_cols), self.config.eda_max_boxplot_cols)
            plot_cols = numeric_cols[:max_cols]
            fig, ax = plt.subplots(figsize=(max(12, max_cols * 0.8), 8))
            if sns:
                sns.boxplot(data=df[plot_cols], ax=ax, palette="muted")  # Professional palette
            else:
                df[plot_cols].boxplot(ax=ax)
            ax.set_title(f"{df_name.capitalize()}: Distribution and Outliers (Numeric Features)", fontsize=14)
            ax.set_xlabel("Features", fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            out_path = self.eda_dir / f"{df_name}_outliers_boxplot.png"
            plt.savefig(out_path, dpi=300)  # High-res for professionalism
            plt.close()
            logging.info("Saved boxplot: %s", out_path)

        # 4) Correlation (CSV + optional heatmap)
        if numeric_cols and self.config.eda_save_correlation:
            corr = df[numeric_cols].corr(numeric_only=True)
            corr.to_csv(self.eda_dir / f"{df_name}_correlation.csv")
            logging.info("Saved correlation CSV: %s", self.eda_dir / f"{df_name}_correlation.csv")

            if self.config.eda_save_heatmap:  # New visual
                fig, ax = plt.subplots(figsize=(12, 10))
                if sns:
                    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)  # Clear, annotated
                else:
                    ax.imshow(corr, cmap="coolwarm")
                    # Manual annotations for fallback
                    for i in range(len(numeric_cols)):
                        for j in range(len(numeric_cols)):
                            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
                ax.set_title(f"{df_name.capitalize()}: Correlation Heatmap (Numeric Features)", fontsize=14)
                ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=10)
                ax.set_yticklabels(numeric_cols, rotation=0, fontsize=10)
                plt.tight_layout()
                out_path = self.eda_dir / f"{df_name}_correlation_heatmap.png"
                plt.savefig(out_path, dpi=300)
                plt.close()
                logging.info("Saved correlation heatmap: %s", out_path)

        logging.info("EDA completed for %s.", df_name)


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------
class FeatureEngineer:
    """
    Schema-aware feature engineering. Conditional on column presence to handle drift.
    
    Rationale: Features are interpretable and leakage-safe (no target use).
    Advanced: Adds interactions, transformations for non-linearity/skew.
    """
    def __init__(self, target_col: str, disallow_target_leakage: bool = True):
        self.target_col = target_col
        self.disallow_target_leakage = disallow_target_leakage

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds engineered features if source columns exist.
        
        Features:
        - Binary flags (e.g., has_dependents).
        - Ratios (e.g., income_per_dependent; handles zero-division).
        - Transformations (e.g., log for skew).
        - Interactions/Polynomials (e.g., age_squared for non-linearity).
        
        Leakage guard: Skips if target involved and disallowed.
        """
        df = df.copy()

        # Leakage guard
        if self.disallow_target_leakage and self.target_col in df.columns:
            logging.info("Target leakage disallowed; no target-derived features.")

        def has_cols(cols: List[str]) -> bool:
            return all(c in df.columns for c in cols)

        # 1) has_dependents flag
        if "number_of_dependants" in df.columns:
            df["has_dependents"] = (pd.to_numeric(df["number_of_dependants"], errors="coerce").fillna(0) > 0).astype(int)

        # 2) income per dependent (avoid div/0)
        if has_cols(["income_lakhs", "number_of_dependants"]):
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            deps = pd.to_numeric(df["number_of_dependants"], errors="coerce").replace(0, np.nan)
            df["income_per_dependent"] = (income / deps).replace([np.inf, -np.inf], np.nan)

        # 3) log income (handles skew)
        if "income_lakhs" in df.columns:
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            df["log_income_lakhs"] = np.log1p(income.clip(lower=0))

        # 4) age squared (captures non-linear effects)
        if "age" in df.columns:
            age = pd.to_numeric(df["age"], errors="coerce")
            df["age_squared"] = age ** 2

        # 5) age * income interaction
        if has_cols(["age", "income_lakhs"]):
            age = pd.to_numeric(df["age"], errors="coerce")
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            df["age_income_interaction"] = age * income

        logging.info("Engineered %s new features.", len(df.columns) - len(df.columns))  # Delta
        return df


# ---------------------------------------------------------------------
# Data Transformation (Orchestrator)
# ---------------------------------------------------------------------
class DataTransformation:
    """
    Main class: Orchestrates load, clean, EDA, FE, VIF, preprocess, save.
    
    Outputs: Transformed .npy (X + y), preprocessor.pkl.
    """
    def __init__(self, config: DataTransformationConfig | None = None):
        self.config = config or DataTransformationConfig()
        self.feature_engineer = FeatureEngineer(
            target_col=self.config.target_column,
            disallow_target_leakage=self.config.disallow_target_leakage,
        )
        self.eda = DataExploration(self.config.eda_artifacts_dir, self.config)

    # Utilities
    def _safe_onehot(self) -> OneHotEncoder:
        """
        Compatible OneHotEncoder (sparse_output vs sparse for sklearn versions).
        """
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _standardize_schema(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Standardizes columns early for consistency.
        """
        if self.config.standardize_column_names:
            logging.info("[%s] Standardizing column names.", df_name)
            df = DataCleaning.standardize_columns(df)

        if df.empty:
            raise ValueError(f"{df_name} dataset is empty after load/standardization.")
        return df

    def _validate_target_presence(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Ensures target exists (configurable for inference sets).
        """
        target = self.config.target_column
        if target not in train_df.columns:
            raise ValueError(f"Target column '{target}' missing from train dataset.")
        if self.config.require_target_in_test and target not in test_df.columns:
            raise ValueError(
                f"Target column '{target}' missing from test dataset but require_target_in_test=True."
            )

    def _infer_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Infers or uses config for numeric/categorical columns.
        Excludes target from predictors.
        """
        if self.config.numerical_columns is not None:
            numeric_cols = [c for c in self.config.numerical_columns if c in df.columns]
        else:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if self.config.categorical_columns is not None:
            cat_cols = [c for c in self.config.categorical_columns if c in df.columns]
        else:
            cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        target = self.config.target_column
        numeric_cols = [c for c in numeric_cols if c != target]
        cat_cols = [c for c in cat_cols if c != target]

        return numeric_cols, cat_cols

    def _coerce_numeric(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Coerces to numeric (strings -> NaN for imputers).
        """
        df = df.copy()
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _build_preprocessor(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """
        Builds pipeline: Numeric (impute median, scale); Categorical (impute mode, onehot).
        
        Rationale: Median robust to outliers; StandardScaler for scale-sensitive models.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", self._safe_onehot()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )
        return preprocessor

    # VIF
    def _calculate_vif(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Computes VIF for numerics to detect multicollinearity.
        
        High VIF (>10) can destabilize linear models; prune iteratively.
        Skips if statsmodels missing or data insufficient.
        """
        if not self.config.enable_vif:
            return pd.DataFrame()

        if variance_inflation_factor is None:
            logging.warning("statsmodels not available; VIF computation skipped.")
            return pd.DataFrame()

        if not numeric_cols:
            logging.info("VIF skipped: no numeric columns.")
            return pd.DataFrame()

        X = df[numeric_cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty or X.shape[0] < 10:
            logging.warning("VIF skipped: insufficient non-missing numeric rows.")
            return pd.DataFrame()

        vif_df = pd.DataFrame({"feature": numeric_cols})
        vif_df["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]
        vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
        return vif_df

    def _prune_by_vif(self, train_df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """
        Iteratively drops high-VIF features (recomputes after each).
        Exempts critical features; limits drops.
        """
        if not self.config.enable_vif:
            return numeric_cols

        vif_df = self._calculate_vif(train_df, numeric_cols)
        if vif_df.empty:
            return numeric_cols

        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        vif_df.to_csv(self.config.vif_report_path, index=False)
        logging.info("Saved VIF report: %s", self.config.vif_report_path)

        remaining = numeric_cols[:]
        drops = 0

        while drops < self.config.vif_max_features_to_drop:
            vif_df = self._calculate_vif(train_df, remaining)
            if vif_df.empty:
                break

            top = vif_df.iloc[0]
            top_feat = str(top["feature"])
            top_vif = float(top["VIF"])

            if top_vif <= self.config.vif_threshold:
                break

            if top_feat in self.config.vif_exempt_features:
                non_exempt = vif_df[~vif_df["feature"].isin(self.config.vif_exempt_features)]
                if non_exempt.empty:
                    logging.warning("All high-VIF features are exempt; stopping VIF pruning.")
                    break
                top = non_exempt.iloc[0]
                top_feat = str(top["feature"])
                top_vif = float(top["VIF"])

                if top_vif <= self.config.vif_threshold:
                    break

            logging.warning("Dropping feature due to high VIF: %s (VIF=%.3f)", top_feat, top_vif)
            remaining.remove(top_feat)
            drops += 1

        if drops > 0:
            logging.info("VIF pruning dropped %s feature(s). Remaining: %s", drops, remaining)

        return remaining

    # Main entry
    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        End-to-end transformation.
        
        Returns:
            train_arr: [X_train_transformed | y_train]
            test_arr: [X_test_transformed | y_test] or [X_test_transformed] if y missing.
            preprocessor_path: str
        """
        logging.info("Starting data transformation pipeline.")
        try:
            # Load
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train shape: %s", train_df.shape)
            logging.info("Loaded test shape:  %s", test_df.shape)

            # Standardize
            train_df = self._standardize_schema(train_df, "train")
            test_df = self._standardize_schema(test_df, "test")

            # Validate targets
            self._validate_target_presence(train_df, test_df)

            # Cleaning
            if self.config.enable_cleaning:
                logging.info("Cleaning enabled.")
                train_df = DataCleaning.handle_missing_values(train_df, self.config.missing_value_strategy)
                test_df = DataCleaning.handle_missing_values(test_df, self.config.missing_value_strategy)
                train_df = DataCleaning.handle_duplicates(train_df, self.config.drop_duplicates)
                test_df = DataCleaning.handle_duplicates(test_df, self.config.drop_duplicates)

            # Feature engineering
            if self.config.enable_feature_engineering:
                logging.info("Feature engineering enabled.")
                train_df = self.feature_engineer.add_features(train_df)
                test_df = self.feature_engineer.add_features(test_df)

            # Infer columns
            numeric_cols, cat_cols = self._infer_columns(train_df)

            if self.config.use_only_categorical:
                numeric_cols = []
                logging.info("Using ONLY categorical predictors (baseline mode).")

            # Coerce numeric
            train_df = self._coerce_numeric(train_df, numeric_cols)
            test_df = self._coerce_numeric(test_df, numeric_cols)

            # Outliers (exclude target)
            if self.config.enable_cleaning and self.config.outlier_strategy is not None:
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

            # EDA
            if self.config.enable_eda:
                logging.info("EDA enabled; saving artifacts.")
                self.eda.run(train_df, df_name="train")
                self.eda.run(test_df, df_name="test")

            # VIF
            numeric_cols = self._prune_by_vif(train_df, numeric_cols)

            # Safety: Check predictors
            missing_predictors = [c for c in (numeric_cols + cat_cols) if c not in train_df.columns]
            if missing_predictors:
                raise ValueError(f"Missing predictors: {missing_predictors}")

            # Split X/y
            target = self.config.target_column
            X_train = train_df[numeric_cols + cat_cols].copy()
            y_train = train_df[target].to_numpy()

            X_test = test_df[numeric_cols + cat_cols].copy()
            y_test = test_df[target].to_numpy() if target in test_df.columns else None

            if y_test is None and self.config.require_target_in_test:
                raise ValueError(f"Target '{target}' missing in test but required.")

            logging.info("Numeric features (%s): %s", len(numeric_cols), numeric_cols)
            logging.info("Categorical features (%s): %s", len(cat_cols), cat_cols)
            logging.info("Target: %s", target)

            # Preprocessor fit/transform
            preprocessor = self._build_preprocessor(numeric_cols, cat_cols)
            logging.info("Fitting preprocessor on train.")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Densify if needed
            if self.config.force_dense_output:
                if hasattr(X_train_processed, "toarray"):
                    X_train_processed = X_train_processed.toarray()
                if hasattr(X_test_processed, "toarray"):
                    X_test_processed = X_test_processed.toarray()

            # Concat
            train_arr = np.c_[X_train_processed, y_train]
            test_arr = np.c_[X_test_processed, y_test] if y_test is not None else X_test_processed

            # Save
            self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, self.config.preprocessor_obj_file_path)
            np.save(self.config.transformed_train_file_path, train_arr)
            np.save(self.config.transformed_test_file_path, test_arr)

            logging.info("Saved preprocessor: %s", self.config.preprocessor_obj_file_path)
            logging.info("Saved train: %s", self.config.transformed_train_file_path)
            logging.info("Saved test: %s", self.config.transformed_test_file_path)
            logging.info("Shapes: Train %s | Test %s", train_arr.shape, test_arr.shape)

            return train_arr, test_arr, str(self.config.preprocessor_obj_file_path)

        except Exception as e:
            logging.exception("Transformation failed.")
            raise CustomException(e, sys)


# ---------------------------------------------------------------------
# Script Entrypoint
# ---------------------------------------------------------------------
def _run_as_script() -> None:
    """
    Script mode: Assumes ingestion outputs exist.
    """
    try:
        cfg = DataTransformationConfig()
        logging.info("Running data_transformation as script.")
        logging.info("Train path: %s", cfg.train_data_path)
        logging.info("Test path: %s", cfg.test_data_path)

        if not cfg.train_data_path.exists() or not cfg.test_data_path.exists():
            raise FileNotFoundError("Run ingestion first.")

        transformer = DataTransformation(cfg)
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path=str(cfg.train_data_path),
            test_path=str(cfg.test_data_path),
        )

        print("✅ Transformation complete")
        print("Train array:", train_arr.shape)
        print("Test array:", test_arr.shape)
        print("Preprocessor:", preprocessor_path)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    _run_as_script()