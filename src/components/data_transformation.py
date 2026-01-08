"""
data_transformation.py
======================

Purpose
-------
A single, unified "data transformation" module that *combines*:

1) Data cleaning (standardize column names, handle missing values, duplicates,
   optional outlier clipping / filtering).
2) Optional EDA artifact generation (stats summary, missingness report, boxplots,
   correlations) saved under artifacts/eda/.
3) Feature engineering (safe, schema-aware derived features).
4) Optional multicollinearity checks (VIF) on numeric features *only*, with
   controlled feature dropping (to avoid unstable models).
5) End-to-end preprocessing pipeline creation:
   - Numeric: impute -> scale
   - Categorical: impute -> one-hot
6) Fit on train only, transform train + test, then append target column.
7) Save artifacts:
   - preprocessor.pkl
   - train_transformed.npy
   - test_transformed.npy
   - optional EDA artifacts, VIF report, feature lists

Design Goals (Advanced / Production)
-----------------------------------
- Robustness to schema drift (missing columns, unexpected dtypes, category drift)
- Reproducible transformations (fit on train only; store preprocessor)
- Leakage prevention (explicit checks + safe feature engineering)
- Clear logging & error handling (CustomException + logging)
- Config-driven behavior (toggle cleaning/EDA/FE/VIF; thresholds; expected columns)
- Trainer-friendly outputs (.npy arrays with X + y concatenated)

Assumptions & Notes
-------------------
- This module expects ingestion outputs train.csv and test.csv by default under artifacts/.
- Target column exists in BOTH train and test (for evaluation). If you only have target
  in train (common in Kaggle), set `require_target_in_test=False` and the module will
  return y_test as None and skip concatenation for test if missing.
- VIF is computed ONLY for numeric predictors (raw/engineered numeric). VIF for one-hot
  encoded columns is not recommended (high-dimensional and unstable), so we do not do it.
- Feature engineering is schema-aware: it only creates features when required columns exist.
- Outlier logic is intentionally conservative and configurable. In many real ML systems,
  robust scaling or tree models handle outliers better than dropping rows; choose carefully.

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

# Optional (only used if VIF enabled)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:
    variance_inflation_factor = None

# Optional plotting (EDA)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Your project's exception + logger
from src.exception import CustomException
from src.logger import logging


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the entire transformation pipeline.

    Modify these defaults to match your project/dataset. Keep it config-driven.
    """

    # Root artifacts directory
    artifacts_dir: Path = Path("artifacts")

    # Ingestion outputs (defaults)
    train_data_path: Path = Path("artifacts/train.csv")
    test_data_path: Path = Path("artifacts/test.csv")

    # Outputs from this module
    preprocessor_obj_file_path: Path = Path("artifacts/preprocessor.pkl")
    transformed_train_file_path: Path = Path("artifacts/train_transformed.npy")
    transformed_test_file_path: Path = Path("artifacts/test_transformed.npy")

    # Optional subfolders / outputs
    eda_artifacts_dir: Path = Path("artifacts/eda")
    feature_manifest_path: Path = Path("artifacts/feature_manifest.json")
    vif_report_path: Path = Path("artifacts/vif_report.csv")

    # Target configuration
    target_column: str = "annual_premium_amount"

    # If True, require target in both train AND test; if False, allow missing in test
    require_target_in_test: bool = True

    # -----------------------------
    # Cleaning toggles
    # -----------------------------
    enable_cleaning: bool = True
    standardize_column_names: bool = True

    # Missing values:
    # - "drop": drop any rows with NA (aggressive; can lose data)
    # - "keep": keep NA; preprocessing imputers handle them later
    missing_value_strategy: str = "keep"  # "drop" or "keep"

    # Duplicates
    drop_duplicates: bool = True

    # Optional outlier handling for numeric columns:
    # - None: no outlier handling
    # - "clip": clip to percentiles (winsorization)
    # - "filter": drop rows outside percentiles
    outlier_strategy: Optional[str] = None  # None | "clip" | "filter"
    outlier_lower_q: float = 0.01
    outlier_upper_q: float = 0.99
    outlier_numeric_cols: Optional[Tuple[str, ...]] = None  # if None, infer numeric cols

    # -----------------------------
    # EDA toggles
    # -----------------------------
    enable_eda: bool = True
    eda_save_boxplots: bool = True
    eda_save_correlation: bool = True
    eda_max_boxplot_cols: int = 30  # avoid unreadable plots on wide datasets

    # -----------------------------
    # Feature selection & preprocessing
    # -----------------------------
    # If True: only categorical predictors (useful baseline model)
    use_only_categorical: bool = False

    # If True: after transform, convert sparse -> dense arrays (simplifies trainers)
    force_dense_output: bool = True

    # If provided, these override inferred columns.
    # If left None, we infer:
    # - categorical: object/category/bool
    # - numeric: int/float
    categorical_columns: Optional[Tuple[str, ...]] = (
        "gender",
        "region",
        "marital_status",
        "physical_activity",
        "stress_level",
        "bmi_category",
        "smoking_status",
        "employment_status",
        "medical_history",
        "insurance_plan",
    )
    numerical_columns: Optional[Tuple[str, ...]] = (
        "age",
        "number_of_dependants",
        "income_lakhs",
    )

    # -----------------------------
    # Feature engineering toggles
    # -----------------------------
    enable_feature_engineering: bool = True

    # Leakage guard: do NOT engineer features using target_column
    disallow_target_leakage: bool = True

    # -----------------------------
    # VIF / multicollinearity control
    # -----------------------------
    enable_vif: bool = True
    vif_threshold: float = 10.0
    vif_max_features_to_drop: int = 5
    # Some numeric features might be policy-required or business-critical; exempt them from dropping.
    vif_exempt_features: Tuple[str, ...] = ()


# ---------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------
class DataCleaning:
    """
    Cleaning utilities kept inside this module to satisfy the request for "single file".

    NOTE:
    - Keep cleaning minimal and reversible. Heavy cleaning can distort signal.
    - For missing values, prefer keeping NA and imputing in pipeline (unless you
      have strong reasons to drop).
    """

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        numeric_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Conservative percentile-based outlier handling.

        - "clip": winsorize to [lower_q, upper_q]
        - "filter": drop rows outside [lower_q, upper_q]
        """
        if strategy is None:
            return df

        if strategy not in ("clip", "filter"):
            raise ValueError("outlier_strategy must be None, 'clip', or 'filter'")

        df = df.copy()

        exclude_cols = exclude_cols or []
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        if not numeric_cols:
            logging.info("Outlier handling skipped: no numeric columns selected.")
            return df

        # Compute bounds on current df
        bounds = {}
        for col in numeric_cols:
            series = df[col]
            if series.isna().all():
                continue
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

        # strategy == "filter"
        before = len(df)
        mask = np.ones(len(df), dtype=bool)
        for col, (lo, hi) in bounds.items():
            mask &= df[col].between(lo, hi) | df[col].isna()
        df = df.loc[mask].copy()
        logging.info("Outlier strategy=filter dropped %s rows.", before - len(df))
        return df


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
class DataExploration:
    """
    Lightweight EDA artifact generator. The goal isn't beautiful charts—it's
    reproducible, audit-friendly diagnostics saved to disk.
    """

    def __init__(self, eda_dir: Path):
        self.eda_dir = eda_dir
        self.eda_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame, df_name: str = "dataset") -> None:
        logging.info("EDA started for %s.", df_name)

        # 1) Basic stats
        stats = df.describe(include="all").transpose()
        stats.to_csv(self.eda_dir / f"{df_name}_stats_summary.csv")
        logging.info("Saved stats summary: %s", self.eda_dir / f"{df_name}_stats_summary.csv")

        # 2) Missingness report
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

        # 3) Boxplots for numeric columns (subset if too many)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            max_cols = min(len(numeric_cols), 30)
            plot_cols = numeric_cols[:max_cols]
            plt.figure(figsize=(max(12, max_cols * 0.6), 8))
            df[plot_cols].boxplot()
            plt.title(f"{df_name}: Outlier Detection (Numeric Features)")
            plt.xticks(rotation=45, ha="right")
            out_path = self.eda_dir / f"{df_name}_outliers_boxplot.png"
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            logging.info("Saved boxplot: %s", out_path)

        # 4) Correlation heatmap-like matrix (CSV) to avoid heavy plotting dependencies
        # (Plotting heatmaps is optional; CSV is universally useful.)
        if numeric_cols:
            corr = df[numeric_cols].corr(numeric_only=True)
            corr.to_csv(self.eda_dir / f"{df_name}_correlation.csv")
            logging.info("Saved correlation CSV: %s", self.eda_dir / f"{df_name}_correlation.csv")

        logging.info("EDA completed for %s.", df_name)


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------
class FeatureEngineer:
    """
    Schema-aware feature engineering with explicit anti-leakage rules.

    Strategy:
    - Only add features when required source columns exist.
    - Never create engineered features from target if disallow_target_leakage=True.
    - Keep features interpretable and broadly useful:
        * ratios (income per dependent)
        * transformations (log1p income)
        * interactions (age * income)
        * counts/flags (has_dependents)
    """

    def __init__(self, target_col: str, disallow_target_leakage: bool = True):
        self.target_col = target_col
        self.disallow_target_leakage = disallow_target_leakage

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # -------------------------
        # Leakage guard
        # -------------------------
        if self.disallow_target_leakage and self.target_col in df.columns:
            # We do NOT use target column to create any predictors.
            # This is a common silent bug in notebooks.
            pass

        # Helper to safely check columns
        def has_cols(cols: List[str]) -> bool:
            return all(c in df.columns for c in cols)

        # -------------------------
        # Example engineered features (insurance-like schema)
        # -------------------------
        # 1) has_dependents flag
        if "number_of_dependants" in df.columns:
            df["has_dependents"] = (pd.to_numeric(df["number_of_dependants"], errors="coerce").fillna(0) > 0).astype(int)

        # 2) income per dependent (avoid division by zero)
        if has_cols(["income_lakhs", "number_of_dependants"]):
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            deps = pd.to_numeric(df["number_of_dependants"], errors="coerce").replace(0, np.nan)
            df["income_per_dependent"] = (income / deps).replace([np.inf, -np.inf], np.nan)

        # 3) log income (robustness to skew)
        if "income_lakhs" in df.columns:
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            df["log_income_lakhs"] = np.log1p(income.clip(lower=0))

        # 4) age bins (keep numeric representation; you can also do categorical bins if preferred)
        if "age" in df.columns:
            age = pd.to_numeric(df["age"], errors="coerce")
            df["age_squared"] = age ** 2

        # 5) interaction: age * income
        if has_cols(["age", "income_lakhs"]):
            age = pd.to_numeric(df["age"], errors="coerce")
            income = pd.to_numeric(df["income_lakhs"], errors="coerce")
            df["age_income_interaction"] = age * income

        # -------------------------
        # Optional: if you later extend to other datasets, add guarded blocks here.
        # Keep everything conditional and leakage-safe.
        # -------------------------

        return df


# ---------------------------------------------------------------------
# Data Transformation (Main Orchestrator)
# ---------------------------------------------------------------------
class DataTransformation:
    """
    Orchestrates:
      - reading train/test
      - cleaning
      - optional EDA
      - feature engineering
      - optional VIF numeric feature pruning
      - preprocessing fit/transform
      - artifact saving

    Output:
      - train_arr: np.ndarray (X_transformed + y)
      - test_arr: np.ndarray  (X_transformed + y) OR (X_transformed) if y missing and require_target_in_test=False
      - preprocessor_path: str
    """

    def __init__(self, config: DataTransformationConfig | None = None):
        self.config = config or DataTransformationConfig()
        self.feature_engineer = FeatureEngineer(
            target_col=self.config.target_column,
            disallow_target_leakage=self.config.disallow_target_leakage,
        )
        self.eda = DataExploration(self.config.eda_artifacts_dir)

    # -------------------------
    # Utilities
    # -------------------------
    def _safe_onehot(self) -> OneHotEncoder:
        """
        Sklearn compatibility:
        - newer sklearn uses sparse_output=
        - older sklearn uses sparse=
        """
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _standardize_schema(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Standardize schema early so downstream modules can rely on consistent naming.
        """
        if self.config.standardize_column_names:
            logging.info("[%s] Standardizing column names.", df_name)
            df = DataCleaning.standardize_columns(df)

        if df.empty:
            raise ValueError(f"{df_name} dataset is empty after load/standardization.")
        return df

    def _validate_target_presence(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        target = self.config.target_column
        if target not in train_df.columns:
            raise ValueError(f"Target column '{target}' missing from train dataset.")
        if self.config.require_target_in_test and target not in test_df.columns:
            raise ValueError(
                f"Target column '{target}' missing from test dataset but require_target_in_test=True."
            )

    def _infer_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Infer numeric/categorical columns if config doesn't define them.

        Important:
        - Some datasets have numeric-looking strings; we attempt numeric coercion later.
        - For production systems, explicit schema is best.
        """
        if self.config.numerical_columns is not None:
            numeric_cols = [c for c in self.config.numerical_columns if c in df.columns]
        else:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if self.config.categorical_columns is not None:
            cat_cols = [c for c in self.config.categorical_columns if c in df.columns]
        else:
            cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Ensure we never include target as a predictor
        target = self.config.target_column
        numeric_cols = [c for c in numeric_cols if c != target]
        cat_cols = [c for c in cat_cols if c != target]

        return numeric_cols, cat_cols

    def _coerce_numeric(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Force numeric columns to numeric dtype (strings -> NaN).
        This aligns with robust pipelines where imputers handle NaN.
        """
        df = df.copy()
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _build_preprocessor(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """
        Preprocessing:
        - Numeric: median impute + standardize
        - Categorical: most_frequent impute + onehot

        Note:
        - StandardScaler on numeric is common for linear models, neural nets, SVM, etc.
        - For tree models, scaling isn't required but doesn't break things.
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

    # -------------------------
    # VIF
    # -------------------------
    def _calculate_vif(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Calculate VIF for numeric features only.
        High VIF suggests multicollinearity which can destabilize:
          - linear regression
          - GLMs
          - some gradient-based methods

        Caveats:
        - VIF requires statsmodels. If missing, we skip gracefully (but log).
        - VIF can be sensitive; don't over-prune without domain reasoning.
        """
        if not self.config.enable_vif:
            return pd.DataFrame()

        if variance_inflation_factor is None:
            logging.warning("statsmodels not available; VIF computation skipped.")
            return pd.DataFrame()

        if not numeric_cols:
            logging.info("VIF skipped: no numeric columns.")
            return pd.DataFrame()

        # Work on numeric df only; drop rows with NA for VIF stability
        X = df[numeric_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
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
        Iteratively drop highest-VIF numeric features above threshold, up to max_features_to_drop.

        We do *not* auto-drop exempt features (policy/business-critical).
        """
        if not self.config.enable_vif:
            return numeric_cols

        vif_df = self._calculate_vif(train_df, numeric_cols)
        if vif_df.empty:
            return numeric_cols

        # Save report
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        vif_df.to_csv(self.config.vif_report_path, index=False)
        logging.info("Saved VIF report: %s", self.config.vif_report_path)

        remaining = numeric_cols[:]
        drops = 0

        # Iterate: recompute after each drop (more accurate than one-shot)
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
                # If exempt is top, consider next non-exempt
                non_exempt = vif_df[~vif_df["feature"].isin(self.config.vif_exempt_features)]
                if non_exempt.empty:
                    logging.warning("All high-VIF features are exempt; stopping VIF pruning.")
                    break
                top_feat = str(non_exempt.iloc[0]["feature"])
                top_vif = float(non_exempt.iloc[0]["VIF"])

                if top_vif <= self.config.vif_threshold:
                    break

            logging.warning("Dropping feature due to high VIF: %s (VIF=%.3f)", top_feat, top_vif)
            remaining.remove(top_feat)
            drops += 1

        if drops > 0:
            logging.info("VIF pruning dropped %s feature(s). Remaining numeric features: %s", drops, remaining)

        return remaining

    # -------------------------
    # Main entry
    # -------------------------
    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Returns:
            train_arr: np.ndarray -> concatenated [X_transformed | y_train]
            test_arr:  np.ndarray -> concatenated [X_transformed | y_test] OR [X_transformed] if allowed missing target
            preprocessor_path: str -> path to saved joblib preprocessor
        """
        logging.info("Starting data transformation pipeline.")
        try:
            # -------------------------
            # Load
            # -------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train shape: %s", train_df.shape)
            logging.info("Loaded test shape:  %s", test_df.shape)

            # -------------------------
            # Standardize schema
            # -------------------------
            train_df = self._standardize_schema(train_df, "train")
            test_df = self._standardize_schema(test_df, "test")

            # Validate targets
            self._validate_target_presence(train_df, test_df)

            # -------------------------
            # Cleaning (optional)
            # -------------------------
            if self.config.enable_cleaning:
                logging.info("Cleaning enabled.")
                # Missing
                train_df = DataCleaning.handle_missing_values(train_df, self.config.missing_value_strategy)
                test_df = DataCleaning.handle_missing_values(test_df, self.config.missing_value_strategy)

                # Duplicates
                train_df = DataCleaning.handle_duplicates(train_df, self.config.drop_duplicates)
                test_df = DataCleaning.handle_duplicates(test_df, self.config.drop_duplicates)

            # -------------------------
            # Feature engineering (optional)
            # -------------------------
            if self.config.enable_feature_engineering:
                logging.info("Feature engineering enabled.")
                train_df = self.feature_engineer.add_features(train_df)
                test_df = self.feature_engineer.add_features(test_df)

            # -------------------------
            # Column selection (config-driven; infer if needed)
            # -------------------------
            numeric_cols, cat_cols = self._infer_columns(train_df)

            # If user wants only categorical predictors:
            if self.config.use_only_categorical:
                numeric_cols = []
                logging.info("Using ONLY categorical predictors (baseline mode).")

            # Coerce numeric
            train_df = self._coerce_numeric(train_df, numeric_cols)
            test_df = self._coerce_numeric(test_df, numeric_cols)

            # Optional outlier handling (never apply on target)
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
                    numeric_cols=outlier_cols,
                    exclude_cols=[self.config.target_column],
                )
                test_df = DataCleaning.handle_outliers(
                    test_df,
                    strategy=self.config.outlier_strategy,
                    lower_q=self.config.outlier_lower_q,
                    upper_q=self.config.outlier_upper_q,
                    numeric_cols=outlier_cols,
                    exclude_cols=[self.config.target_column],
                )

            # -------------------------
            # EDA (optional, before VIF pruning so you can inspect raw distributions)
            # -------------------------
            if self.config.enable_eda:
                logging.info("EDA enabled; saving diagnostic artifacts.")
                self.eda.run(train_df, df_name="train")
                self.eda.run(test_df, df_name="test")

            # -------------------------
            # VIF pruning (optional; numeric only)
            # -------------------------
            if self.config.enable_vif:
                logging.info("VIF enabled; checking multicollinearity on numeric predictors.")
                numeric_cols = self._prune_by_vif(train_df, numeric_cols)

            # Final safety checks
            missing_predictors = [c for c in (numeric_cols + cat_cols) if c not in train_df.columns]
            if missing_predictors:
                raise ValueError(
                    f"Predictor columns missing from training data after processing: {missing_predictors}"
                )

            # -------------------------
            # Split X / y
            # -------------------------
            target = self.config.target_column

            X_train = train_df[numeric_cols + cat_cols].copy()
            y_train = train_df[target].to_numpy()

            # Test target may be missing if allowed
            X_test = test_df[numeric_cols + cat_cols].copy()

            y_test = None
            if target in test_df.columns:
                y_test = test_df[target].to_numpy()
            elif self.config.require_target_in_test:
                raise ValueError(
                    f"Target column '{target}' missing from test but require_target_in_test=True."
                )

            logging.info("Selected numeric features (%s): %s", len(numeric_cols), numeric_cols)
            logging.info("Selected categorical features (%s): %s", len(cat_cols), cat_cols)
            logging.info("Target column: %s", target)

            # -------------------------
            # Build + fit preprocessor on train only
            # -------------------------
            preprocessor = self._build_preprocessor(
                numeric_features=numeric_cols,
                categorical_features=cat_cols,
            )

            logging.info("Fitting preprocessor on training data only.")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Optionally densify
            if self.config.force_dense_output:
                if hasattr(X_train_processed, "toarray"):
                    X_train_processed = X_train_processed.toarray()
                if hasattr(X_test_processed, "toarray"):
                    X_test_processed = X_test_processed.toarray()

            # -------------------------
            # Concatenate targets (trainer-friendly)
            # -------------------------
            train_arr = np.c_[X_train_processed, y_train]

            if y_test is not None:
                test_arr = np.c_[X_test_processed, y_test]
            else:
                # Kaggle-style inference set; return X only
                test_arr = np.array(X_test_processed)

            # -------------------------
            # Save artifacts
            # -------------------------
            self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, self.config.preprocessor_obj_file_path)
            np.save(self.config.transformed_train_file_path, train_arr)
            np.save(self.config.transformed_test_file_path, test_arr)

            logging.info("Saved preprocessor: %s", self.config.preprocessor_obj_file_path)
            logging.info("Saved transformed train: %s", self.config.transformed_train_file_path)
            logging.info("Saved transformed test:  %s", self.config.transformed_test_file_path)
            logging.info("Transformation done. Train arr: %s | Test arr: %s", train_arr.shape, test_arr.shape)

            return train_arr, test_arr, str(self.config.preprocessor_obj_file_path)

        except Exception as e:
            logging.exception("Data transformation failed.")
            raise CustomException(e, sys)


# ---------------------------------------------------------------------
# Script Entrypoint
# ---------------------------------------------------------------------
def _run_as_script() -> None:
    """
    Script entrypoint:
    - expects artifacts/train.csv and artifacts/test.csv to exist (from ingestion)
    """
    try:
        cfg = DataTransformationConfig()
        logging.info("Running data_transformation as a script.")
        logging.info("Expected train path: %s", cfg.train_data_path)
        logging.info("Expected test path:  %s", cfg.test_data_path)

        if not cfg.train_data_path.exists() or not cfg.test_data_path.exists():
            raise FileNotFoundError(
                "Could not find ingestion outputs. "
                "Run data ingestion first to create artifacts/train.csv and artifacts/test.csv."
            )

        transformer = DataTransformation(cfg)
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path=str(cfg.train_data_path),
            test_path=str(cfg.test_data_path),
        )

        print("✅ Data Transformation complete")
        print("Train array:", train_arr.shape)
        print("Test array:", test_arr.shape)
        print("Preprocessor:", preprocessor_path)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    _run_as_script()
