"""
data_ingestion.py
=================

Purpose
-------
This module handles the initial data ingestion process for the insurance premium prediction pipeline.
It loads raw data from a specified input path, performs a train-test split, and saves the splits
along with a raw copy to an artifacts directory. This ensures reproducibility and separation of
concerns in the ML pipeline.

Design Goals (Advanced / Production)
------------------------------------
- Config-driven: Uses a frozen dataclass for paths and settings to allow easy overrides without code changes.
- Robustness: Checks for file existence, handles empty datasets, and uses logging for traceability.
- Leakage Prevention: Splits data early to avoid any downstream leakage.
- Explainability: Detailed logging and exceptions provide clear failure reasons.
- Advanced: Optional stratification can be added via config if target is categorical (not enabled by default here).

Assumptions & Notes
-------------------
- Input data is a CSV file with headers.
- Split is random (80/20) with fixed seed for reproducibility. For imbalanced targets, consider stratifying.
- Artifacts directory is created if missing.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion. Frozen to prevent accidental modifications.
    
    Attributes:
        artifacts_dir: Root directory for saving outputs.
        train_data_path: Path to save train split.
        test_data_path: Path to save test split.
        raw_data_path: Path to save raw data copy.
        input_data_path: Source data file (change as needed).
        test_size: Proportion for test split (default 0.2).
        random_state: Seed for reproducibility.
        stratify: Optional column to stratify split (e.g., for imbalanced classes; None disables).
    """
    artifacts_dir: Path = Path("artifacts")
    train_data_path: Path = Path("artifacts/train.csv")
    test_data_path: Path = Path("artifacts/test.csv")
    raw_data_path: Path = Path("artifacts/data.csv")
    input_data_path: Path = Path("data/premiums_with_life_style.csv")  # Change as needed
    test_size: float = 0.2
    random_state: int = 42
    stratify: str | None = None  # e.g., "target_column" for stratified split


class DataIngestion:
    """
    Orchestrates data ingestion: load, validate, split, and save.
    
    This class ensures the pipeline starts with clean, split data while logging each step for auditing.
    """
    def __init__(self, config: DataIngestionConfig | None = None):
        """
        Initializes with config. Defaults to DataIngestionConfig if None provided.
        """
        self.ingestion_config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Performs data ingestion and returns paths to train/test files.
        
        Steps:
        1. Load input CSV.
        2. Validate non-empty.
        3. Save raw copy.
        4. Split into train/test (with optional stratification).
        5. Save splits.
        
        Returns:
            Tuple of (train_path_str, test_path_str).
        
        Raises:
            FileNotFoundError: If input file missing.
            ValueError: If dataset empty.
            CustomException: For other failures (wrapped with sys info).
        """
        logging.info("Starting data ingestion.")
        try:
            input_path = self.ingestion_config.input_data_path

            # Check existence (robustness to missing files)
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input data file not found at: {input_path.resolve()}"
                )

            # Load data
            df = pd.read_csv(input_path)
            logging.info("Dataset loaded. Shape: %s", df.shape)

            # Validate non-empty
            if df.empty:
                raise ValueError("Input dataset is empty.")

            # Create artifacts dir if needed
            self.ingestion_config.artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save raw copy for reproducibility
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to: %s", self.ingestion_config.raw_data_path)

            # Prepare stratification if enabled
            stratify_col = None
            if self.ingestion_config.stratify and self.ingestion_config.stratify in df.columns:
                stratify_col = df[self.ingestion_config.stratify]
                logging.info("Stratified split enabled on column: %s", self.ingestion_config.stratify)

            # Perform split (advanced: uses sklearn for consistency and options)
            train_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=stratify_col  # Prevents imbalance in splits if enabled
            )

            # Save splits with headers
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(
                "Train/Test saved. Train: %s | Test: %s",
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

            return (
                str(self.ingestion_config.train_data_path),
                str(self.ingestion_config.test_data_path),
            )

        except Exception as e:
            logging.exception("Data ingestion failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Train:", train_path)
    print("Test:", test_path)