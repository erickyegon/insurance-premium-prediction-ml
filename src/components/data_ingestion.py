import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass(frozen=True)
class DataIngestionConfig:
    artifacts_dir: Path = Path("artifacts")
    train_data_path: Path = Path("artifacts/train.csv")
    test_data_path: Path = Path("artifacts/test.csv")
    raw_data_path: Path = Path("artifacts/data.csv")
    input_data_path: Path = Path("data/premiums_with_life_style.csv")  # change as needed


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.ingestion_config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logging.info("Starting data ingestion.")
        try:
            input_path = self.ingestion_config.input_data_path

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input data file not found at: {input_path.resolve()}"
                )

            df = pd.read_csv(input_path)
            logging.info("Dataset loaded. Shape: %s", df.shape)

            if df.empty:
                raise ValueError("Input dataset is empty.")

            # Ensure artifacts directory exists
            self.ingestion_config.artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save raw copy
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to: %s", self.ingestion_config.raw_data_path)

            # Split
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            # Save splits
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
