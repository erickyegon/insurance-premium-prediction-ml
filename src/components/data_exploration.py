from dataclasses import dataclass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.logger import logging
from src.exception import CustomException
from pathlib import Path

@dataclass
class DataExplorationConfig:
    eda_artifacts_dir: Path = Path("artifacts/eda")

class DataExploration:
    def __init__(self, config: DataExplorationConfig | None = None):
        self.config = config or DataExplorationConfig()
        self.config.eda_artifacts_dir.mkdir(parents=True, exist_ok=True)

    def analyze_data(self, df: pd.DataFrame):
        """Generates statistical summaries and saves visualization insights."""
        logging.info("Starting Data Exploration")
        
        # Numeric analysis (Ref: Notebook cell 13)
        stats = df.describe()
        stats.to_csv(self.config.eda_artifacts_dir / "stats_summary.csv")
        
        # Outlier Detection via Boxplots (Ref: Notebook cell 15)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        plt.figure(figsize=(12, 8))
        df[numeric_cols].boxplot()
        plt.title("Outlier Detection - Numeric Features")
        plt.savefig(self.config.eda_artifacts_dir / "outliers_boxplot.png")
        plt.close()
        
        logging.info("EDA completed. Statistics and plots saved to artifacts.")