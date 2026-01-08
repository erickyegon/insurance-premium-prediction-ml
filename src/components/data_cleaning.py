import pandas as pd
from src.exception import CustomException
from src.logger import logging

class DataCleaning:
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting Data Cleaning")
        
        # 1. Standardize columns (Ref: Notebook cell 8)
        df.columns = df.columns.str.replace(" ", "_").str.lower()
        
        # 2. Handle Missing Values (Ref: Notebook cell 10)
        initial_rows = len(df)
        df.dropna(inplace=True)
        logging.info(f"Dropped {initial_rows - len(df)} rows containing null values.")
        
        # 3. Handle Duplicates (Ref: Notebook cell 12)
        df.drop_duplicates(inplace=True)
        
        # 4. Outlier Removal (Specific to notebook logic)
        # The notebook shows specific filtering for 'age' or 'income' if they exceed thresholds
        # Example: df = df[df['age'] <= 100] # Adjust based on notebook EDA findings
        
        logging.info("Data Cleaning completed.")
        return df