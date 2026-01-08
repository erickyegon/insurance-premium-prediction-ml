from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import joblib
from pathlib import Path
import pandas as pd 
from src.components.data_cleaning import DataCleaning
from src.components.feature_engineering import FeatureEngineering   
from src.exception import CustomException
import sys      
from src.logger import logging


class FinalDataPreparation:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)

    def calculate_vif(self, df: pd.DataFrame):
        """Calculates VIF to detect multicollinearity as per notebook Section 4."""
        numeric_df = df.select_dtypes(include=['number']).drop(columns=['annual_premium_amount'])
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_df.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                          for i in range(len(numeric_df.columns))]
        return vif_data

    def prepare_for_modeling(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 1. Clean both sets
            train_df = DataCleaning.initiate_cleaning(train_df)
            test_df = DataCleaning.initiate_cleaning(test_df)

            # 2. Logic for high VIF feature removal (Ref: Notebook section 4.2)
            # In your notebook, if 'income_level' and 'income_lakhs' are redundant:
            cols_to_drop = ["income_level"] # Adjust based on notebook VIF results
            train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # 3. Transform data
            fe = FeatureEngineering()
            preprocessor = fe.get_transformer_object()
            
            X_train = train_df.drop(columns=['annual_premium_amount'])
            y_train = train_df['annual_premium_amount']
            X_test = test_df.drop(columns=['annual_premium_amount'])
            y_test = test_df['annual_premium_amount']

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # 4. Save the preprocessor object
            joblib.dump(preprocessor, self.artifacts_dir / "preprocessor.pkl")

            # Return prepared arrays for the Model Module
            return (
                np.c_[X_train_arr, np.array(y_train)],
                np.c_[X_test_arr, np.array(y_test)]
            )

        except Exception as e:
            raise CustomException(e, sys)