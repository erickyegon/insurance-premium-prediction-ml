from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from src.logger import logging
from src.exception import CustomException
from src.components.data_cleaning import DataCleaning
from dataclasses import dataclass
from pathlib import Path
import sys
import pandas as pd
import numpy as np  

@dataclass
class FeatureEngineeringConfig:
    preprocessor_obj_file_path: Path = Path("artifacts/preprocessor.pkl")

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig | None = None):
        self.config = config or FeatureEngineeringConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """Creates the transformation pipeline based on the notebook logic."""
        try:
            # Defined based on notebook features
            numerical_columns = ["age", "number_of_dependants", "income_lakhs"]
            categorical_columns = [
                "gender", "region", "marital_status", "physical_activity", 
                "stress_level", "bmi_category", "smoking_status", 
                "employment_status", "medical_history", "insurance_plan"
            ]

            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_feature_engineering(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Target column from notebook: annual_premium_amount
            target_column_name = "annual_premium_amount"

            # Clean data first
            train_df = DataCleaning.clean_data(train_df)
            test_df = DataCleaning.clean_data(test_df)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()
            
            # Transform
            train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target for modeling
            train_data = np.c_[train_arr, np.array(target_feature_train_df)]
            test_data = np.c_[test_arr, np.array(target_feature_test_df)]

            # Save preprocessor for production/inference
            with open(self.config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessing_obj, f)

            return train_data, test_data, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)