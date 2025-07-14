import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.configuration.aws_connection import buckets
from src.entity.config_entity import DataCleaningConfig
from src.entity.artifact_entity import DataCleaningArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataCleaning:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_cleaning_config: DataCleaningConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_cleaning_config = data_cleaning_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def impute_missing_values(self, df):
        """
        Creates and returns a data cleaner object for the data.
        """
        logging.info("Entered get_data_cleaning_object method of DataCleaning class")

        try:
            # Initialize transformers
            numerical_imputer= SimpleImputer(strategy='median')
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            logging.info("Transformers Initialized: numeric imputer, categorical imputer")

            # Load schema configurations
            numerical_features = self._schema_config['numerical_columns']
            categorical_features = self._schema_config['categorical_columns']
            logging.info("Cols loaded from schema.")

            # Impute numerical features
            df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
            logging.info("Numerical features imputed.")
            # Impute categorical features
            df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
            logging.info("Categorical features imputed.")
            return df

        except Exception as e:
            logging.exception("Exception occurred in get_data_cleaning_object method of DataCleaning class")
            raise MyException(e, sys) from e

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        categorical_features = self._schema_config['categorical_columns']
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=bool)
        return df
    
    def _cap_outliers(self, df):
        numerical_features = self._schema_config['numerical_columns']
        for col in numerical_features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            # Less aggressive upper bound for extrovert-related features
            upper_bound = Q3 + 2.5 * IQR if col in ['Social_event_attendance', 'Friends_circle_size', 'Post_frequency'] else Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

    def _encode_target(self, df):
        target = self._schema_config['target_column']
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        return df

    def initiate_data_cleaning(self) -> DataCleaningArtifact:
        """
        Initiates the data cleaning component for the pipeline.
        """
        try:
            logging.info("Data cleaning Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            df = self.read_data(file_path=self.data_ingestion_artifact.ingested_data_path)
            logging.info("data loaded")

            logging.info("Encoding target feature")
            df = self._encode_target(df)
            logging.info("Target feature encoded in df.")

            logging.info("Capping outliers in numerical features")
            df = self._cap_outliers(df)
            logging.info("Outliers capped in df.")

            logging.info("Initializing transformation for data")
            target = df[TARGET_COLUMN]
            X = df.drop(columns=[self._schema_config['target_column']])
            df = self.impute_missing_values(X)
            logging.info("Transformation done end to end to df.")

            logging.info("Creating dummy columns for categorical features")
            df = self._create_dummy_columns(df)
            logging.info("Dummy columns created for df.")

            df = pd.concat([df, target], axis=1)
            logging.info("Target column concatenated with df.")

            file_path = self.data_cleaning_config.cleaned_data_file_path
            latest_file_path = self.data_cleaning_config.latest_data_file_path
            client = buckets()
            
            logging.info("Saving cleaned data to file")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            os.makedirs(os.path.dirname(latest_file_path), exist_ok=True)
            logging.info("Directories created for cleaned data and object file")

            df.to_csv(file_path, index=False, header = True)
            df.to_csv(latest_file_path, index=False, header = True)
            client.upload_file(bucket=self.data_ingestion_artifact.bucket_name, key = self.data_cleaning_config.latest_data_file_path, file_path=file_path)

            # logging.info("Data cleaning completed successfully")
            return DataCleaningArtifact(
                cleaned_data_file_path = self.data_cleaning_config.cleaned_data_file_path
            )
            

        except Exception as e:
            raise MyException(e, sys) from e