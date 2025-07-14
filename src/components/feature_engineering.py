import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import os
from src.configuration.aws_connection import buckets
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import FeatureEngineeringConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, DataIngestionArtifact, DataCleaningArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
import pickle

class FeatureEngineering:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 feature_engineering_config: FeatureEngineeringConfig,
                 data_cleaning_artifact: DataCleaningArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.feature_engineering_config = feature_engineering_config
            self.data_cleaning_artifact = data_cleaning_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def train_test_split(self, df: pd.DataFrame):
        try:
            X = df.drop(columns=[self._schema_config['target_column']])
            y = df[self._schema_config['target_column']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.feature_engineering_config.split_size, stratify=y, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise MyException(e, sys)

    def interaction_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        try:
            X_train['Alone_to_Social_Ratio'] = X_train['Time_spent_Alone'] / (X_train['Social_event_attendance'] + 1)
            X_test['Alone_to_Social_Ratio'] = X_test['Time_spent_Alone'] / (X_test['Social_event_attendance'] + 1)
            X_train['Social_Comfort_Index'] = (X_train['Friends_circle_size'] + X_train['Post_frequency'] - X_train['Stage_fear_Yes']) / 3
            X_test['Social_Comfort_Index'] = (X_test['Friends_circle_size'] + X_test['Post_frequency'] - X_test['Stage_fear_Yes']) / 3
            X_train['Social_Overload'] = X_train['Drained_after_socializing_Yes'] * X_train['Social_event_attendance']
            X_test['Social_Overload'] = X_test['Drained_after_socializing_Yes'] * X_test['Social_event_attendance']
            return X_train, X_test
        except Exception as e:
            raise MyException(e, sys)
        
    def binned_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, time_alone_bins) -> pd.DataFrame:
        try:
            X_train['Time_spent_Alone_Binned'] = pd.cut(X_train['Time_spent_Alone'], bins=time_alone_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
            X_test['Time_spent_Alone_Binned'] = pd.cut(X_test['Time_spent_Alone'], bins=time_alone_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
            X_train = pd.get_dummies(X_train, columns=['Time_spent_Alone_Binned'], drop_first=True)
            X_test = pd.get_dummies(X_test, columns=['Time_spent_Alone_Binned'], drop_first=True)
            return X_train, X_test
        except Exception as e:
            raise MyException(e, sys)
    
    def polynomial_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        try:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            poly_features_train = poly.fit_transform(X_train[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
            poly_features_test = poly.transform(X_test[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
            poly_feature_names = poly.get_feature_names_out(['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'])
            X_train[poly_feature_names] = poly_features_train
            X_test[poly_feature_names] = poly_features_test
            return X_train, X_test, poly
        except Exception as e:
            raise MyException(e, sys)
    
    def scaler(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        except Exception as e:
            raise MyException(e, sys)
    
    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        """
        Initiates the feature engineering component for the pipeline.
        """
        try:
            logging.info("Feature Engineering Started !!!")
            df = self.read_data(file_path=self.data_cleaning_artifact.cleaned_data_file_path)
            logging.info("Data loaded successfully")

            logging.info("Computing time spent alone bins")
            time_alone_bins = pd.qcut(df['Time_spent_Alone'], q=3, retbins=True)[1]

            logging.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = self.train_test_split(df)
            logging.info("Train-test split completed")

            logging.info("Creating interaction features")
            X_train, X_test = self.interaction_feature_engineering(X_train, X_test)
            logging.info("Interaction features engineered")

            logging.info("Creating binned features")
            X_train, X_test = self.binned_feature_engineering(X_train, X_test, time_alone_bins)
            logging.info("Binned features engineered")

            logging.info("Applying polynomial feature engineering")
            X_train, X_test, poly = self.polynomial_feature_engineering(X_train, X_test)
            logging.info("Polynomial features engineered")

            logging.info("Scaling data")
            X_train_scaled, X_test_scaled, scaler = self.scaler(X_train, X_test)
            logging.info("Data scaled")

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            logging.info("Creating directories for transformed data")
            os.makedirs(self.feature_engineering_config.train_dir, exist_ok=True)
            os.makedirs(self.feature_engineering_config.test_dir, exist_ok=True)
            os.makedirs(self.feature_engineering_config.latest_train_dir, exist_ok=True)
            os.makedirs(self.feature_engineering_config.latest_test_dir, exist_ok=True)

            train_file_path = os.path.join(self.feature_engineering_config.train_dir, 'train.npy')
            test_file_path = os.path.join(self.feature_engineering_config.test_dir, 'test.npy')
            latest_train_file_path = os.path.join(self.feature_engineering_config.latest_train_dir, 'train.npy')
            latest_test_file_path = os.path.join(self.feature_engineering_config.latest_test_dir, 'test.npy')

            logging.info("Saving transformed data to files")
            save_numpy_array_data(file_path=train_file_path, array=train_arr)
            save_numpy_array_data(file_path=test_file_path, array=test_arr)
            save_numpy_array_data(file_path=latest_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=latest_test_file_path, array=test_arr)
            logging.info("Transformed data saved successfully")

            client = buckets()
            logging.info("Uploading transformed data to S3 bucket")
            client.upload_file(bucket=self.data_ingestion_artifact.bucket_name, key=latest_train_file_path, file_path=latest_train_file_path)
            client.upload_file(bucket=self.data_ingestion_artifact.bucket_name, key=latest_test_file_path, file_path=latest_test_file_path)
            logging.info("Transformed data uploaded to S3 bucket successfully")

            feature_engineering_artifact = FeatureEngineeringArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                time_alone_bins = time_alone_bins,
                scaler = scaler,
                poly_features = poly
            )

            client.upload_file(bucket=self.data_ingestion_artifact.bucket_name, key=self.feature_engineering_config.artifact_dir, body=pickle.dumps(feature_engineering_artifact))
            logging.info("Feature Engineering completed successfully")
            return feature_engineering_artifact
        except Exception as e:
            logging.exception("Exception occurred in initiate_feature_engineering method of DataTransformation class")
            raise MyException(e, sys) from e
        
