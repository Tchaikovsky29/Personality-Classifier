import sys
from typing import Tuple
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, feature_engineering_artifact: FeatureEngineeringArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.feature_engineering_artifact = feature_engineering_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training RandomForestClassifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            # Initialize RandomForestClassifier with specified parameters
            model = LogisticRegression(max_iter=self.model_trainer_config.max_iter,
                                       solver=self.model_trainer_config.solver,
                                       C= self.model_trainer_config.c)

            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(accuracy_score=accuracy, f1_score=f1, precision_score=precision, recall_score=recall)
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.feature_engineering_artifact.train_file_path)
            test_arr = load_numpy_array_data(file_path=self.feature_engineering_artifact.test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expecected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")

            logging.info(f"Saving model object at {self.model_trainer_config.trained_model_file_path}")
            os.makedirs(self.model_trainer_config.folder_name, exist_ok=True)
            os.makedirs(self.model_trainer_config.latest_folder_name, exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, trained_model)
            save_object(self.model_trainer_config.latest_trained_model_file_path, trained_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")
            
            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.latest_trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e