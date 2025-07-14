from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from dataclasses import dataclass
from src.configuration.aws_connection import buckets
import pickle 
from src.utils.main_utils import load_numpy_array_data

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifact,
                 feature_engineering_artifact: FeatureEngineeringArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.model_trainer_artifact = model_trainer_artifact
            self.feature_engineering_artifact = feature_engineering_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self):
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_trainer_artifact.trained_model_file_path
            buck = buckets()

            if buck.path_exists_in_s3(bucket_name=bucket_name, path=model_path):
                return True
            return None
        except Exception as e:
            raise  MyException(e,sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Best model found in production stage, loading it for evaluation.")
                buck = buckets()
                best_model = buck.download_file(
                    bucket=self.model_eval_config.bucket_name,
                    key=self.model_trainer_artifact.trained_model_file_path,
                    as_object=True
                )
                best_model = pickle.loads(best_model)
                test = load_numpy_array_data(file_path = self.feature_engineering_artifact.test_file_path)
                x, y = test[:, :-1], test[:, -1]
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)

            trained_model_f1_score=self.model_trainer_artifact.metric_artifact.f1_score

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=tmp_best_model_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_trainer_artifact.trained_model_file_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e