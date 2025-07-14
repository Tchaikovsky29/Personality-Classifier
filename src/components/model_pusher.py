import sys
from src.configuration.aws_connection import buckets
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.configuration.aws_connection import buckets
import pickle

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = buckets()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self):
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Checking if new model is better than the existing model in production stage....")
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("New model is not better than the existing model in production stage. Exiting model pusher.")
                return None
            else:
                logging.info("New model is better than the existing model in production stage. Proceeding with model pusher.")
                logging.info("Loading the best model from the model evaluation artifact...")
                with open(self.model_evaluation_artifact.trained_model_path, 'rb') as f:
                    best_model = pickle.load(f)

                self.s3.upload_file(
                    bucket=self.model_pusher_config.bucket_name,
                    key=self.model_evaluation_artifact.s3_model_path,
                    body=pickle.dumps(best_model)
                )
                logging.info("Best model uploaded to S3 bucket successfully.")
            
        except Exception as e:
            raise MyException(e, sys) from e