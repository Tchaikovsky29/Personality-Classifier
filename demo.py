from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_cleaning import DataCleaning
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataCleaningConfig, FeatureEngineeringConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig

ingest = DataIngestion(data_ingestion_config=DataIngestionConfig())
data_ingestion_artifact = ingest.initiate_data_ingestion()
validate = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=DataValidationConfig())
data_validation_artifact = validate.initiate_data_validation()
clean = DataCleaning(data_ingestion_artifact=data_ingestion_artifact,
                     data_cleaning_config=DataCleaningConfig(),
                     data_validation_artifact=data_validation_artifact)
cleaning_data_artifact = clean.initiate_data_cleaning()
features = FeatureEngineering(data_ingestion_artifact=data_ingestion_artifact,
                                    feature_engineering_config=FeatureEngineeringConfig(),
                                    data_cleaning_artifact=cleaning_data_artifact)
feature_engineering_artifact = features.initiate_feature_engineering()
trainer = ModelTrainer(feature_engineering_artifact=feature_engineering_artifact,
                        model_trainer_config=ModelTrainerConfig())
model_trainer_artifact = trainer.initiate_model_trainer()
evaluate = ModelEvaluation(model_eval_config=ModelEvaluationConfig(),
                           model_trainer_artifact=model_trainer_artifact,
                           feature_engineering_artifact=feature_engineering_artifact)
model_evaluation_artifact = evaluate.initiate_model_evaluation()
pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                     model_pusher_config=ModelPusherConfig())
pusher.initiate_model_pusher()