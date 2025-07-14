import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%Y_%m_%d_%H")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    latest_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_LATEST_DIR_NAME)
    timestamp: str = TIMESTAMP
    latest: str = DATA_INGESTION_LATEST_DIR_NAME

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    database_name: str = DATABASE_NAME
    folder_name: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    latest_data_folder_name: str = os.path.join(training_pipeline_config.latest_dir, DATA_INGESTION_DIR_NAME)
    data_file_path: str = os.path.join(folder_name, DATA_FILE_NAME)
    latest_data_file_path: str = os.path.join(latest_data_folder_name, DATA_FILE_NAME)
    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    bucket_name: str = MODEL_BUCKET_NAME
    artifact_path: str = os.path.join(folder_name, DATA_INGESTION_ARTIFACT_NAME)

@dataclass
class DataCleaningConfig:
    folder_name: str = os.path.join(training_pipeline_config.artifact_dir, DATA_CLEANING_DIR_NAME)
    latest_data_folder_name: str = os.path.join(training_pipeline_config.latest_dir, DATA_CLEANING_DIR_NAME)
    cleaned_data_dir: str = os.path.join(folder_name, DATA_CLEANING_CLEANED_DATA_DIR)
    latest_cleaned_data_dir: str = os.path.join(latest_data_folder_name, DATA_CLEANING_CLEANED_DATA_DIR)
    cleaned_data_file_path: str = os.path.join(folder_name, cleaned_data_dir, DATA_CLEANING_CLEANED_FILE_NAME)
    latest_data_file_path: str = os.path.join(latest_cleaned_data_dir, DATA_CLEANING_CLEANED_FILE_NAME)

@dataclass
class DataValidationConfig:
    folder_name: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    latest_folder_name: str = os.path.join(training_pipeline_config.latest_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(folder_name, DATA_VALIDATION_REPORT_FILE_NAME)
    latest_report_file_path: str = os.path.join(latest_folder_name, DATA_VALIDATION_REPORT_FILE_NAME)
    validation_status: bool = False

@dataclass
class FeatureEngineeringConfig:
    folder_name: str = os.path.join(training_pipeline_config.artifact_dir, FEATURE_ENGINEERING_DIR_NAME)
    latest_folder_name: str = os.path.join(training_pipeline_config.latest_dir, FEATURE_ENGINEERING_DIR_NAME)
    train_dir: str = os.path.join(folder_name, TRAIN_DIR_NAME)
    test_dir: str = os.path.join(folder_name, TEST_DIR_NAME)
    latest_train_dir: str = os.path.join(latest_folder_name, TRAIN_DIR_NAME)
    latest_test_dir: str = os.path.join(latest_folder_name, TEST_DIR_NAME)
    artifact_dir: str = os.path.join(latest_folder_name, FEATURE_ENGINEERING_ARTIFACT_DIR)
    split_size: float = SPLIT_SIZE

@dataclass
class ModelTrainerConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    folder_name: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    latest_folder_name: str = os.path.join(training_pipeline_config.latest_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(folder_name, MODEL_FILE_NAME)
    latest_trained_model_file_path: str = os.path.join(latest_folder_name, MODEL_FILE_NAME)
    max_iter: int = MODEL_TRAINER_MAX_ITER
    solver: str = MODEL_TRAINER_SOLVER
    c: float = MODEL_TRAINER_C
    expecected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE

@dataclass
class ModelEvaluationConfig:
    bucket_name: str = MODEL_BUCKET_NAME

@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME