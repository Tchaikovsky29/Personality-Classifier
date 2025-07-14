from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    ingested_data_path: str
    bucket_name: str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str

@dataclass
class DataCleaningArtifact:
    cleaned_data_file_path:str

@dataclass
class FeatureEngineeringArtifact:
    train_file_path: str
    test_file_path: str
    time_alone_bins: list
    scaler: object 
    poly_features: object

@dataclass
class ClassificationMetricArtifact:
    accuracy_score:float
    f1_score:float
    precision_score:float
    recall_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str
