import os
import sys
from pandas import DataFrame
from src.configuration.aws_connection import buckets
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.data import Data
from io import StringIO

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)

    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method imports data from mongodb to csv file, and then uploads it to S3 bucket
        
        Output      :   data is stored in csv file in S3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Importing data from mongodb")
            my_data = Data()
            dataframe = my_data.export_collection_as_dataframe(database_name=self.data_ingestion_config.database_name, collection_name=self.data_ingestion_config.collection_name)
            if "_id" in dataframe.columns:
                dataframe.drop(columns=["_id"], inplace=True)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            logging.info(f"Imported data from mongodb successfully")

            os.makedirs(self.data_ingestion_config.folder_name, exist_ok=True)
            os.makedirs(self.data_ingestion_config.latest_data_folder_name, exist_ok=True)

            logging.info("Exporting data to feature store")
            file_path = self.data_ingestion_config.data_file_path
            latest_file_path = self.data_ingestion_config.latest_data_file_path
            client = buckets()
            
            dataframe.to_csv(file_path, index=False, header = True)
            dataframe.to_csv(latest_file_path, index=False, header = True)
            client.upload_file(bucket=self.data_ingestion_config.bucket_name, key = self.data_ingestion_config.latest_data_file_path, file_path=file_path)
            logging.info(f"Data exported to {file_path} in S3 bucket {self.data_ingestion_config.bucket_name}")

        except Exception as e:
            logging.error(f"Error in exporting data to feature store: {e}")
            raise MyException(e,sys)

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   data set is saved in S3 bucket and locally
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            self.export_data_into_feature_store()

            logging.info("Got the data from mongodb and exported it to feature store")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
        
            data_ingestion_artifact = DataIngestionArtifact(
                ingested_data_path=self.data_ingestion_config.data_file_path, bucket_name=self.data_ingestion_config.bucket_name
            )
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e