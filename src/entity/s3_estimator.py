from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame
from src.logger import logging
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class Proj1Estimator:
    """
    This class is used to save and retrieve our model from s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model:MyModel=None


    def is_model_present(self,model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self,)->MyModel:
        """
        Load the model from the model_path
        :return:
        """

        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)

    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            raise MyException(e, sys)


    def interaction_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['Alone_to_Social_Ratio'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
            df['Social_Comfort_Index'] = (df['Friends_circle_size'] + df['Post_frequency'] - df['Stage_fear_Yes']) / 3
            df['Social_Overload'] = df['Drained_after_socializing_Yes'] * df['Social_event_attendance']
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def binned_feature_engineering(self, df: pd.DataFrame, time_alone_bins) -> pd.DataFrame:
        try:
            df['Time_spent_Alone_Binned'] = pd.cut(df['Time_spent_Alone'], bins=time_alone_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
            df = pd.get_dummies(df, columns=['Time_spent_Alone_Binned'], drop_first=True)
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def polynomial_feature_engineering(self, df: pd.DataFrame, poly = PolynomialFeatures) -> pd.DataFrame:
        try:
            poly = poly
            df = poly.fit_transform(df[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
            poly_feature_names = poly.get_feature_names_out(['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'])
            df[poly_feature_names] = df
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def scaler(self, df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        try:
            scaler = scaler
            df = scaler.fit(df)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, df: pd.DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply scaling transformations using the pre-trained preprocessing object
            df = self.interaction_feature_engineering(df)
            df = self.binned_feature_engineering(df, self.feature_engineering_artifact.time_alone_bins)
            df = self.polynomial_feature_engineering(df, self.feature_engineering_artifact.ploy_features)
            transformed_feature = self.scaler(df, self.feature_engineering_artifact.scaler)

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"