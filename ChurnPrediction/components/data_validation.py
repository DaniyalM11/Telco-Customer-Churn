from ChurnPrediction.entity.config_entity import DataValidationConfig
from ChurnPrediction.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.logging.logger import logging
from ChurnPrediction.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from ChurnPrediction.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH,'columns')
        except Exception as e:
            raise ChurnPredictionException(e,sys)   
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
         try:
             return pd.read_csv(file_path)
         except Exception as e:
             raise ChurnPredictionException(e,sys)
         
    def validate_schema(self,df:pd.DataFrame)-> bool:   
        try:
            number_of_columns = len(self._schema_config)
            logging.info(f"Required number of columns in the schema:{number_of_columns}")
            logging.info(f"Dataframe has columns:{len(df.columns)}")
            if number_of_columns != len(df.columns):
                logging.error("Number of columns in the dataframe does not match the schema")
                return False
            else:
                logging.info("Number of columns in the dataframe matches the schema")
                return True            
        except Exception as e:
            raise ChurnPredictionException(e,sys)  
        
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
               d1=base_df[column]
               d2=current_df[column]
               is_same_dist=ks_2samp(d1,d2)
               if threshold <= is_same_dist.pvalue:
                   is_found=False
               else:
                   is_found=True
                   status=False    

               report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path    

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(drift_report_file_path,report)

        except Exception as e:
            raise ChurnPredictionException(e,sys)        
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating data validation")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path           

            #read the data from train and test file
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)            

            #validate the schema
            status = self.validate_schema(df=train_df)
            if not status:
                logging.error("Trained Data Schema validation failed")
            

            status = self.validate_schema(df=test_df)
            if not status:
                logging.error("Test Data Schema validation failed")

            # lets check datadrift
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)

            test_df.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)   

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path, 
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)    
