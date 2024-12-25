from ChurnPrediction.logging.logger import logging
from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.constant import training_pipeline
from ChurnPrediction.constant.training_pipeline import TARGET_LABEL

# Config of the data ingestion

from ChurnPrediction.entity.config_entity import DataIngestionConfig
from ChurnPrediction.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def export_collection_as_dataframe(self):
        try:
            collection_name = self.data_ingestion_config.collection_name    

            df=pd.read_csv(self.data_ingestion_config.filedrop_location + training_pipeline.FILE_NAME)
            return df
        except Exception as e:
            raise ChurnPredictionException(e,sys)

    def export_data_into_feature_store(self,df:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            #createing folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            df.to_csv(feature_store_file_path,index=False,header=True)
            return df
        
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def split_data_as_train_test(self,df:pd.DataFrame):
        try:
            train_set, test_set = train_test_split(df,
                test_size=self.data_ingestion_config.train_test_split_ratio, stratify=df[TARGET_LABEL],random_state=training_pipeline.RANDOM_STATE)
        
            logging.info("Train and Test data split done")

            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")

            #this dir_path will work for both train and test data
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path,exist_ok=True)

            logging.info("exporting created for train and test data to filepath")

            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)

            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info("exported train and test data to filepath")    
            
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)
            dataingestionartifact = DataIngestionArtifact(self.data_ingestion_config.training_file_path,
                                                          self.data_ingestion_config.testing_file_path)
            return dataingestionartifact    
            
        except Exception as e:
            raise ChurnPredictionException(e,sys)    

                

        
