from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.components.data_validation import DataValidation
from ChurnPrediction.components.data_transformation import DataTransformation
from ChurnPrediction.logging.logger import logging
from ChurnPrediction.exception.exception import ChurnPredictionException 
from ChurnPrediction.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig

import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config) 
        logging.info("Initiating ingestion of data")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion done")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("Initiating validation of data")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation done")
        print(data_validation_artifact)
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initiating transformation of data")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation done")
        print(data_transformation_artifact)

    except Exception as e:
        raise ChurnPredictionException(e,sys)        
