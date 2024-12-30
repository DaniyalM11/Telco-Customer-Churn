from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.components.data_validation import DataValidation
from ChurnPrediction.components.data_transformation import DataTransformation
from ChurnPrediction.components.model_trainer import ModelTrainer
from ChurnPrediction.custom_logging.logger import logging
from ChurnPrediction.exception.exception import ChurnPredictionException 
from ChurnPrediction.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
from ChurnPrediction.entity.config_entity import ModelTrainerConfig 
from ChurnPrediction.utils.main_utils import save_object, load_object
import os 

import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()


#        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
#        data_ingestion = DataIngestion(data_ingestion_config) 
#        logging.info("Initiating ingestion of data")
#        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
#        logging.info("Data ingestion done")
#        print(data_ingestion_artifact)
#        data_validation_config = DataValidationConfig(training_pipeline_config)
#        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
#        logging.info("Initiating validation of data")
#        data_validation_artifact = data_validation.initiate_data_validation()
#        logging.info("Data validation done")
#        print(data_validation_artifact)
#        data_transformation_config = DataTransformationConfig(training_pipeline_config)
#        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
#        logging.info("Initiating transformation of data")
#        data_transformation_artifact = data_transformation.initiate_data_transformation()
#        logging.info("Data transformation done")
#        print(data_transformation_artifact)
#        save_object(r"C:\Users\Daniy\Telco Customer Churn Prediction\data_transformation_artifact",data_transformation_artifact)

        data_transformation_artifact = load_object(r"Artifacts\data_transformation_artifact.pkl")
        logging.info("Model training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config= model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer.initiate_model_trainer()

    except Exception as e:
        raise ChurnPredictionException(e,sys)        
