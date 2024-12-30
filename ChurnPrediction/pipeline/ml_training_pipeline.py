import os
import sys

from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.custom_logging.logger import logging

from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.components.data_validation import DataValidation
from ChurnPrediction.components.data_transformation import DataTransformation
from ChurnPrediction.components.model_trainer import ModelTrainer
from ChurnPrediction.utils.main_utils import load_object

from ChurnPrediction.entity.config_entity import(
TrainingPipelineConfig,
DataIngestionConfig,
DataValidationConfig,
DataTransformationConfig,
ModelTrainerConfig
)

from ChurnPrediction.entity.artifact_entity import(
DataIngestionArtifact,
DataValidationArtifact,
DataTransformationArtifact,
ModelTrainerArtifact
)    

class ML_TrainingPipeline:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
        except Exception as e:
            raise ChurnPredictionException(e,sys) from e

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
            logging.info("Initiate the data Validation")
            data_validation_artifact=data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def run_pipeline(self):
        try:
            data_transformation_artifact = load_object("/DE_Artifact")
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            
            return model_trainer_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)   

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        ml_training_pipeline = ML_TrainingPipeline(training_pipeline_config)
        model_trainer_artifact = ml_training_pipeline.run_pipeline()
        print(model_trainer_artifact)
    except Exception as e:
        raise ChurnPredictionException(e,sys)                 