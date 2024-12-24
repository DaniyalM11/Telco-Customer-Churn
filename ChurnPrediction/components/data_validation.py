from ChurnPrediction.entity.config_entity import DataValidationConfig
from ChurnPrediction.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.logging.logger import logging
from ChurnPrediction.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from ChurnPrediction.utils.main_utils import read_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ChurnPredictionException(e,sys)   
