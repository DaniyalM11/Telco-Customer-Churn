import os
import sys
import numpy as np
import pandas as pd
from ChurnPrediction.utils.main_utils import churn_filename


"""
defining common constant variables for training pipeline
"""

TARGET_LABEL: str = "Churn"
PIPELINE_NAME: str = "ChurnPrediction"
ARTIFACTS_DIR: str = "Artifacts"
FILE_NAME: str = churn_filename()

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
RANDOM_STATE: int = 42

SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR: str = "saved_model"
MODEL_FILE_NAME: str = "model.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "ChurnPrediction"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_FILEDROP_LOCATION = os.path.join("Telco_Customer_Data","")


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train.csv"

DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test.csv"
DATA_TRANSFORMATION_COLUMNS_TO_FLOAT64: list = ["TotalCharges"]
DATA_TRANSFORMATION_COLUMNS_TO_STR_FOR_BINARY_DECISION: list = ["SeniorCitizen"]
DATA_TRANSFORMATION_ROWS_DROPPED_BASEDS_ON_NULL_VALUES_OF_COLUMN: list = ["TotalCharges"]

DATA_TRANSFORMATION_VIF_REPORT_DIR: str = "VIF_report"
DATA_TRANSFORMATION_REPORT_BEFORE_FILE_NAME: str = "vif_report_before.yaml"
DATA_TRANSFORMATION_REPORT_AFTER_FILE_NAME: str = "vif_report_after.yaml"
DATA_TRANSFORMATION_COLUMNS_TO_DROP: list = ["customerID"]


"""
Model Trainer related constant start with MODEL_TRAINER VAR NAME 
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_METRICS_FILE_NAME: str = "metrics.yaml"
MODEL_TRAINER_METRICS_DIR_NAME: str = "metrics"




