import os
import sys
import numpy as np
import pandas as pd
from utils.main_utils import churn_filename


"""
defining common constant variables for training pipeline
"""

TARGET_LABEL: str = "Churn"
PIPELINE_NAME: str = "ChurnPrediction"
ARTIFACTS_DIR: str = "Artifacts"
FILE_NAME: str = churn_filename()

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "ChurnPrediction"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2