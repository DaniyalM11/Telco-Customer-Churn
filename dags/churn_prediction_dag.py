import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.components.data_validation import DataValidation
from ChurnPrediction.components.data_transformation import DataTransformation
from ChurnPrediction.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from ChurnPrediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from ChurnPrediction.custom_logging.logger import logging
from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.utils.main_utils import save_object

# Define the pipeline configuration
training_pipeline_config = TrainingPipelineConfig()

# Default args for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id="churn_prediction_pipeline",
    default_args=default_args,
    description="Pipeline for churn prediction (data ingestion, validation, transformation)",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["churn_prediction"],
) as dag:

    def data_ingestion_task():
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            logging.info("Initiating ingestion of data")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            # Serialize the artifact explicitly
            return data_ingestion_artifact.__dict__
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def data_validation_task(ti):
        try:
            # Deserialize manually
            data_ingestion_artifact_dict = ti.xcom_pull(task_ids="data_ingestion")
            data_ingestion_artifact = DataIngestionArtifact(**data_ingestion_artifact_dict)

            data_validation_config = DataValidationConfig(training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            logging.info("Initiating validation of data")

            data_validation_artifact = data_validation.initiate_data_validation()

            # Serialize the artifact explicitly
            return data_validation_artifact.__dict__
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def data_transformation_task(ti):
        try:
            # Deserialize manually
            data_validation_artifact_dict = ti.xcom_pull(task_ids="data_validation")
            data_validation_artifact = DataValidationArtifact(**data_validation_artifact_dict)

            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            logging.info("Initiating transformation of data")

            data_transformation_artifact = data_transformation.initiate_data_transformation()

            # Save artifact to a mounted volume
            artifact_file_path = "/opt/airflow/Artifacts/data_transformation_artifact.pkl"
            save_object(artifact_file_path, data_transformation_artifact)
            logging.info(f"Data Transformation artifact saved at {artifact_file_path}")

            # Serialize the artifact explicitly
            return data_transformation_artifact.__dict__
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    # Define tasks
    ingest_data = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion_task,
    )

    validate_data = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation_task,
    )

    transform_data = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation_task,
    )

    # Task dependencies
    ingest_data >> validate_data >> transform_data
