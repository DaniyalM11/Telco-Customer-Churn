import sys
sys.path.insert(0, '/opt/airflow')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.components.data_validation import DataValidation
from ChurnPrediction.components.data_transformation import DataTransformation
from ChurnPrediction.components.model_trainer import ModelTrainer
from ChurnPrediction.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from ChurnPrediction.custom_logging.logger import logging
from ChurnPrediction.exception.exception import ChurnPredictionException

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
    description="Pipeline for churn prediction",
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
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def data_validation_task(ti):
        try:
            data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion")
            data_validation_config = DataValidationConfig(training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            logging.info("Initiating validation of data")
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def data_transformation_task(ti):
        try:
            data_validation_artifact = ti.xcom_pull(task_ids="data_validation")
            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            logging.info("Initiating transformation of data")
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def model_trainer_task(ti):
        try:
            data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            logging.info("Model training started")
            return model_trainer.initiate_model_trainer()
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

    train_model = PythonOperator(
        task_id="model_training",
        python_callable=model_trainer_task,
    )

    # Task dependencies
    ingest_data >> validate_data >> transform_data >> train_model
