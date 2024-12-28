Telco Customer Churn Prediction

This repository contains a comprehensive project for predicting customer churn in the telecommunications industry. The project demonstrates a robust end-to-end machine learning pipeline, incorporating data engineering, model development, deployment, and monitoring processes. Designed with scalability and production-readiness in mind, it leverages modern tools and practices such as Airflow, Docker, and MLflow.

Features

Data Engineering:

ETL processes for ingesting and validating data.

Handling missing values, type conversions, and preprocessing.

Feature transformations such as scaling and one-hot encoding.

Model Development:

Implements multiple machine learning models including Logistic Regression, Random Forest, XGBoost, and LightGBM.

Hyperparameter tuning using GridSearchCV.

Model selection based on performance metrics.

MLOps Integration:

MLflow for tracking experiments, logging models, and managing artifacts.

Automated pipelines with Apache Airflow for orchestration of all tasks.

Deployment:

FastAPI for serving predictions via a REST API.

Frontend interface for displaying prediction results.

Monitoring:

Tracks model performance in production using logs and visualizations.

Alerts for data drift and model degradation.

Infrastructure:

Dockerized environment for reproducibility.

GitHub Actions for CI/CD pipelines.

Configurable for deployment on cloud platforms such as AWS.

Technologies Used

Programming Language: Python

Libraries:

pandas, NumPy, scikit-learn, statsmodels

XGBoost, LightGBM

FastAPI, Jinja2

Tools:

Apache Airflow

MLflow

Docker

GitHub Actions

Pipeline Overview

1. Data Ingestion

Source: Raw CSV file (Telco_Customer_DataSet.csv).

Process:

Data is split into training and testing sets.

Stored in an Artifacts directory for reuse.

2. Data Validation

Checks:

Schema validation.

Missing values and data consistency.

Outputs:

Validated training and testing datasets.

3. Data Transformation

Transformations:

Numeric features scaled with StandardScaler.

Categorical features encoded with OneHotEncoder.

Columns dropped based on domain relevance.

Pipelines:

Preprocessing pipeline for feature engineering.

VIF (Variance Inflation Factor) analysis to handle multicollinearity.

Output:

Transformed datasets stored for model training and evaluation.

4. Model Training

Models:

Logistic Regression

Random Forest

XGBoost

LightGBM

Metrics:

Recall, Precision, F1 Score, Accuracy, ROC-AUC.

Selection: Best model saved based on cross-validation performance.

5. Deployment and Inference

FastAPI:

Serves predictions through REST endpoints.

Upload CSVs for batch inference.

Frontend:

Displays results in an interactive table using HTML templates.

6. Monitoring

MLflow:

Logs model parameters, metrics, and artifacts.

Alerts:

Drift detection for incoming data compared to training distribution.

Setup Instructions

1. Clone the Repository

git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction

2. Install Dependencies

pip install -r requirements.txt

3. Run the Pipeline

Training the Model:

python main.py

Starting the FastAPI Server:

uvicorn app:app --reload

4. Access the API

Documentation available at: http://localhost:8000/docs

Upload CSV files and view predictions.

5. Run Airflow DAGs

Start Airflow:

airflow standalone

Access UI: http://localhost:8080

6. Docker

Build and Run:

docker-compose up --build

CI/CD Pipeline

GitHub Actions Workflow:

Linting and unit tests.

Docker image builds and push to ECR.

Deploys containers to production.

Future Improvements

Integration with cloud storage (AWS S3 or GCP Storage).

Implementing advanced hyperparameter tuning (Optuna or Ray Tune).

Real-time model monitoring with Prometheus and Grafana.

Extending frontend for advanced visualizations.

Project Structure

Telco_Customer_Churn_Prediction/
├── app.py               # FastAPI application
├── main.py              # Entry point for the pipeline
├── ChurnPrediction/
│   ├── components/      # Data ingestion, transformation, model training
│   ├── entity/          # Configurations and artifacts
│   ├── exception/       # Custom exception handling
│   ├── logging/         # Logger setup
│   ├── pipeline/        # Training pipeline
│   ├── utils/           # Utility functions
├── Dockerfile           # Docker setup
├── requirements.txt     # Python dependencies
├── templates/           # HTML templates for frontend
├── tests/               # Unit tests
└── README.md            # Project documentation

License

This project is licensed under the MIT License.

Contact

Author: Your Name

Email: your.email@example.com

LinkedIn: Your Profile