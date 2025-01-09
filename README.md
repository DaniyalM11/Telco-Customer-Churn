# Telco Customer Churn Prediction

This repository contains a comprehensive project for predicting customer churn in the telecommunications industry. The project demonstrates a robust end-to-end machine learning pipeline, incorporating data engineering, model development, deployment, and monitoring processes. Designed with scalability and production-readiness in mind, it leverages modern tools and practices such as Airflow, Docker, and MLflow.

---

## **Features**

1. **Data Engineering:**
   - ETL processes for ingesting and validating data.
   - Handling missing values, type conversions, and preprocessing.
   - Feature transformations such as scaling and one-hot encoding.

2. **Model Development:**
   - Implements multiple machine learning models including Logistic Regression, Random Forest, XGBoost, and LightGBM.
   - Hyperparameter tuning using GridSearchCV.
   - Model selection based on performance metrics.

3. **MLOps Integration:**
   - **MLflow** for tracking experiments, logging models, and managing artifacts.
   - Automated pipelines with **Apache Airflow** for orchestration of all tasks.

4. **Deployment:**
   - **FastAPI** for serving predictions via a REST API.
   - Frontend interface for displaying prediction results.

5. **Monitoring:**
   - Tracks model performance in production using logs and visualizations.
   - Alerts for data drift and model degradation.

6. **Infrastructure:**
   - Dockerized environment for reproducibility.
   - GitHub Actions for CI/CD pipelines.
   - Configurable for deployment on cloud platforms such as AWS.

---

## **Technologies Used**

- **Programming Language:** Python
- **Libraries:**
  - pandas, NumPy, scikit-learn, statsmodels
  - XGBoost, LightGBM
  - FastAPI, Jinja2
- **Tools:**
  - Apache Airflow
  - MLflow
  - Docker
  - GitHub Actions

---

## **Pipeline Overview**

### **1. Data Ingestion**
- **Source:** Raw CSV file (`Telco_Customer_DataSet.csv`).
- **Process:**
  - Data is split into training and testing sets.
  - Stored in an `Artifacts` directory for reuse.

### **2. Data Validation**
- **Checks:**
  - Schema validation.
  - Missing values and data consistency.
- **Outputs:**
  - Validated training and testing datasets.

### **3. Data Transformation**
- **Transformations:**
  - Numeric features scaled with StandardScaler.
  - Categorical features encoded with OneHotEncoder.
  - Columns dropped based on domain relevance.
- **Pipelines:**
  - Preprocessing pipeline for feature engineering.
  - VIF (Variance Inflation Factor) analysis to handle multicollinearity.
- **Output:**
  - Transformed datasets stored for model training and evaluation.

### **4. Model Training**
- **Models:**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- **Metrics:**
  - Recall, Precision, F1 Score, Accuracy, ROC-AUC.
- **Selection:** Best model saved based on cross-validation performance.

### **5. Deployment and Inference**
- **FastAPI:**
  - Serves predictions through REST endpoints.
  - Upload CSVs for batch inference.
- **Frontend:**
  - Displays results in an interactive table using HTML templates.

### **6. Monitoring**
- **MLflow:**
  - Logs model parameters, metrics, and artifacts.
- **Alerts:**
  - Drift detection for incoming data compared to training distribution.

---





