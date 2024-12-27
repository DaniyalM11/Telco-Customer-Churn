import os
import sys
import pandas as pd

from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.logging.logger import logging

from ChurnPrediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from ChurnPrediction.entity.config_entity import ModelTrainerConfig

from ChurnPrediction.utils.main_utils import save_object, read_csv_data, write_yaml_file
from ChurnPrediction.utils.ml_utils.metric.classification_metric import get_classification_score
from ChurnPrediction.utils.ml_utils.model.estimator import ChurnPredictionModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    roc_auc_score,
    make_scorer
)
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import parallel_backend
import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys) from e        

    def track_mlflow(self, best_model, classificationmetric, model_name="model"):
        with mlflow.start_run():
            accuracy_score = classificationmetric.accuracy_score
            f1_score = classificationmetric.f1_score
            recall_score = classificationmetric.recall_score
            precision_score = classificationmetric.precision_score
            roc_auc_score = classificationmetric.roc_auc_score

            mlflow.log_metric("accuracy_score", accuracy_score)
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("roc_auc_score", roc_auc_score)
            mlflow.sklearn.log_model(best_model, model_name)


    def train_model(self,X_train, y_train, X_test, y_test):
        #Custom Scorer for GridSearchCV
        #scoring_metric = 'f1'  # or 'recall', 'precision', 'accuracy', etc.
        my_scorer = make_scorer(recall_score, zero_division=0)

        param_grid_lr = {
            'C': [0.01, 0.1, 1.0, 10.0],            
            'penalty': ['l1', 'l2'],               
            'class_weight': ['balanced'] 
        }

        # Random Forest
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }

        # XGBoost & LightGBM
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight_val = (neg_count / pos_count) if pos_count else 1

        param_grid_xgb = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [scale_pos_weight_val*0.8, scale_pos_weight_val, scale_pos_weight_val*1.2]
        }

        # LightGBM
        param_grid_lgb = {
            'n_estimators': [100, 200],
            'max_depth': [-1, 5],     
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 63],
            'scale_pos_weight': [scale_pos_weight_val]
        }


        models = [
            (
                'LogisticRegression',
                LogisticRegression(
                    solver='liblinear',
                    random_state=42,
                    class_weight='balanced'
                ),
                param_grid_lr
            ),
            (
                'RandomForest',
                RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced'
                ),
                param_grid_rf
            ),
            (
                'XGBoost',
                xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    scale_pos_weight=scale_pos_weight_val
                ),
                param_grid_xgb
            ),
            (
                'LightGBM',
                lgb.LGBMClassifier(
                    random_state=42,
                    scale_pos_weight=scale_pos_weight_val
                ),
                param_grid_lgb
            )
        ]


        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        results = []

        for model_name, model_obj, param_grid in models:
            print(f"\n=== Tuning {model_name} ===")
            grid_search = GridSearchCV(
                estimator=model_obj,
                param_grid=param_grid,
                scoring=my_scorer,         
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit on the training data
            with parallel_backend('threading'):
                grid_search.fit(X_train, y_train)
            
            # Best estimator and parameters
            best_model = grid_search.best_estimator_
            print(f"Best Params: {grid_search.best_params_}")
            
            # Evaluate on the test set at default threshold=0.5
            probs_test = best_model.predict_proba(X_test)[:, 1]
            preds_test = best_model.predict(X_test)
            
            acc = accuracy_score(y_test, preds_test)
            prec = precision_score(y_test, preds_test, zero_division=0)
            rec = recall_score(y_test, preds_test, zero_division=0)
            f1 = f1_score(y_test, preds_test, zero_division=0)
            roc = roc_auc_score(y_test, probs_test)
            
            model_result = {
                'Model': model_name,
                'Best_Params': grid_search.best_params_,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'ROC_AUC': roc
            }
            results.append(model_result)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_test, y_test_pred, probs_test)

            # Track the best model and metrics in MLflow
            self.track_mlflow(best_model, classification_test_metric, model_name = model_name)

            save_object(f"final_model/{model_name}_final_model/model.pkl",best_model)

        # Convert to DataFrame and display
        results_df = pd.DataFrame(results)
        write_yaml_file(self.model_trainer_config.model_metrics_file_path, results_df.to_dict())        

        #model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                      metric_artifact=classification_test_metric)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #load the train and test data
            train_data = read_csv_data(train_file_path)
            test_data = read_csv_data(test_file_path)

            x_train, y_train = train_data.drop(columns=['Churn']), train_data['Churn']
            x_test, y_test = test_data.drop(columns=['Churn']), test_data['Churn']

            model_trainer_artifact=self.train_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)

        except Exception as e:
            raise ChurnPredictionException(e,sys) from e    





