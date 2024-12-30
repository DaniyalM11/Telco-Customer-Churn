from ChurnPrediction.entity.artifact_entity import ClassificationMetricArtifact
from ChurnPrediction.exception.exception import ChurnPredictionException
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    roc_auc_score,
    make_scorer
)
import sys

def get_classification_score(y_true, y_pred, y_probs,zero_division=0) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred, zero_division=0)
        model_recall_score = recall_score(y_true, y_pred, zero_division=0)
        model_precision_score = precision_score(y_true, y_pred, zero_division=0)
        model_accuracy_score = accuracy_score(y_true, y_pred)
        model_roc_auc_score = roc_auc_score(y_true, y_probs)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            recall_score=model_recall_score,
            precision_score=model_precision_score,
            accuracy_score=model_accuracy_score,
            roc_auc_score=model_roc_auc_score
        )
        return classification_metric
    except Exception as e:
        raise ChurnPredictionException(e,sys) from e