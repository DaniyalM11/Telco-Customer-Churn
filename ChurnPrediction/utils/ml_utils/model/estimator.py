import os
import sys
from ChurnPrediction.exception.exception import ChurnPredictionException
from ChurnPrediction.logging.logger import logging

from ChurnPrediction.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

class ChurnPredictionModel:
    def __init__(self,preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise ChurnPredictionException(e,sys) from e

    def predict(self, X):
        try:
            X_transform = self.preprocessor.transform(X)
            y_hat = self.model.predict(X_transform)
            return y_hat
        except Exception as e:
            raise ChurnPredictionException(e,sys) from e