import os
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    """
    A class to evaluate machine learning models.

    Attributes:
        config (ModelEvaluationConfig): Configuration object containing model evaluation settings.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes ModelEvaluation class with provided configuration.

        Args:
            config (ModelEvaluationConfig): Configuration object containing model evaluation settings.
        """
        self.config = config
        
    def evaluate_clf(self,true, predicted):
        '''
        This function takes in true values and predicted values
        Returns: Accuracy, F1-Score, Precision, Recall, Roc-auc Score
        '''
        acc = accuracy_score(true, predicted) # Calculate Accuracy
        f1 = f1_score(true, predicted) # Calculate F1-score
        precision = precision_score(true, predicted) # Calculate Precision
        recall = recall_score(true, predicted)  # Calculate Recall
        roc_auc = roc_auc_score(true, predicted) #Calculate Roc
        return acc, f1 , precision, recall, roc_auc

    def log_into_mlflow(self):
        """
        Logs model evaluation metrics and model artifacts into MLflow.
        """
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(X_test)

            (acc, f1, precision,recall,roc_auc) = self.evaluate_clf(y_test, predicted_qualities)
            
            # Saving metrics as local
            scores = {"acc": acc, "f1": f1, "precision": precision, "recall": recall, "roc_auc": roc_auc}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("acc", acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)
        
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="GBMClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")
