import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject import logger
from mlProject.utils.common import create_directories

class PredictionPipeline:
    """
    A class representing the prediction pipeline.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes PredictionPipeline class.
        """
        # Load preprocessor and model from disk
        self.preprocessor= joblib.load(Path('artifacts\data_transformation\preprocessor.joblib'))
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def feature_eng_data_transform(self,data_path):
        """
        Transform the input data with feature engineering steps.

        Args:
            data_path (str): The path to the input data file.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        data = pd.read_csv(data_path)

        # Feature engineering steps
        data['Discount Percentage'] = ((data['Total Sales'] - data['Discounted Sales']) / data['Total Sales']) * 100
        data['Unique Items per Total Item'] = data['Unique Items'] / data['Total Items']
        data['Month'] = pd.to_datetime(data['Date']).dt.month
        logger.info("New feature created")

        # Drop unnecessary columns
        data.drop(columns=['Customer ID', 'Transaction ID','Date'], inplace=True)
        logger.info("Useless columns were dropped")

        # Convert data types
        data['Month'] = data['Month'].astype(str)
        data['Loyalty Card'] = data['Loyalty Card'].astype(str)
        logger.info("Data types of 'Month' and 'Loyalty Card' were changed to string")
        return data

    def predict(self, data):
        """
        Make predictions on input data.

        Args:
            data (pandas.DataFrame): The input data to make predictions on.

        Returns:
            None
        """
        # Transform data using preprocessor
        data = self.preprocessor.transform(data)
        # Make predictions using the model
        prediction = self.model.predict(data)

        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({'Prediction': prediction})
        # Create directory for saving predictions
        create_directories([Path('artifacts/predictions')])
        # Define file path for saving predictions
        file_path = Path("artifacts/predictions/prediction.csv")
        # Save predictions to a CSV file
        predictions_df.to_csv(file_path, index=False)
        logger.info(f"Predictions saved to {file_path}")