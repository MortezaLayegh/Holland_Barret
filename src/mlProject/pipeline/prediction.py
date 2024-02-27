import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject import logger
from mlProject.utils.common import create_directories





class PredictionPipeline:
    def __init__(self):
        self.preprocessor= joblib.load(Path('artifacts\data_transformation\preprocessor.joblib'))
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))


    def feature_eng_data_transform(self,data_path):
        data = pd.read_csv(data_path)

        data['Discount Percentage'] = ((data['Total Sales'] - data['Discounted Sales']) / data['Total Sales']) * 100
        data['Unique Items per Total Item'] = data['Unique Items'] / data['Total Items']
        data['Month'] = pd.to_datetime(data['Date']).dt.month
        logger.info("New feature created")

        data.drop(columns=['Customer ID', 'Transaction ID','Date'], inplace=True)
        logger.info("Useless columns were dropped")

        data['Month'] = data['Month'].astype(str)
        data['Loyalty Card'] = data['Loyalty Card'].astype(str)
        logger.info("Data types of 'Month' and 'Loyalty Card' were changed to string")
        return data

    
    def predict(self, data):
        #data = pd.read_csv(data_path)
        data = self.preprocessor.transform(data)
        prediction = self.model.predict(data)

        # Assuming y_new_pred is a 1-dimensional array containing the predictions
        predictions_df = pd.DataFrame({'Prediction': prediction})
        create_directories([Path('artifacts/predictions')])
        file_path = Path("artifacts/predictions/prediction.csv")
        # Save predictions to a CSV file
        predictions_df.to_csv(file_path, index=False)
