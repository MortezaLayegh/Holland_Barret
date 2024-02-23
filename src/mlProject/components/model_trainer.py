
from mlProject.entity.config_entity import ModelTrainerConfig
import pandas as pd
import os
from mlProject import logger
import joblib
from xgboost import XGBClassifier




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self):
        logger.info("Initiating model training")
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.iloc[:, :-1]
        X_test = test_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        y_test = test_data.iloc[:, -1]


        xgb = XGBClassifier( n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, 
                            learning_rate=self.config.learning_rate, random_state=self.config.random_state,
                            scale_pos_weight=self.config.scale_pos_weight, min_child_weight=self.config.min_child_weight, 
                            subsample=self.config.subsample)
        
        xgb.fit(X_train, y_train)

        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_name))