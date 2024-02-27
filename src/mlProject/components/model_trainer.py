
from mlProject.entity.config_entity import ModelTrainerConfig
import pandas as pd
import os
from mlProject import logger
import joblib
from sklearn.ensemble import  GradientBoostingClassifier




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

        GBM = GradientBoostingClassifier(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, 
                            learning_rate=self.config.learning_rate, random_state=self.config.random_state,
                            subsample=self.config.subsample, min_samples_split=self.config.min_samples_split,
                            min_samples_leaf=self.config.min_samples_leaf)



        GBM.fit(X_train, y_train)

        joblib.dump(GBM, os.path.join(self.config.root_dir, self.config.model_name))