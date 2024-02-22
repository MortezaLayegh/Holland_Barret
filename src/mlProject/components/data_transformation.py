import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from mlProject.entity.config_entity import DataTransformationConfig




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data = self.feature_eng_data_transform()  # Call feature_eng_data_transform upon initialization

    ## Note: You can add i want to create a new feature to the data and then split 
    # df['Discount Percentage'] = ((df['Total Sales'] - df['Discounted Sales']) / df['Total Sales']) * 100
    def feature_eng_data_transform(self):
        data = pd.read_csv(self.config.data_path)

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

    def train_test_splitting(self):
        X = self.data.drop('Incomplete Transaction', axis=1)
        y = self.data['Incomplete Transaction']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def get_data_transformer_object(self):
        '''
        Get data transformation object for preprocessing.
        '''

        # Define numerical and categorical features
        X = self.data.drop('Incomplete Transaction', axis=1)
        num_features = X.select_dtypes(exclude="object").columns
        cat_features = X.select_dtypes(include="object").columns

        # Define a pipeline for processing numeric features
        numeric_processor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler())
            ]
        )

        # Define a pipeline for processing categorical features
        categorical_processor = Pipeline(
            steps=[
                ("Imputer", SimpleImputer(strategy='most_frequent')),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        logger.info(f"Categorical columns: {cat_features}")
        logger.info(f"Numerical columns: {num_features}")

        # Combine numeric and categorical processors
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numeric_processor, num_features),
                ("categorical", categorical_processor, cat_features)
            ]
        )

        return preprocessor
    
    def initiate_data_transformation(self):
        preprocessing_obj = self.get_data_transformer_object()
        X_train, X_test, y_train, y_test = self.train_test_splitting()
        X_train = preprocessing_obj.fit_transform(X_train)
        X_test = preprocessing_obj.transform(X_test)

        # Combine input features with target feature
        train_arr = np.c_[X_train, y_train]
        test_arr = np.c_[X_test, y_test]

        train_df = pd.DataFrame(train_arr)
        test_df = pd.DataFrame(test_arr)

        train_df.to_csv(os.path.join(self.config.root_dir, "train_df.csv"), index=False)
        test_df.to_csv(os.path.join(self.config.root_dir, "test_df.csv"), index=False)

        logger.info("data into training and test sets (scalled and imputed)")
        logger.info(X_train.shape)
        logger.info(y_train.shape)

        print(X_train.shape)
        print(y_train.shape)

        joblib.dump(preprocessing_obj, os.path.join(self.config.preprocessor_obj_file_path))

        