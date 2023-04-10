
import sys
import traceback
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig():
    preprocessr_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransorformationConfig()
    
    def get_data_transformer_object(self):

        try:
            numerical_columns    = ['writing_score','reading']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'

            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                
                ]
            )


            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))


                ]
            )


            logging.info(f'Categorical columns : {categorical_columns}')
            logging.info(f'Numerical columns : {numerical_columns}')



            preproceesor =  ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)

                ]
            )

            return preproceesor
        
        
        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and Test Data')
            logging.info('Obtaining preprocessor object')

            preprocessor_obj = self.get_data_transformer_object()
            target_colums_name='math_score'
            numerical_columns =['writing_sco']