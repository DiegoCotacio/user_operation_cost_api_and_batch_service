## TRANSFORMADOR AUXILIAR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from typing import Union
from sklearn.model_selection import train_test_split

class PreprocessData:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data.drop_duplicates(inplace=True)
            return data
        
        except Exception as e:
            logging.error(e)
            raise e
    

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            # numerical columns
            categorical_cols = df.select_dtypes(include='object').columns
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            
            # categorical columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            
            return df
        
        except Exception as e:
            logging.error(e)
            raise e
    

    def map_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            mapping = {
              'sex': {'female': 1, 'male':0},
              'smoker': {'yes': 1, 'no':0},
              'region': {'southwest': 0, 'southeast': 0.3, 'northwest':0.6, 'northeast': 1}
              }
            df.replace(mapping, inplace=True)
            return df
        
        except Exception as e:
            logging.error(e)
            raise e
    
    

    def normalize_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            numeric_cols = ['bmi', 'age', 'children']
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            return df
        
        except Exception as e:
            logging.error(e)
            raise e
    


class DataSpliter:
    
    def divide_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        It divides the data into train and test data.
        """
        try:
            X = df.drop("charges", axis=1)
            y = df["charges"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(e)
            raise e