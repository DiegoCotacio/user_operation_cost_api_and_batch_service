from fastapi import FastAPI, UploadFile, File
import numpy as np
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import uvicorn
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from mangum import Mangum
import os

app = FastAPI()
handler = Mangum(app)


#---------------- PASOS DE PREPROCESAMIENTO DE INPUTS -----------------------------

def map_categorical_features(input_df):
        mapping = {
          'sex': {'female': 1, 'male':0},
          'smoker': {'yes': 1, 'no':0},
          'region': {'southwest': 0, 'southeast': 0.3, 'northwest':0.6, 'northeast': 1}}
        input_df.replace(mapping, inplace=True)
        return input_df

def normalize_numeric_features(input_df):
    input_df = input_df.copy()  # Copia el DataFrame
    numeric_cols = ['bmi', 'age', 'children']
    scaler = MinMaxScaler()
    input_df[numeric_cols] = scaler.fit_transform(input_df[numeric_cols])
    return input_df

def impute_missing_values(input_df):
    # Variables categóricas
    categorical_cols = input_df.select_dtypes(include='object').columns
    #categorical_cols = categorical_cols.drop('churn')  # Excluir 'churn'
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    input_df[categorical_cols] = categorical_imputer.fit_transform(input_df[categorical_cols])

    # Variables numéricas
    numeric_cols = input_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_imputer = SimpleImputer(strategy='mean')
    input_df[numeric_cols] = numeric_imputer.fit_transform(input_df[numeric_cols])

    return input_df

def remove_duplicates(input_df):
    input_df.drop_duplicates(inplace=True)
    return input_df

def format_dtypes(input_df):
    input_df['sex'] = input_df['sex'].astype(str)
    input_df['smoker'] = input_df['smoker'].astype(str)
    input_df['region'] = input_df['region'].astype(str)
    input_df['bmi'] = input_df['bmi'].astype(float)
    input_df['age'] = input_df['age'].astype(int)
    input_df['children'] = input_df['children'].astype(int)
    return input_df

#-----------------**** CREACION DE LA APP  *****----------------------------------

# Ruta del archivo de modelo
model_path = 'models/xgb_predictor.joblib'
# Cargar el modelo entrenado desde el archivo
model = load(model_path)

class Input_Data(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


@app.post("/online_predict")
async def predict(input_data: Input_Data):
    #input_dict = jsonable_encoder(input_data)
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    input_df = format_dtypes(input_df)
    input_df = normalize_numeric_features(input_df)
    input_df = map_categorical_features(input_df)
    output = model.predict(input_df)[0].item()
    return {'prediction': output}

## BATCH PREDICTION USING CSV

@app.post("/batch_predict")
async def batch_predict(file: UploadFile):

    input_df = pd.read_csv(file.file)

    # Crear una copia de los datos iniciales para preservarlos
    original_df = input_df.copy()

    input_df = remove_duplicates(input_df)
    input_df = impute_missing_values(input_df)
    input_df = normalize_numeric_features(input_df)
    input_df = map_categorical_features(input_df)

    # Realizar la predicción utilizando los datos preprocesados
    predictions = model.predict(input_df)

    # Agregar la etiqueta de predicción a los datos iniciales
    original_df['prediction_label'] = predictions
    
    return original_df.to_dict(orient='records')
    
    """
    result = pd.DataFrame()
    result['age'] = input_df['age']
    result['sex'] = input_df['sex']
    result['bmi'] = input_df['bmi']
    result['children'] = input_df['children']
    result['smoker'] = input_df['smoker']
    result['region'] = input_df['region']
    result['prediction_label'] = predictions"""

    #return result.to_dict(orient='records')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)