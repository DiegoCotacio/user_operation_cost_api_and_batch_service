from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import(
    HTMLResponse,
    JSONResponse,
    Response,
    FileResponse
)
import numpy as np
import json
#from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Callable, Text
import pandas as pd
import uvicorn
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from mangum import Mangum
import os
import logging


from evidently import ColumnMapping
from api_utils.monitoring_utils import (
    load_current_data,
    load_reference_data,
    save_predictions_to_bigquery,
    build_model_performance_report,
    build_target_drift_report #, get_column_mapping,
)

logging.basicConfig(
    level=logging.INFO,
    format='FASTAPI_APP - %(asctime)s - %(levelname)s - %(message)s'
)

#----------------------------------------------------------------

app = FastAPI()
handler = Mangum(app)


#---------------- LIST OF FIXED PREPROCESSING STEPS -----------------------------

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

#-----------------**** MODEL ENPOINTS  *****----------------------------------

model_path = 'models/xgb_predictor.joblib'
# Cargar el modelo entrenado desde el archivo
model = load(model_path)

#-----------------------------------

@app.get('/')
def index() -> HTMLResponse:
    return HTMLResponse('<h1><i>Evidently + FastAPI</i></h1>')



#-------------------------- Endpoint 1. Online prediction
class Features(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post("/online_predict")
async def predict(
    response: Response,
    features_item: Features,
    background_tasks: BackgroundTasks = BackgroundTasks()) -> float:

    try:
        #input_dict = jsonable_encoder(input_data)
        input_dict = features_item.dict()
        input_df = pd.DataFrame([input_dict])
        original_row = input_df.copy()
        input_df = format_dtypes(input_df)
        #preprocess
        input_df = normalize_numeric_features(input_df)
        input_df = map_categorical_features(input_df)
        #generate predictions
        prediction = model.predict(input_df)[0].item()
        #save monitoring input

        original_row['prediction'] = prediction
        background_tasks.add_task(save_predictions_to_bigquery, original_row)

        return prediction

    except Exception as e:
        response.status_code = 500
        logging.error(e, exc_info=True)
        return JSONResponse(content={'error_msg': str(e)})




#-------------------------- Endpoint 2. Online prediction for uploaded csv

@app.post("/batch_predict")
async def batch_predict(file: UploadFile):

    input_df = pd.read_csv(file.file)

    # Crear una copia de los datos iniciales para preservarlos
    original_df = input_df.copy()
    input_df = format_dtypes(input_df)
    input_df = remove_duplicates(input_df)
    input_df = impute_missing_values(input_df)
    input_df = normalize_numeric_features(input_df)
    input_df = map_categorical_features(input_df)
    
    #model: Callable = model_loader.get_model()
    # Realizar la predicción utilizando los datos preprocesados
    predictions = model.predict(input_df)

    # Agregar la etiqueta de predicción a los datos iniciales
    original_df['prediction'] = predictions
    
    return original_df.to_dict(orient='records')




#-------------------------- Endpoint 3. Batch prediction for inference pipeline

@app.post("/batch_predict_pipeline")
async def batch_predict_pipeline(input_list: List[Features]):

    output_list = [input_data.dict() for input_data in input_list]
    input_df = pd.DataFrame(output_list)
    input_df = format_dtypes(input_df)
    original_df = input_df.copy()

    input_df = remove_duplicates(input_df)
    input_df = impute_missing_values(input_df)
    input_df = normalize_numeric_features(input_df)
    input_df = map_categorical_features(input_df)

    #model: Callable = model_loader.get_model()

    # Realizar la predicción utilizando los datos preprocesados
    predictions = model.predict(input_df)

    # Agregar la etiqueta de predicción a los datos iniciales
    original_df['prediction'] = predictions

    prediction_list = original_df.to_dict(orient='records')

    return prediction_list




#-----------------**** MONITORING ENDPOINTS *****----------------------------------

# config to column mapping parameter for Evidently
def get_column_mapping() -> ColumnMapping:
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    column_mapping.prediction = None #'prediction' 
    column_mapping.numerical_features = ['age', 'bmi', 'children']
    column_mapping.categorical_features = ['sex', 'smoker', 'region']
    return column_mapping

# -------- Model Performance Dashboard

@app.get('/monitor-model')
def monitor_model_performance(window_size: int = 3000) -> FileResponse:

    logging.info('Read current data')
    current_data: pd.DataFrame = load_current_data(window_size)
    current_data.drop(columns=['user_cod', 'date'], inplace=True)
    current_data.rename(columns={'prediction': 'target'}, inplace=True)

    logging.info('Read reference data')
    reference_data = load_reference_data()#columns=DATA_COLUMNS['columns'])
    reference_data.rename(columns={'charges': 'target'}, inplace=True)
    
    logging.info('Build report')
    column_mapping: ColumnMapping = get_column_mapping()
    report_path: Text = build_model_performance_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping)
    

    logging.info('Return report as html')
    return FileResponse(report_path)


# -------- Data Drift Dashboard

@app.get('/monitor-target')
def monitor_target_drift(window_size: int = 3000) -> FileResponse:

    logging.info('Read current data')
    current_data: pd.DataFrame = load_current_data(window_size)
    current_data.drop(columns=['user_cod', 'date'], inplace=True)
    current_data.rename(columns={'prediction': 'target'}, inplace=True)

    logging.info('Read reference data')
    reference_data = load_reference_data()
    reference_data.rename(columns={'charges': 'target'}, inplace=True)

    logging.info('Build report')
    column_mapping: ColumnMapping = get_column_mapping()
    report_path: Text = build_target_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping= column_mapping
        )
    

    logging.info('Return report as html')
    return FileResponse(report_path)


   
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
