import pandas as pd
import requests
from datetime import datetime
from prefect import flow, task
from connections.bq_conect import connect_to_bigquery
from google.cloud import bigquery
import logging

"""
Batch inference pipeline design or EPL Pipeline.

  Components:
  1. Extract data from feature store (Bigquery simulates a 2 layer of data)
  2. Generate predictios (using a API endpoint of a model deploy on AWS Lambda)
  3. Return predictions to a Bigquery Table
  * 4. Visualization of predictions on Google Data Studio.

"""

# ------------------------ -EXTRACT DATA:

@task
def extract_dataframe_from_bigquery(project_id, dataset_extract_id, table_extract_id) -> pd.DataFrame:  
   
   """ Función que realiza una conexion a una tabla de BQ y extrae todos los datos del dia actual """
   
   try:
       client = connect_to_bigquery()
       current_date = datetime.now().strftime("%Y-%m-%d") 
       query = f"SELECT * FROM `{project_id}.{dataset_extract_id}.{table_extract_id}` WHERE DATE_TRUNC(date, DAY) = '{current_date}'" 
       df = client.query(query).to_dataframe() 
       return df
   
   except Exception as e:
        logging.error(e)
        raise e
  
#-------------------------- GENERATE PREDICTIONS

@task
def generate_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:

    """ Crea copia del df original, pasa datos al endpoint en JSON (dict list), recupera predicciones y las une al df original """

    try:
        df_original = df
        input_df = df_original.drop(['user_cod', 'date'], axis=1)
        input_list = input_df.to_dict(orient='records')
        
        API_ENDPOINT = "http://127.0.0.1:8000/batch_predict_pipeline"
        response = requests.post(API_ENDPOINT, json=input_list)
        if response.status_code == 200:
             predictions_output = response.json()
             predictions_df = pd.DataFrame(predictions_output)
        else:
             print("Error al realizar las estimaciones") 
        
        df_final = pd.concat([df_original[['user_cod', 'date']], predictions_df], axis=1)  
        return df_final
    
    except Exception as e:
        logging.error(e)
        raise e
    
#------------------------- LOAD PREDICTIONS

@task
def save_predictions_to_bigquery(df_final: pd.DataFrame,
                                 table_export_id: str,
                                 dataset_export_id: str):

    try:
    
        client = connect_to_bigquery()
        table_ref = client.dataset(dataset_export_id).table(table_export_id)
        table_exists = client.get_table(table_ref) 

        if table_exists is None: 
           schema = []
           for column_name, column_type in df_final.dtypes.items():
               schema.append(bigquery.SchemaField(name=column_name, field_type=column_type.name))

        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        print(f"Se ha creado la tabla {dataset_export_id}.{table_export_id} en BigQuery.")

        job_config = bigquery.LoadJobConfig()
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        job = client.load_table_from_dataframe(df_final, table_ref, job_config=job_config)
        job.result()

    except Exception as e:
        logging.error(e)
        raise e
    
#------------------------- EPL PIPELINE

@flow(name="Insurance batch inference pipeline")
def generate_insurance_predictions():
    
    try:
       project_id = 'protean-fabric-386717'
       dataset_extract_id = "ml_datasets"
       table_extract_id = "insurance_synt"
       df= extract_dataframe_from_bigquery(project_id, dataset_extract_id, table_extract_id)
       
       df_final = generate_batch_predictions(df)
       table_export_id= 'insurance_predictions_v2'
       dataset_export_id = 'ml_datasets'
       
       load_completed = save_predictions_to_bigquery(df_final, table_export_id, dataset_export_id)
       
       return load_completed
    
    except Exception as e:
        logging.error(e)
        raise e

#------------------------ RUN BATCH PREDICTION PIPELINE

if __name__ == '__main__':
 generate_insurance_predictions()

