## General Functions
from prefect import flow, task
import pandas as pd
import numpy as np
import random
from connections.bq_conect import connect_to_bigquery
from google.cloud import bigquery
import datetime

"""
FEATURE PIPELINE:

Componentes ETL:
Extract: generacion sintetica de datos
Transform: no aplica
Load: carga de datos a Bigquery.
"""

# Extract: Generacion datos sinteticos

@task
def generate_dataframe():
    # Configuración de las variables
    num_rows = 10

    numeric_variables = {
        'age': {'min': 20, 'mean': 35, 'max': 70},
        'bmi': {'min': 15.96, 'mean': 30.4, 'max': 53.3},
        'children': {'min': 0, 'mean': 1, 'max': 5}}

    categorical_variables = {
        'smoker': {'classes': ['False', 'True'], 'distribution': [80, 20]},
        'region': {'classes': ['southeast', 'southwest','northwest','northeast'], 'distribution': [25, 25, 25, 25]},
        'sex': {'classes': ['False', 'True'], 'distribution': [50, 50]}}

    # Generación de los datos
    data = {}

    idies = np.array([str(''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=6)))
                 for _ in range(num_rows)])
    
    data['user_cod'] = idies

    current_date = datetime.date.today()
    data['date'] = current_date

    for variable, info in categorical_variables.items():
        classes = info['classes']
        distribution = info['distribution']
        data[variable] = np.random.choice(classes, size=num_rows, p=[p/100 for p in distribution])


    for variable, info in numeric_variables.items():
        min_val = info['min']
        mean_val = info['mean']
        max_val = info['max']
        num_rows = 10  # Número de filas de datos que deseas generar

        if variable == 'bmi':
            data[variable] = np.random.normal(loc=mean_val, scale=(max_val - min_val) / 6, size=num_rows).round(2)
        else:
            data[variable] = np.random.normal(loc=mean_val, scale=(max_val - min_val) / 6, size=num_rows).astype(int)
            data[variable] = np.clip(data[variable], min_val, max_val)
        
    df = pd.DataFrame(data)
    df['user_cod'] = df['user_cod'].astype(str)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['age'] = df['age'].astype(int)
    df['sex'] = df['sex'].astype(str)
    df['bmi'] = df['bmi'].astype(float)
    df['children'] = df['children'].astype(int)
    df['smoker'] = df['smoker'].astype(str)
    df['region'] = df['region'].astype(str)
    
    new_order = ['user_cod','date','age','sex','bmi','children','smoker','region']
    df = df.reindex(columns=new_order)
    
    return df



@task
def ingest_or_create_to_bigquery(df, table_name, dataset_id):
    #configurar cliente de bigqury

    client = connect_to_bigquery()
    table_ref = client.dataset(dataset_id).table(table_name)

    # Verificar si la tabla existe
   # table_exists = client.get_table(table_ref, retry=retry.Retry(deadline=30))
    
    # Verificar si la tabla existe
    table_exists = client.get_table(table_ref)
    
    if table_exists is None:
        # Crear la tabla si no existe
        table = bigquery.Table(table_ref)
        schema = []
        for column_name, column_type in df.dtypes.items():
            schema.append(bigquery.SchemaField(name=column_name, field_type=column_type.name))
        table.schema = schema
        table = client.create_table(table)
        print(f"Se ha creado la tabla {dataset_id}.{table_name} en BigQuery.")
    
    # Cargar los datos del DataFrame en la tabla
    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

@flow(name="Synt insurance data generator x Bigquery")
def generate_insurance_data_bigquery():
    table_name = 'insurance_synt'
    dataset_id = 'ml_datasets'
    df= generate_dataframe()
    load_completed = ingest_or_create_to_bigquery(df, table_name, dataset_id)
    return load_completed

if __name__ == '__main__':
 generate_insurance_data_bigquery()