# Este modulo contiene las siguientes clases:
"""
1. Guardar predicciones

* Funciones para la construcción del Monitoring Dashboard:

2. Cargar datos de entrenamiento
3. Carga de resultados de inferencia recientes (last 3000)
4. Constructor del Reporte
5. Constructor del Dashboard

"""

#------- LIBRERIAS

# Internas
from src.connections.bq_conect import connect_to_bigquery

# Externas
from evidently import ColumnMapping
from evidently.metrics import (
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionPredictedVsActualPlot,
    RegressionErrorPlot,
    RegressionAbsPercentageErrorPlot,
    RegressionErrorDistribution,
    RegressionErrorNormality,
    RegressionTopErrorMetric
)
from evidently.metric_preset import TargetDriftPreset
from evidently.report import Report
import pandas as pd
from typing import List, Text
from google.cloud import bigquery
import logging
from evidently import ColumnMapping
# -------- Funciones de Almacenamiento y Carga de predicciones para Monitoring:


# Almacenamiento de datos del endpoint 1. Online predictions
   # Nota: Para Batch ya se encuentra configurada esta función dentro del pipeline de inferencia. 

def save_predictions_to_bigquery(df_final: pd.DataFrame):

    try:
    
        client = connect_to_bigquery()
        table_export_id= 'insurance_predictions_v2'
        dataset_export_id = 'ml_datasets'

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





# Loads out training/reference dataset


def load_reference_data() -> pd.DataFrame: #columns: List[Text]) -> pd.DataFrame:
    #train_file = "data/insurance.csv"
    train_file = "insurance.csv"
    reference_data = pd.read_csv(train_file)
    return reference_data


def load_current_data(window_size: int = 3000) -> pd.DataFrame:
    try:
        client = connect_to_bigquery()
        project_id = 'protean-fabric-386717'
        dataset_extract_id = "ml_datasets"
        table_extract_id = "insurance_predictions_v2"

        query = f"SELECT * FROM `{project_id}.{dataset_extract_id}.{table_extract_id}`\
        ORDER BY  date  DESC\
        LIMIT {window_size};\
        """
        current_data= client.query(query).to_dataframe() 
        return current_data
   
    except Exception as e:
        logging.error(e)
        raise e

#----------------- MODEL PERFORMANCE REPORT

def build_model_performance_report(
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: ColumnMapping) -> Text:

    model_performance_report = Report(metrics =[
        RegressionQualityMetric(),
        RegressionPredictedVsActualScatter(),
        RegressionPredictedVsActualPlot(),
        RegressionErrorPlot(),
        RegressionAbsPercentageErrorPlot(),
        RegressionErrorDistribution(),
        RegressionErrorNormality(),
        RegressionTopErrorMetric()
    ])
    model_performance_report.run(
        reference_data = reference_data,
        current_data = current_data,
        column_mapping = column_mapping)

    report_path = "reports/model_performance.html"
    model_performance_report.save_html(report_path)

    return report_path

    
#---------------------- DRIFT REPORT


def build_target_drift_report(   
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping
) -> Text:

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data = reference_data,
        current_data = current_data,
        column_mapping = column_mapping
    )
    report_path = 'reports/target_drift.html'
    target_drift_report.save_html(report_path)

    return report_path