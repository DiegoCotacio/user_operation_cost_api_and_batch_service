from sklearn.base import RegressorMixin
import logging
import pandas as pd
import numpy as np
from prefect import task, flow
import mlflow
import xgboost as xgb
import os
import joblib
import optuna
import xgboost as xgb
from lightgbm import LGBMRegressor
from urllib.parse import urlparse
#--------- Custom packages
from utils.preprocessor import PreprocessData, DataSpliter
from utils.evaluator import Evaluation
from utils.model_trainer import ModelTraining, Hyperparameter_Optimization
from utils.reports import run_data_integrity_report,run_validation_train_test_split_report, run_validate_model_performance
# Crear o establecer el experimento deseado


#************** START THE TRAINING PIPELINE  *******************

# -------------------- DATA INGESTION STEP 

class IngestData:
    
    def __init__(self) -> None:
       """ Initialize the data ingestion class"""
       pass

    def  get_data(self) -> pd.DataFrame:
         data = pd.read_csv("./data/insurance.csv")
         return data


@task
def ingest_data() -> pd.DataFrame:
    """
    Args: None
    Returns: df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        data = ingest_data.get_data()
        data_report_path = run_data_integrity_report(data)

        if data_report_path:
            mlflow.log_artifact(data_report_path)

        return data
    except Exception as e:
        logging.error(e)
        raise e
  
  # --------------------DATA CLEANING SETP

@task
def transform_data(data: pd.DataFrame):
    """ Es una clase que preprocesa y divide los datos """
    try:
        logging.info(f"data type: {type(data)}")
        preprocessor = PreprocessData()  # Crear una instancia de PreprocessData
        logging.info("Preprocessor instance created successfully")
        df = preprocessor.remove_duplicates(data)
        df = preprocessor.impute_missing_values(df)
        df  = preprocessor.map_categorical_features(df)
        df = preprocessor.normalize_numeric_features(df)
        df_proc = df.fillna(0)

        data_spliter = DataSpliter()  # Crear una instancia de DataSpliter
        x_train, x_test, y_train, y_test = data_spliter.divide_data(df_proc)
        
        train_report_path = run_validation_train_test_split_report(x_train, y_train, x_test, y_test)
        if train_report_path:
            mlflow.log_artifact(train_report_path)

        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logging.error(e)
        raise e

#-------------------------- MODEL TRAINING STEP

@task
def train_model(
    x_train : pd.DataFrame,
    x_test : pd.DataFrame,
    y_train : pd.Series,
    y_test : pd.Series
):
    try:
        hy_opt = Hyperparameter_Optimization(x_train, y_train, x_test, y_test)
        study = optuna.create_study(direction = "maximize")
        study.optimize(hy_opt.optimize_xgboost_regressor, n_trials = 10)
        trial = study.best_trial

        n_estimators = trial.params["n_estimators"]
        mlflow.log_param('n_estimators', n_estimators)

        learning_rate = trial.params["learning_rate"]
        mlflow.log_param('learning_rate', learning_rate)

        max_depth = trial.params["max_depth"]
        mlflow.log_param('max_depth', max_depth)

        model = LGBMRegressor(
                n_estimators = n_estimators,
                learning_rate = learning_rate,
                max_depth = max_depth,
                )
            
        model.fit(x_train, y_train)
            
        #Save de model
        models_folder = "../models" 
        os.makedirs(models_folder, exist_ok=True)
        model_file = os.path.join(models_folder, "ligthgmb_model_v1.joblib")
        joblib.dump(model, model_file)

        # Registro del modelo en MLflow
        mlflow.log_artifact(model_file)

        #model validation
        model_performance_report_path = run_validate_model_performance(x_train, y_train, x_test, y_test, model)
        if model_performance_report_path:
            mlflow.log_artifact(model_performance_report_path)

        return model
        
    except Exception as e:
        logging.error(e)
        raise e

#--------------------- MODEL EVALUATION STEP

@task
def evaluation(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series):
    
    "Args: model, x_test, y_test  and Returns: r2_score and rmse"

    try:
        prediction = model.predict(x_test)
        evaluation = Evaluation()

        r2_score = evaluation.r2_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        mse = evaluation.mean_squared_error(y_test, prediction)
        mlflow.log_metric("mse", mse)

        rmse = evaluation.root_mean_squared_error(y_test, prediction)  # Passing y_true and y_pred
        mlflow.log_metric("rmse", rmse)

        return mse, rmse
    
    except Exception as e:
        logging.error(e)
        raise e

#- ------------------------- TRAINING PIPELINE
@flow(name="Training pipeline for batch inference service")
def train_pipeline():

    remote_server_uri = "https://dagshub.com/diego.cotacio/CI-CD_deployment_AWSLambda_FastAPI_Docker.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    experiment_name = "Training Pipeline for LigthGBM Model V1"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name = "LightGBM Experiment Global"):
        df = ingest_data()
        x_train, x_test, y_train, y_test = transform_data(df)
        model = train_model(x_train, x_test, y_train, y_test)
        mse, rmse = evaluation(model, x_test, y_test)


if __name__ == "__main__":
 train_pipeline()