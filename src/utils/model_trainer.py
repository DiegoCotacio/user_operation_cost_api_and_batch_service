import logging
import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
#from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import mlflow

"""
INDEX:
Definimos 2 clases: ModelTraining enfocada al entrenamiento y Hyperparameter_Optimization para generar mejores parametros

"""



class Hyperparameter_Optimization:

    """
    Class for doing hyperparameter optimization

    """

    def __init__(
        self, x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Initialize the class with the training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize_xgboost_regressor(self, trial: optuna.Trial) -> float:

        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 10.0),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        }
        reg = LGBMRegressor(**param)
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)

        return val_accuracy


class ModelTraining:

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Initialize the class with the training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def xgboost_trainer(self, fine_tuning: bool = True):

        logging.info("Started training XGBoost model.")
        try:
            if fine_tuning:
                  hy_opt = Hyperparameter_Optimization(self.x_train, self.y_train, self.x_test, self.y_test)
                  study = optuna.create_study(direction = "maximize")
                  study.optimize(hy_opt.optimize_xgboost_regressor, n_trials = 10)
                  trial = study.best_trial

                  n_estimators = trial.params["n_estimators"]
                  learning_rate = trial.params["learning_rate"]
                  max_depth = trial.params["max_depth"]

                  model = LGBMRegressor(
                    n_estimators = n_estimators,
                    learning_rate = learning_rate,
                    max_depth = max_depth,
                  )
                  model.fit(self.x_train, self.y_train)
                  return model
            
            else:
                  model = LGBMRegressor(
                      n_estimators=100, learning_rate=0.01, max_depth=20
                  )
                  model.fit(self.x_train, self.y_train)
                  return model
            
        except Exception as e:
                  logging.error("Error in training XGBoost model")
                  logging.error(e)
                  return None    