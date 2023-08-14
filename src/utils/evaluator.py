import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation:

    #- -------------------- CLASS CONSTRUCTOR:

    def __init__(self) -> None:
        pass

    #- -------------------- MEAN_SQUARED_ERROR ESTIMATOR METHOD


    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
       
        try:
            logging.info("Entered the mean_squaerd_error methos of the Evaluation Class")
            mse = mean_squared_error(y_true, y_pred)
            
            logging.info("The mean squeared error value is: " + str(mse))

            return mse
        
        except Exception as e:
            logging.info("Exception ocurred in mean squeared error method of the Evaluation class. Exception message:  " + str(e))
            logging.info("Exited the mean squeared error method of the Evaluation class")
            raise Exception()
    
    #- --------------------- R2 SCORE ESTIMATOR METHOD

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        try:
            logging.info("Entered the r2_score method of the Evaluation Class")

            r2 = r2_score(y_true, y_pred)
            
            logging.info("The r2_score value is:" + str(r2))
            logging.info("Exited the r2_score method of the Evaluation class")

            return r2
        
        except Exception as e:
            logging.info("Exception ocurred in r2_score method of the Evaluation class. Exception message:  " + str(e))
            logging.info("Exited the r2_score method of the Evaluation class")
            raise Exception()


    #------------------- ROOT MEAN SQUARED ERROR (RMSE) ESTIMATOR METHOD 
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        try:
            logging.info("Entered the root_mean_squared_error method of the Evaluation Class")

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            logging.info("The root_mean_squared_error value is:" + str(rmse))
            logging.info("Exited the root_mean_squared_error method of the Evaluation class")

            return rmse
        
        except Exception as e:
            logging.info("Exception ocurred in root_mean_squared_error method of the Evaluation class. Exception message:  " + str(e))
            logging.info("Exited the root_mean_squared_erro method of the Evaluation class")
            raise Exception()
                         
