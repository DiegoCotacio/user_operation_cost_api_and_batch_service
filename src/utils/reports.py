import pandas as pd
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.suites import train_test_validation
from deepchecks.tabular.suites import model_evaluation

import os
from datetime import datetime

#------------------ DATA INTEGRITY REPORT

def run_data_integrity_report(data):
    try:

        # Create Deepchecks Datasets
        dataset= Dataset(data, label= 'charges', cat_features=['sex', 'smoker', 'region'])
        suite_result = data_integrity().run(dataset)
        
        # Save Deepchecks Suite Result as HTML Report
        html_report_name = "Data_Integrity_Report_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".html"
        output_folder = "reports"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, html_report_name)
        suite_result.save_as_html(output_path)

        return output_path
    except Exception as e:
        print("Error al generar el informe Data Integrity:", e)
        return None
    
#------------------ TRAIN TEST SPLIT REPORT

def run_validation_train_test_split_report(x_train, y_train, x_test, y_test):
    try:

        # Create Deepchecks Datasets
        train = Dataset(x_train, label=y_train,cat_features=['sex', 'smoker', 'region'])
        test = Dataset(x_test, label=y_test, cat_features=['sex', 'smoker', 'region'])

        # Run Deepchecks train_test_validation
        suite_result = train_test_validation().run(train_dataset=train, test_dataset=test)

        # Save Deepchecks train_test_validation Result as HTML Report
        html_report_name = "TrainTestSplit_Validation_Report_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".html"
        output_folder = "reports"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, html_report_name)
        suite_result.save_as_html(output_path)

        return output_path
    
    except Exception as e:
        print("Error al generar el informe de validacion de Train/Test split:", e)
        return None 


#------------------ DATA MODEL VALIDATION


def run_validate_model_performance(x_train, y_train, x_test, y_test, model):
    try:

        # Create Deepchecks Datasets
        train = Dataset(x_train, label=y_train, cat_features=['sex', 'smoker', 'region'])
        test = Dataset(x_test, label=y_test, cat_features=['sex', 'smoker', 'region'])

        # Run Deepchecks Full Suite
        suite_result = model_evaluation().run(train_dataset=train, test_dataset=test, model = model)

        # Save Deepchecks Suite Result as HTML Report
        html_report_name = "ModelPerfomance_Validation_Report_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".html"
        output_folder = "reports"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, html_report_name)
        suite_result.save_as_html(output_path)

        return output_path
    except Exception as e:
        print("Error al generar el informe de validacion de Model Performance:", e)
        return None 



def run_deepchecks_and_save_report(x_train, y_train, x_test, y_test, model):
    try:

        # Create Deepchecks Datasets
        ds_train = Dataset(x_train, label=y_train)
        ds_test = Dataset(x_test, label=y_test)

        # Run Deepchecks Full Suite
        suite = full_suite()
        suite_result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)

        # Save Deepchecks Suite Result as HTML Report
        html_report_name = "DeepCheckReport_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".html"
        output_folder = "reports"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, html_report_name)
        suite_result.save_as_html(output_path)

        return output_path
    except Exception as e:
        print("Error al generar el informe de Deepchecks:", e)
        return None 