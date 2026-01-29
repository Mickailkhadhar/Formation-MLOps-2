import os
import sys
from datetime import timedelta, datetime

import pendulum
from airflow.decorators import dag, task

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dags.config import GENERATED_DATA_PATH, DATA_FOLDER, PREDICTIONS_FOLDER, MODEL_PATH
from formation_indus_ds_avancee.feature_engineering import prepare_features_with_io
from formation_indus_ds_avancee.train_and_predict import predict_with_io


@dag(
    default_args={'owner': 'airflow'},
    schedule=timedelta(minutes=2),
    start_date=pendulum.today('UTC').add(hours=-1),
)
def predict():

    @task
    def prepare_features_with_io_task():
        features_path = os.path.join(
            DATA_FOLDER, f'prepared_features_{datetime.now()}.parquet'
        )
        prepare_features_with_io(
            data_path=GENERATED_DATA_PATH,
            features_path=features_path,
            training_mode=False
        )
        return features_path

    @task
    def predict_with_io_task(features_path):
        predict_with_io(
            features_path=features_path,
            model_path=MODEL_PATH,
            predictions_folder=PREDICTIONS_FOLDER
        )

    features_path = prepare_features_with_io_task()
    predict_with_io_task(features_path=features_path)

    @task
    def get_latest_engie_data_file():
        return GENERATED_DATA_PATH

    @task
    def prepare_features_from_engie_task(engie_file_path):
        features_path = os.path.join(
            DATA_FOLDER, f'prepared_features_from_engie_{datetime.now()}.parquet'
        )
        prepare_features_with_io(
            data_path=engie_file_path,
            features_path=features_path,
            training_mode=False
        )
        return features_path

    engie_file_path = get_latest_engie_data_file()
    features_path_from_engie = prepare_features_from_engie_task(engie_file_path)
    predict_with_io_task(features_path=features_path_from_engie)

predict_dag = predict()
