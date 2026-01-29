import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model_with_io(features_path: str, model_registry_folder: str) -> None:
    features = pd.read_parquet(features_path)

    train_model(features, model_registry_folder)


def train_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    X = features.drop(columns=[target])
    y = features[target]
    model = RandomForestRegressor(n_estimators=1, max_depth=10, n_jobs=1)
    model.fit(X, y)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_filename = f'model_{timestamp}.joblib'
    joblib.dump(model, os.path.join(model_registry_folder, model_filename))


def predict_with_io(features_path: str, model_path: str, predictions_folder: str) -> None:
    features = pd.read_parquet(features_path)
    features = predict(features, model_path)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    features['predictions_time'] = time_str
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, time_str + '.csv'),
                                                         index=False)
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, 'latest.csv'), index=False)


def predict(features: pd.DataFrame, model_registry_folder: str) -> pd.DataFrame:
    import os

    # Find the latest model file by timestamp in the filename
    model_files = [
        f for f in os.listdir(model_registry_folder) if f.startswith('model_') and f.endswith('.joblib')
    ]
    if not model_files:
        raise FileNotFoundError("No model file found in model_registry_folder")
    
    # Sort by timestamp extracted from filename
    model_files.sort(reverse=True)
    latest_model_path = os.path.join(model_registry_folder, model_files[0])

    model = joblib.load(latest_model_path)
    features['predictions'] = model.predict(features)
    return features

