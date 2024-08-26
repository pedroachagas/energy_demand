import os
import numpy as np
from numba import njit
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from statsmodels.tsa.seasonal import seasonal_decompose
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

from src.utils.azure_utils import load_gold_data
from src.utils.logging_utils import logger
from src.config.config import config

# Load environment variables

RANDOM_STATE = 0
N_ESTIMATORS = 100

def split_data(data, start_date, split_date):
    df_train = data[(data["ds"] >= start_date) & (data["ds"] < split_date)]
    df_oot = data[data["ds"] >= split_date]

    return df_train, df_oot

@njit
def diff(x, lag):
    x2 = np.full_like(x, np.nan)
    for i in range(lag, len(x)):
        x2[i] = x[i] - x[i-lag]
    return x2

@njit
def rolling_mean(x, window):
    x2 = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        x2[i] = np.mean(x[i-window+1:i+1])
    return x2

@njit
def rolling_std(x, window):
    x2 = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        x2[i] = np.std(x[i-window+1:i+1])
    return x2

def seasonal_decomposition_features(df):
    result = seasonal_decompose(df.set_index('ds')['y'], model='additive')
    df['trend'] = result.trend
    df['seasonality'] = result.seasonal
    df['residual'] = result.resid
    return df


def train_model(df_train, models):
    logger.info("Training model")
    return MLForecast(
        models=models,
        freq='D',
        lags=[1, 7, 14, 28],
        lag_transforms={
            1: [
                (rolling_mean, 3),
                (rolling_mean, 7),
                (rolling_mean, 14),
                (rolling_mean, 28),
                (rolling_std, 7),
                (rolling_std, 14),
                (rolling_std, 28),
                (diff, 1),
                (diff, 7),
                (diff, 15),
                (diff, 28)
            ],
        },
        date_features=[
            'month',
            'day',
            'week',
            'dayofyear',
            'quarter',
            'dayofweek',
        ],
        num_threads=12
        ).fit(
            df_train,
            id_col='unique_id',
            time_col='ds',
            target_col='y',
            static_features=[],
            prediction_intervals=PredictionIntervals(n_windows=3, method="conformal_distribution"),
            fitted=True
        )

def save_model(model):
    logger.info("Saving model")
    os.makedirs('models', exist_ok=True)  # Create 'models' directory in the current working directory
    model_path = 'models/trained_model.joblib'  # Save the model in the 'models' directory
    joblib.dump(model, model_path)
    logger.info(f"Model saved successfully at {model_path}")

def run_pipeline():
    # Retrieve and use the Gold layer data
    data = load_gold_data()

    # Select the models to be trained
    models = [
        CatBoostRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        LGBMRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        XGBRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS)
    ]

    # Train the models
    model = train_model(data, models)

    # Save the model
    save_model(model)

if __name__ == "__main__":
    run_pipeline()