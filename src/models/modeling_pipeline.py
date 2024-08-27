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
import pandas as pd
from typing import Tuple, List
from numpy.typing import NDArray

from src.utils.azure_utils import load_gold_data
from src.utils.logging_utils import logger
from src.config.config import config

# Load environment variables
RANDOM_STATE = 0
N_ESTIMATORS = 100

def split_data(data: pd.DataFrame, start_date: str, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and out-of-time datasets.

    Args:
        data (pd.DataFrame): The complete dataset.
        start_date (str): Start date for the training data.
        split_date (str): Date to split the training and out-of-time data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training data and out-of-time data.
    """
    df_train = data[(data["ds"] >= start_date) & (data["ds"] < split_date)]
    df_oot = data[data["ds"] >= split_date]
    return df_train, df_oot

@njit # type: ignore
def diff(x: NDArray[np.float64], lag: int) -> NDArray[np.float64]:
    """
    Compute the difference between each element and its lag.

    Args:
        x (NDArray[np.float64]): Input array.
        lag (int): Number of periods to lag.

    Returns:
        NDArray[np.float64]: Array of lag differences.
    """
    x2 = np.full_like(x, np.nan)
    for i in range(lag, len(x)):
        x2[i] = x[i] - x[i-lag]
    return x2

@njit # type: ignore
def rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Compute the rolling mean over a specified window.

    Args:
        x (NDArray[np.float64]): Input array.
        window (int): Rolling window size.

    Returns:
        NDArray[np.float64]: Array of rolling means.
    """
    x2 = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        x2[i] = np.mean(x[i-window+1:i+1])
    return x2

@njit # type: ignore
def rolling_std(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Compute the rolling standard deviation over a specified window.

    Args:
        x (NDArray[np.float64]): Input array.
        window (int): Rolling window size.

    Returns:
        NDArray[np.float64]: Array of rolling standard deviations.
    """
    x2 = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        x2[i] = np.std(x[i-window+1:i+1])
    return x2

def seasonal_decomposition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonal decomposition features (trend, seasonality, residual) to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with 'ds' as date and 'y' as target columns.

    Returns:
        pd.DataFrame: Dataframe with additional columns for trend, seasonality, and residual.
    """
    result = seasonal_decompose(df.set_index('ds')['y'], model='additive')
    df['trend'] = result.trend
    df['seasonality'] = result.seasonal
    df['residual'] = result.resid
    return df

def train_model(df_train: pd.DataFrame, models: List[object]) -> MLForecast:
    """
    Train the machine learning models using MLForecast.

    Args:
        df_train (pd.DataFrame): Training dataset.
        models (List[object]): List of models to be trained.

    Returns:
        MLForecast: Trained MLForecast object.
    """
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

def save_model(model: MLForecast) -> None:
    """
    Save the trained model to a file.

    Args:
        model (MLForecast): Trained MLForecast model.
    """
    logger.info("Saving model")
    os.makedirs('models', exist_ok=True)  # Create 'models' directory if it doesn't exist
    model_path = 'models/trained_model.joblib'  # Save the model in the 'models' directory
    joblib.dump(model, model_path)
    logger.info(f"Model saved successfully at {model_path}")

def run_pipeline() -> None:
    """
    Execute the full pipeline: load data, train models, and save the trained model.
    """
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