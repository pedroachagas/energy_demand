import os
import numpy as np
import json
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import loguru as logging
import pendulum
from load_dotenv import load_dotenv
from utils import get_gold_data, split_data, train_model, score_data, save_predictions

# Initialize logger
logger = logging.logger

# Load environment variables
load_dotenv()

# Load environment variables
MODEL_START_DATE = os.environ["MODEL_START_DATE"]
MODEL_SPLIT_DATE = os.environ["MODEL_SPLIT_DATE"]
LEVELS = [int(p) for p in os.environ['LEVELS'].split(', ')]
RANDOM_STATE = 0
N_ESTIMATORS = 100

def run_pipeline():

    # Retrieve and use the Gold layer data
    gold_data = get_gold_data()

    # Split the data into training and testing sets
    df_train, df_oot = split_data(gold_data, MODEL_START_DATE, MODEL_SPLIT_DATE)

    # Train the models
    models = [
        CatBoostRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        LGBMRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        XGBRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS)
        ]

    model = train_model(df_train, models)

    # Score data
    forecast_df = score_data(df_oot, model, levels=LEVELS)

    # Save predictions
    process_date = pendulum.now().to_date_string().replace("-", "")
    save_predictions(forecast_df, process_date)

if __name__ == "__main__":
    run_pipeline()