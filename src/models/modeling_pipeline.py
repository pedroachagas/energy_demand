import os
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import loguru as logging
from load_dotenv import load_dotenv
from ..utils import get_gold_data, train_model
import joblib

# Initialize logger
logger = logging.logger

# Load environment variables
load_dotenv()

# Load environment variables
MODEL_START_DATE = os.environ["MODEL_START_DATE"]
MODEL_SPLIT_DATE = os.environ["MODEL_SPLIT_DATE"]
LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
RANDOM_STATE = 0
N_ESTIMATORS = 100

def run_pipeline():
    # Retrieve and use the Gold layer data
    data = get_gold_data()

    # Train the models
    models = [
        CatBoostRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        LGBMRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        XGBRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS),
        RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS)
    ]

    model = train_model(data, models)

    # Save the trained model
    joblib.dump(model, 'trained_model.joblib')
    logger.info("Model trained and saved successfully")

if __name__ == "__main__":
    run_pipeline()