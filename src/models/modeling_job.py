import os
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from utils import get_gold_data, split_data, train_model
import joblib

# Load environment variables
load_dotenv()

def run_modeling():
    MODEL_START_DATE = os.environ["MODEL_START_DATE"]
    MODEL_SPLIT_DATE = os.environ["MODEL_SPLIT_DATE"]

    # Retrieve and use the Gold layer data
    gold_data = get_gold_data()

    # Split the data into training and testing sets
    df_train, _ = split_data(gold_data, MODEL_START_DATE, MODEL_SPLIT_DATE)

    # Train the models
    models = [
        CatBoostRegressor(random_state=0, n_estimators=100),
        LGBMRegressor(random_state=0, n_estimators=100),
        XGBRegressor(random_state=0, n_estimators=100),
        RandomForestRegressor(random_state=0, n_estimators=100)
    ]

    model = train_model(df_train, models)

    # Save the trained model
    joblib.dump(model, 'trained_model.joblib')

if __name__ == "__main__":
    run_modeling()