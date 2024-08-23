import os
import pendulum
from load_dotenv import load_dotenv
from ..utils import get_gold_data, score_data, save_predictions, load_predictions
import loguru as logging
import joblib
import pandas as pd
import requests
import zipfile
import io
import glob

# Initialize logger
logger = logging.logger

# Load environment variables
load_dotenv()

# Load environment variables
LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

def download_and_extract_model():
    # ... (keep the existing implementation)

def run_pipeline():
    try:
        # Download and extract the trained model
        download_and_extract_model()

        # Find the extracted joblib file
        joblib_files = glob.glob("model_folder/*.joblib")
        if not joblib_files:
            logger.error("No .joblib file found. Unable to proceed with scoring.")
            return

        model_path = joblib_files[0]

        # Load the trained model
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        # Get the latest data
        df_hist = get_gold_data()

        # Load existing predictions
        existing_predictions = load_predictions()

        # Get the date of the last actual data point
        last_actual_date = df_hist['ds'].max()

        # Score data
        new_forecast = score_data(df_hist, model, levels=LEVELS)

        # Combine existing predictions with new predictions
        if existing_predictions is not None and not existing_predictions.empty:
            # Keep existing predictions up to last_actual_date
            existing_predictions = existing_predictions[existing_predictions['ds'] <= last_actual_date]
            updated_predictions = pd.concat([existing_predictions, new_forecast])
        else:
            updated_predictions = new_forecast

        # Keep only the latest prediction for each date
        updated_predictions = updated_predictions.sort_values('ds').groupby('ds').last().reset_index()

        # Ensure we have a prediction for 60 days from now
        today = pendulum.now().start_of('day')
        future_date = today.add(days=60)
        if future_date.date() not in updated_predictions['ds'].dt.date.values:
            logger.warning(f"Missing prediction for {future_date.date()}. Rerunning prediction.")
            new_forecast = score_data(df_hist, model, levels=LEVELS)
            updated_predictions = pd.concat([updated_predictions, new_forecast]).sort_values('ds').groupby('ds').last().reset_index()

        # Save updated predictions
        process_date = pendulum.now().to_date_string().replace("-", "")
        save_predictions(updated_predictions, process_date)

    except Exception as e:
        logger.error(f"Scoring pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()