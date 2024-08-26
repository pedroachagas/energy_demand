import os
import pendulum
import pandas as pd
import requests
import zipfile
import glob
import tempfile
import joblib

from src.utils.azure_utils import load_gold_data, load_predictions, get_azure_blob_fs
from src.utils.logging_utils import logger
from src.config.config import config


def download_and_extract_model():
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        logger.warning("GITHUB_TOKEN not found in environment variables. Skipping model download.")
        return

    url = "https://api.github.com/repos/pedroachagas/energy_demand/actions/artifacts"
    headers = {"Authorization": f"token {github_token}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        artifacts = response.json()["artifacts"]

        logger.info(f"Found {len(artifacts)} artifacts")
        artifact_names = [artifact["name"] for artifact in artifacts]
        logger.info(f"Available artifacts: {', '.join(artifact_names)}")

        model_artifacts = [artifact for artifact in artifacts if artifact["name"] == "trained-model"]
        if not model_artifacts:
            raise ValueError("No 'trained-model' artifact found")

        model_artifact = model_artifacts[0]
        logger.info(f"Downloading artifact: {model_artifact['name']}")
        download_url = model_artifact["archive_download_url"]
        zip_content = requests.get(download_url, headers=headers).content

        # Save as zip file
        with open("model.zip", "wb") as zip_file:
            zip_file.write(zip_content)

        # Extract zip file
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall("model_folder")

        # Find the extracted joblib file
        joblib_files = glob.glob("model_folder/*.joblib")
        if not joblib_files:
            raise ValueError("No .joblib file found in the extracted contents")

        logger.info(f"Model file extracted: {joblib_files[0]}")

    except Exception as e:
        logger.error(f"Error in download_and_extract_model: {str(e)}")
        raise

    return joblib.load(joblib_files[0])

def update_predictions(df_hist, preds):
    # Ensure 'ds' column is of datetime type in both dataframes
    df_hist['ds'] = pd.to_datetime(df_hist['ds'])
    preds['ds'] = pd.to_datetime(preds['ds'])

    # Merge the dataframes on 'ds'
    merged = pd.merge(preds, df_hist[['ds', 'y']], on='ds', how='left', suffixes=('_pred', '_hist'))

    # Update 'y' column in merged dataframe
    merged['y'] = merged['y_hist'].fillna(merged['y_pred'])

    # Drop unnecessary columns
    updated_preds = merged.drop(['y_pred', 'y_hist'], axis=1)

    # Ensure the columns are in the same order as the original preds dataframe
    updated_preds = updated_preds[preds.columns]

    return updated_preds


def score_data(df, model, levels):
    logger.info("Scoring data")

    # Update the model with the new data
    data = df[['ds', 'y', 'unique_id']].dropna()
    model.update(data)

    # Make predictions
    forecast_df = model.predict(h=config.HORIZON, level=levels)

    # Merge the predictions with the original data
    forecast_df = pd.concat([df, forecast_df], axis=0).drop_duplicates(subset=['ds'], keep='last')

    return forecast_df

def save_predictions(df, date):

    logger.info("Saving predictions to Gold layer")
    predictions_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/predictions/predictions_{date}.parquet"

    # Save predictions to local temporary file
    abfs = get_azure_blob_fs()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        local_predictions_path = tmp_file.name
        df.to_parquet(local_predictions_path, index=False)

    # Upload the local file to Azure Blob Storage
    abfs.put(local_predictions_path, predictions_blob_path)

    logger.info(f"Predictions saved to Gold layer: abfs://{predictions_blob_path}")


def run_pipeline():
    try:
        # Download the trained model
        try:
            model = download_and_extract_model()
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        # Get the latest data
        df_hist = load_gold_data()

        # Load existing predictions
        existing_predictions = load_predictions()

        # Update the predictions
        updated_predictions = update_predictions(df_hist, existing_predictions)

        # Score the data
        new_predictions = score_data(updated_predictions, model, config.LEVELS)

        # Save updated predictions
        process_date = pendulum.now().to_date_string().replace("-", "")
        save_predictions(new_predictions, process_date)

    except Exception as e:
        logger.error(f"Scoring pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()