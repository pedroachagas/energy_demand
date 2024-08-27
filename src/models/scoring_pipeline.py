import os
import pendulum
import pandas as pd
import requests
import zipfile
import glob
import tempfile
import joblib
from typing import Optional, Any

from src.utils.azure_utils import load_gold_data, load_predictions, get_azure_blob_fs
from src.utils.logging_utils import logger
from src.config.config import config


def download_and_extract_model() -> Optional[Any]:
    """Downloads and extracts the trained model from GitHub, returning the model object."""
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        logger.warning("GITHUB_TOKEN not found in environment variables. Skipping model download.")
        return None

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


def update_predictions(df_hist: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Updates predictions by merging historical data."""
    df_hist['ds'] = pd.to_datetime(df_hist['ds'])
    preds['ds'] = pd.to_datetime(preds['ds'])

    merged = pd.merge(preds, df_hist[['ds', 'y']], on='ds', how='left', suffixes=('_pred', '_hist'))
    merged['y'] = merged['y_hist'].fillna(merged['y_pred'])
    updated_preds = merged.drop(['y_pred', 'y_hist'], axis=1)

    return updated_preds[preds.columns]


def score_data(df: pd.DataFrame, model: Any, levels: list[float]) -> pd.DataFrame:
    """Scores the data using the model and returns the forecast."""
    logger.info("Scoring data")

    data = df[['ds', 'y', 'unique_id']].dropna()
    model.update(data)

    forecast_df = model.predict(h=config.HORIZON, level=levels)
    forecast_df = pd.concat([df, forecast_df], axis=0).drop_duplicates(subset=['ds'], keep='last')

    return forecast_df


def save_predictions(df: pd.DataFrame, date: str) -> None:
    """Saves predictions to Azure Blob Storage."""
    logger.info("Saving predictions to Gold layer")
    predictions_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/predictions/predictions_{date}.parquet"

    abfs = get_azure_blob_fs()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        local_predictions_path = tmp_file.name
        df.to_parquet(local_predictions_path, index=False)

    abfs.put(local_predictions_path, predictions_blob_path)

    logger.info(f"Predictions saved to Gold layer: abfs://{predictions_blob_path}")


def run_pipeline() -> None:
    """Runs the full pipeline: download model, update, score, and save predictions."""
    try:
        model = download_and_extract_model()
        if model is None:
            return

        df_hist = load_gold_data()
        existing_predictions = load_predictions()

        updated_predictions = update_predictions(df_hist, existing_predictions)
        new_predictions = score_data(updated_predictions, model, config.LEVELS)

        process_date = pendulum.now().to_date_string().replace("-", "")
        save_predictions(new_predictions, process_date)

    except Exception as e:
        logger.error(f"Scoring pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_pipeline()
