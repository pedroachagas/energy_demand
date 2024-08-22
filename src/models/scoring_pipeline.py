import os
import pendulum
from load_dotenv import load_dotenv
from utils import get_gold_data, score_data, save_predictions
import loguru as logging
import joblib
import requests

# Initialize logger
logger = logging.logger

# Load environment variables
load_dotenv()

# Load environment variables
LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
HORIZON = 60

def download_model():
    # Replace with your actual GitHub repository and artifact details
    url = "https://api.github.com/repos/OWNER/REPO/actions/artifacts"
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
    response = requests.get(url, headers=headers)
    artifacts = response.json()["artifacts"]

    # Find the latest trained model artifact
    model_artifact = next(artifact for artifact in artifacts if artifact["name"] == "trained-model")

    # Download the artifact
    download_url = model_artifact["archive_download_url"]
    model_data = requests.get(download_url, headers=headers).content

    with open("trained_model.joblib", "wb") as f:
        f.write(model_data)

def run_pipeline():
    try:
        # Download the trained model
        download_model()

        # Load the trained model
        model = joblib.load('trained_model.joblib')

        # Get the latest data
        gold_data = get_gold_data()

        # Prepare data for scoring (use all available data)
        df_score = gold_data

        # Score data
        forecast_df = score_data(df_score, model, levels=LEVELS)

        # Save predictions
        process_date = pendulum.now().to_date_string().replace("-", "")
        save_predictions(forecast_df, process_date)

    except Exception as e:
        logger.error(f"Scoring pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()