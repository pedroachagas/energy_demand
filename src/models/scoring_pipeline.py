import os
import pendulum
from load_dotenv import load_dotenv
from ..utils import get_gold_data, score_data, save_predictions
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
    url = "https://api.github.com/repos/pedroachagas/energy_demand/actions/artifacts"
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        artifacts = response.json()["artifacts"]

        logger.info(f"Found {len(artifacts)} artifacts")

        # List all artifact names
        artifact_names = [artifact["name"] for artifact in artifacts]
        logger.info(f"Available artifacts: {', '.join(artifact_names)}")

        # Find the latest trained model artifact
        model_artifacts = [artifact for artifact in artifacts if artifact["name"] == "trained-model"]

        if not model_artifacts:
            raise ValueError("No 'trained-model' artifact found")

        model_artifact = model_artifacts[0]  # Get the latest one

        # Download the artifact
        download_url = model_artifact["archive_download_url"]
        model_data = requests.get(download_url, headers=headers).content

        with open("trained_model.joblib", "wb") as f:
            f.write(model_data)

        logger.info("Model downloaded successfully")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to GitHub API: {e}")
        raise
    except KeyError as e:
        logger.error(f"Unexpected response format from GitHub API: {e}")
        raise
    except ValueError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

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