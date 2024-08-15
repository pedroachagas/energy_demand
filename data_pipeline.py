import os
import pendulum
from load_dotenv import load_dotenv
from dashboard import FILE_NAME
from utils import fetch_data, transform_data, load_data_to_azure_blob
import loguru as logging

# Initialize logger
logger = logging.logger

# Load environment variables
load_dotenv()

# Load environment variables
ACC_KEY = os.environ["ACC_KEY"]
CONTAINER_NAME = os.environ["CONTEINER"]
FOLDER = os.environ["FOLDER"]
START_DATE = os.environ["START_DATE"]
FILE_NAME = os.environ["AZURE_FILE_NAME"]

def run_pipeline():
    END_DATE = os.getenv('END_DATE', pendulum.now().subtract(days=1).to_date_string())
    AREA_CODE = os.getenv('AREA_CODE', 'SP')

    try:
        data = fetch_data(START_DATE, END_DATE, AREA_CODE)
        transformed_data = transform_data(data)
        load_data_to_azure_blob(
            dataframe=transformed_data,
            container_name=CONTAINER_NAME,
            file_name=os.path.join(FOLDER, FILE_NAME)
            )
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
