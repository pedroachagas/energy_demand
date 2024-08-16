import os
import pendulum
from load_dotenv import load_dotenv
from utils import fetch_data, save_to_bronze, process_to_silver, transform_to_gold, get_gold_data
import loguru as logging

# Initialize logger
logger = logging.logger

# Load environment variables
load_dotenv()

# Load environment variables
START_DATE = os.environ["START_DATE"]

def run_pipeline():
    END_DATE = os.getenv('END_DATE', pendulum.now().subtract(days=1).to_date_string())
    AREA_CODE = os.getenv('AREA_CODE', 'SP')

    try:
        # Fetch data
        data = fetch_data(START_DATE, END_DATE, AREA_CODE)

        # Process date
        process_date = pendulum.now().to_date_string().replace("-", "")

        # Save to Bronze layer
        save_to_bronze(data, process_date)

        # Process to Silver layer
        process_to_silver(process_date)

        # Transform to Gold layer
        transform_to_gold(process_date)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()