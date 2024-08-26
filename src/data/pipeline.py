import pendulum
from src.config.config import config
from src.utils.logging_utils import logger
from src.data.extract import fetch_data
from src.data.transform import transform_to_bronze, transform_to_silver, transform_to_gold
from src.data.load import load_to_bronze, load_to_silver, load_to_gold
from src.utils.azure_utils import check_file_exists, sanitize_blob_name

def run_pipeline():
    try:
        end_date = config.END_DATE or pendulum.now().subtract(days=1).to_date_string()
        process_date = pendulum.now().to_date_string().replace("-", "")

        # Check if data for the process_date already exists
        if data_exists(process_date):
            logger.info(f"Data for {process_date} already processed. Skipping.")
            return

        # Extract
        raw_data = fetch_data(config.START_DATE, end_date, config.AREA_CODE)

        # Transform and Load - Bronze
        bronze_df = transform_to_bronze(raw_data)
        load_to_bronze(bronze_df, process_date)

        # Transform and Load - Silver
        silver_df = transform_to_silver(bronze_df)
        load_to_silver(silver_df, process_date)

        # Transform and Load - Gold
        gold_df = transform_to_gold(silver_df)
        load_to_gold(gold_df, process_date)

        logger.info(f"Pipeline completed successfully for {process_date}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

def data_exists(process_date):
    gold_path = sanitize_blob_name(f"{config.FOLDER}/gold/aggregated_data_{process_date}.parquet")
    return check_file_exists(gold_path)

if __name__ == "__main__":
    run_pipeline()