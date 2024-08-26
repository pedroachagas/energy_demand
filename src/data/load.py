import tempfile
from src.utils.logging_utils import logger
from src.utils.azure_utils import save_to_azure
from src.config.config import config

def load_to_bronze(df, process_date):
    logger.info("Loading data to Bronze layer")
    bronze_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/bronze/raw_data_{process_date}.json"

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_file:
        df.to_json(tmp_file.name, orient='records')
        save_to_azure(tmp_file.name, bronze_path)

    logger.info(f"Data loaded to Bronze layer: {bronze_path}")

def load_to_silver(df, process_date):
    logger.info("Loading data to Silver layer")
    silver_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/silver/cleaned_data_{process_date}.parquet"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        df.to_parquet(tmp_file.name, index=False)
        save_to_azure(tmp_file.name, silver_path)

    logger.info(f"Data loaded to Silver layer: {silver_path}")

def load_to_gold(df, process_date):
    logger.info("Loading data to Gold layer")
    gold_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/gold/aggregated_data_{process_date}.parquet"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        df.to_parquet(tmp_file.name, index=False)
        save_to_azure(tmp_file.name, gold_path)

    logger.info(f"Data loaded to Gold layer: {gold_path}")