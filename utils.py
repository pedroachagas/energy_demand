import os
import pendulum
import pandas as pd
import duckdb
from adlfs import AzureBlobFileSystem
import pyarrow.dataset as ds
from load_dotenv import load_dotenv
import loguru as logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import tempfile

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.logger

# Load environment variables
ACC_KEY = os.environ["ACC_KEY"]
CONTAINER_NAME = os.environ["CONTEINER"]
FOLDER = os.environ["FOLDER"]
ACC_NAME = os.environ["ACC_NAME"]

# Initialize Azure Blob FileSystem
abfs = AzureBlobFileSystem(
    account_name=ACC_NAME,
    account_key=ACC_KEY,
    container_name=CONTAINER_NAME
)

def fetch_data(start_date, end_date, area_code):
    logger.info("Starting data fetch")
    start = pendulum.parse(start_date)
    end = pendulum.parse(end_date)

    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start.add(days=90), end)

        url = f'https://apicarga.ons.org.br/prd/cargaverificada?dat_inicio={current_start.to_date_string()}&dat_fim={current_end.to_date_string()}&cod_areacarga={area_code}'

        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))

        response = session.get(url)
        response.raise_for_status()

        chunk_data = response.json()

        # Check for correct number of records
        expected_records = (current_end.add(days=1) - current_start).in_hours() * 2
        if len(chunk_data) != expected_records:
            raise ValueError(f"Data integrity check failed. Expected {expected_records} records, but received {len(chunk_data)} for period {current_start.to_date_string()} to {current_end.to_date_string()}")

        all_data.extend(chunk_data)

        current_start = current_end.add(days=1)

    logger.info("Data fetch completed")
    return all_data

def save_to_bronze(data, date):
    logger.info("Saving data to Bronze layer")
    bronze_path = f"abfs://{CONTAINER_NAME}/{FOLDER}/bronze/raw_data_{date}.json"

    # Convert data to JSON string
    json_data = json.dumps(data)

    # Save JSON string to file
    with abfs.open(bronze_path, 'w') as f:
        f.write(json_data)

    logger.info(f"Data saved to Bronze layer: {bronze_path}")


def process_to_silver(date):
    logger.info("Processing data to Silver layer")
    bronze_path = f"abfs://{CONTAINER_NAME}/{FOLDER}/bronze/raw_data_{date}.json"
    silver_blob_path = f"{CONTAINER_NAME}/{FOLDER}/silver/cleaned_data_{date}.parquet"

    # Read JSON data from Bronze layer
    with abfs.open(bronze_path, 'r') as f:
        json_data = json.load(f)

    # Create a DuckDB connection
    conn = duckdb.connect(":memory:")

    # Convert JSON data to a pandas DataFrame
    df = pd.DataFrame(json_data)

    # Register the DataFrame as a table in DuckDB
    conn.register("bronze_data", df)

    # Process data (select required columns and enforce schema)
    conn.execute(f"""
    CREATE TABLE silver_data AS
    SELECT
        CAST(din_referenciautc AS TIMESTAMP) AS data,
        CAST(val_cargaglobal AS DOUBLE) AS carga_mw
    FROM bronze_data
    WHERE din_referenciautc < '{pendulum.now().to_date_string()}'
    """)

    # Save processed data to local temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        local_silver_path = tmp_file.name
        conn.execute(f"""
        COPY (SELECT * FROM silver_data)
        TO '{local_silver_path}'
        (FORMAT PARQUET)
        """)

    # Upload the local file to Azure Blob Storage
    abfs.put(local_silver_path, silver_blob_path)

    logger.info(f"Data processed and saved to Silver layer: abfs://{silver_blob_path}")

def transform_to_gold(date):
    logger.info("Transforming data to Gold layer")
    silver_blob_path = f"{CONTAINER_NAME}/{FOLDER}/silver/cleaned_data_{date}.parquet"
    gold_blob_path = f"{CONTAINER_NAME}/{FOLDER}/gold/aggregated_data_{date}.parquet"

    # Download the Silver parquet file locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        local_silver_path = tmp_file.name
        with abfs.open(silver_blob_path, 'rb') as data:
            with open(local_silver_path, 'wb') as out_file:
                out_file.write(data.read())

    # Create a DuckDB connection
    conn = duckdb.connect(":memory:")

    # Read data from the local Silver file
    conn.execute(f"CREATE TABLE silver_data AS SELECT * FROM parquet_scan('{local_silver_path}')")

    # Transform data (resample to daily sum)
    conn.execute(f"""
    CREATE TABLE gold_data AS
    SELECT
        DATE_TRUNC('day', data) AS date,
        SUM(carga_mw) AS daily_carga_mw
    FROM silver_data
    GROUP BY DATE_TRUNC('day', data)
    ORDER BY date
    """)

    # Save transformed data to local temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        local_gold_path = tmp_file.name
        conn.execute(f"""
        COPY (SELECT * FROM gold_data)
        TO '{local_gold_path}'
        (FORMAT PARQUET)
        """)

    # Upload the local file to Azure Blob Storage
    abfs.put(local_gold_path, gold_blob_path)

    logger.info(f"Data transformed and saved to Gold layer: abfs://{gold_blob_path}")


def get_gold_data(date):
    logger.info(f"Fetching data from Gold layer for date: {date}")
    gold_blob_path = f"{CONTAINER_NAME}/{FOLDER}/gold/aggregated_data_{date}.parquet"

    # Download the Gold parquet file locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        local_gold_path = tmp_file.name
        with abfs.open(gold_blob_path, 'rb') as data:
            with open(local_gold_path, 'wb') as out_file:
                out_file.write(data.read())

    # Create a DuckDB connection
    conn = duckdb.connect(":memory:")

    # Read the Parquet file from the local path
    query = f"SELECT * FROM parquet_scan('{local_gold_path}')"
    result = conn.execute(query).fetchdf()

    logger.info("Data successfully fetched from Gold layer")
    return result