import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pendulum
import pandas as pd
from azure.storage.blob import BlobServiceClient
import loguru as logging
import os
import io
import pyarrow as pa
import pyarrow.parquet as pq
from load_dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.logger

# Load environment variables
ACC_KEY = os.environ["ACC_KEY"]
CONTAINER_NAME = os.environ["CONTEINER"]
FOLDER = os.environ["FOLDER"]
ACC_NAME = os.environ["ACC_NAME"]
AZURE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={ACC_NAME};AccountKey={ACC_KEY};EndpointSuffix=core.windows.net"


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


def transform_data(data):
    logger.info("Starting data transformation")
    df = pd.DataFrame(data)[['din_referenciautc', 'val_cargaglobal']]
    df = df.rename(columns={'val_cargaglobal': 'carga_mw'})
    df = df[df['din_referenciautc'] < pendulum.now().to_date_string()]
    df['din_referenciautc'] = pd.to_datetime(df['din_referenciautc'])
    df.set_index('din_referenciautc', inplace=True)
    daily_data = df.resample('D').sum()
    logger.info("Data transformation completed")
    return daily_data


def load_data_to_azure_blob(dataframe, container_name, file_name):
    logger.info("Starting data loading to Azure Blob Storage")

    # Convert DataFrame to Parquet format in-memory
    table = pa.Table.from_pandas(dataframe)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)

    # Initialize Azure Blob Service client
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)

    # Upload the parquet file to Azure Blob Storage
    blob_client.upload_blob(buffer, overwrite=True)
    logger.info(f"Data successfully loaded to {container_name}/{file_name}")


def get_blob_data(file_name):
    # Initialize Azure Blob Service client
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=os.path.join(FOLDER, file_name))

    # Download the parquet file from Azure Blob Storage
    download_stream = blob_client.download_blob()
    parquet_bytes = download_stream.readall()

    # Read the parquet file into a pandas DataFrame
    dataframe = pq.read_table(io.BytesIO(parquet_bytes)).to_pandas()
    return dataframe
