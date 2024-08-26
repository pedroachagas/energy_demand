from adlfs import AzureBlobFileSystem
from src.config.config import config
from src.utils.logging_utils import logger
import pandas as pd
import pyarrow.dataset as ds

def get_azure_blob_fs():
    return AzureBlobFileSystem(
        account_name=config.ACC_NAME,
        account_key=config.ACC_KEY,
        container_name=config.CONTAINER_NAME
    )

def save_to_azure(local_path, azure_path):
    abfs = get_azure_blob_fs()
    with open(local_path, 'rb') as local_file:
        with abfs.open(azure_path, 'wb') as azure_file:
            azure_file.write(local_file.read())

def load_from_azure(azure_path, local_path):
    abfs = get_azure_blob_fs()
    with abfs.open(azure_path, 'rb') as azure_file:
        with open(local_path, 'wb') as local_file:
            local_file.write(azure_file.read())

def check_file_exists(azure_path):
    abfs = get_azure_blob_fs()
    return abfs.exists(azure_path)

def load_gold_data():
    gold_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/gold/"

    # find the file latest file
    abfs = get_azure_blob_fs()
    files = abfs.ls(gold_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    gold_blob_path = latest_file

    # Download the Gold parquet file locally
    logger.info(f"Loading Gold data from {gold_blob_path}")
    pqdata = ds.dataset(gold_blob_path, filesystem=abfs)

    return (
        pqdata
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
        .assign(
            date=lambda x: pd.to_datetime(x["date"]),
            unique_id=0
        )
        .rename(columns={
            "date": "ds",
            "daily_carga_mw": "y",
        })
        .sort_values("ds")
    )

def load_predictions():
    predictions_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/predictions/"

    # find the file latest file
    abfs = get_azure_blob_fs()
    files = abfs.ls(predictions_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    predictions_blob_path = latest_file

    # Download the predictions parquet file locally
    logger.info(f"Loading predictions from {predictions_blob_path}")
    pqdata = ds.dataset(predictions_blob_path, filesystem=abfs)

    return (
        pqdata
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
        .assign(
            ds=lambda x: pd.to_datetime(x["ds"]),
        )
        .sort_values("ds")
    )