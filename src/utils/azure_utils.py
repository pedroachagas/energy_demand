from adlfs import AzureBlobFileSystem
from src.config.config import config
from src.utils.logging_utils import logger
import pandas as pd
import pyarrow.dataset as ds
from typing import Any
import json

def get_azure_blob_fs() -> AzureBlobFileSystem:
    """
    Initialize and return an AzureBlobFileSystem object for the configured Azure storage account.

    Returns:
        AzureBlobFileSystem: The Azure Blob File System object.
    """
    return AzureBlobFileSystem(
        account_name=config.ACC_NAME,
        account_key=config.ACC_KEY,
        container_name=config.CONTAINER_NAME
    )

def save_to_azure(local_path: str, azure_path: str) -> None:
    """
    Save a local file to Azure Blob Storage.

    Args:
        local_path (str): The path to the local file.
        azure_path (str): The path in Azure Blob Storage where the file should be saved.
    """
    abfs = get_azure_blob_fs()
    with open(local_path, 'rb') as local_file:
        with abfs.open(azure_path, 'wb') as azure_file:
            azure_file.write(local_file.read())

def check_file_exists(azure_path: str) -> Any|bool:
    """
    Check if a file exists in Azure Blob Storage.

    Args:
        azure_path (str): The path in Azure Blob Storage to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    abfs = get_azure_blob_fs()
    return abfs.exists(azure_path)

def load_gold_data() -> pd.DataFrame:
    """
    Load the latest Gold layer data from Azure Blob Storage into a Pandas DataFrame.

    Returns:
        pd.DataFrame: The Gold layer data with columns renamed and sorted by date.
    """
    gold_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/gold/"

    # Find the latest file
    logger.info(f"Finding the latest file in {gold_blob_path}")
    abfs = get_azure_blob_fs()
    files = abfs.ls(gold_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    gold_blob_path = latest_file

    # Load the Gold Parquet file
    logger.info(f"Loading Gold data from {gold_blob_path}")
    pqdata = ds.dataset(gold_blob_path, filesystem=abfs)

    return (
        pqdata
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
    )

def load_silver_data():
    """
    Load the latest Silver layer data from Azure Blob Storage into a Pandas DataFrame.

    Returns:
        pd.DataFrame: The Silver layer data with columns renamed and sorted by date.
    """
    silver_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/silver/"

    # Find the latest file
    logger.info(f"Finding the latest file in {silver_blob_path}")
    abfs = get_azure_blob_fs()
    files = abfs.ls(silver_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    silver_blob_path = latest_file

    # Load the Silver Parquet file
    logger.info(f"Loading Silver data from {silver_blob_path}")
    pqdata = ds.dataset(silver_blob_path, filesystem=abfs)

    return (
        pqdata
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
    )

def load_bronze_data() -> pd.DataFrame:
    """
    Load the latest Bronze layer data from Azure Blob Storage into a Pandas DataFrame.

    Returns:
        pd.DataFrame: The Bronze layer data with columns renamed and sorted by date.
    """
    bronze_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/bronze/"

    # Find the latest file
    logger.info(f"Finding the latest file in {bronze_blob_path}")
    abfs = get_azure_blob_fs()
    files = abfs.ls(bronze_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    bronze_blob_path = latest_file

    # Load the Bronze JSON file
    logger.info(f"Loading Bronze data from {bronze_blob_path}")
    with abfs.open(bronze_blob_path, "r") as file:
        return pd.DataFrame(json.load(file))


def load_predictions() -> pd.DataFrame:
    """
    Load the latest predictions data from Azure Blob Storage into a Pandas DataFrame.

    Returns:
        pd.DataFrame: The predictions data sorted by date.
    """
    predictions_blob_path = f"{config.CONTAINER_NAME}/{config.FOLDER}/predictions/"

    # Find the latest file
    abfs = get_azure_blob_fs()
    files = abfs.ls(predictions_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    predictions_blob_path = latest_file

    # Load the predictions Parquet file
    logger.info(f"Loading predictions from {predictions_blob_path}")
    pqdata = ds.dataset(predictions_blob_path, filesystem=abfs)

    return (
        pqdata
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
    )