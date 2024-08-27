import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

class Config:
    """
    Configuration class that loads environment variables for the application.
    """
    START_DATE: str = os.getenv("START_DATE", "")
    END_DATE: str = os.getenv("END_DATE", "")
    AREA_CODE: str = os.getenv("AREA_CODE", "")
    ACC_NAME: str = os.getenv("ACC_NAME", "")
    ACC_KEY: str = os.getenv("ACC_KEY", "")
    CONTAINER_NAME: str = os.getenv("CONTEINER", "")
    FOLDER: str = os.getenv("FOLDER", "")
    AZURE_FILE_NAME: str = os.getenv("AZURE_FILE_NAME", "")
    MODEL_START_DATE: str = os.getenv("MODEL_START_DATE", "")
    MODEL_SPLIT_DATE: str = os.getenv("MODEL_SPLIT_DATE", "")
    LEVELS: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    HORIZON: int = 60

config = Config()
