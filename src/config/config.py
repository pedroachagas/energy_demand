import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    START_DATE = os.getenv("START_DATE")
    END_DATE = os.getenv("END_DATE")
    AREA_CODE = os.getenv("AREA_CODE")
    ACC_NAME = os.getenv("ACC_NAME")
    ACC_KEY = os.getenv("ACC_KEY")
    CONTAINER_NAME = os.getenv("CONTEINER")
    FOLDER = os.getenv("FOLDER")
    AZURE_FILE_NAME = os.getenv("AZURE_FILE_NAME")
    MODEL_START_DATE = os.getenv("MODEL_START_DATE")
    MODEL_SPLIT_DATE = os.getenv("MODEL_SPLIT_DATE")
    LEVELS = [int(level.strip()) for level in os.getenv("LEVELS", "").split(",") if level.strip()]
    HORIZON = 60

config = Config()