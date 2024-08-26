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

import numpy as np
from numba import njit
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from statsmodels.tsa.seasonal import seasonal_decompose
from utilsforecast.plotting import plot_series


import plotly.graph_objs as go


# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.logger

# Load environment variables
ACC_KEY = os.environ["ACC_KEY"]
CONTAINER_NAME = os.environ["CONTEINER"]
FOLDER = os.environ["FOLDER"]
ACC_NAME = os.environ["ACC_NAME"]
HORIZON = 60

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


def get_gold_data():
    logger.info(f"Fetching data from Gold layer for date")
    gold_blob_path = f"{CONTAINER_NAME}/{FOLDER}/gold/"

    # find the file latest file
    files = abfs.ls(gold_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    gold_blob_path = latest_file

    # Download the Gold parquet file locally
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
    )

def split_data(data, start_date, split_date):
    df_train = data[(data["ds"] >= start_date) & (data["ds"] < split_date)]
    df_oot = data[data["ds"] >= split_date]

    return df_train, df_oot

@njit
def diff(x, lag):
    x2 = np.full_like(x, np.nan)
    for i in range(lag, len(x)):
        x2[i] = x[i] - x[i-lag]
    return x2

@njit
def rolling_mean(x, window):
    x2 = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        x2[i] = np.mean(x[i-window+1:i+1])
    return x2

@njit
def rolling_std(x, window):
    x2 = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        x2[i] = np.std(x[i-window+1:i+1])
    return x2

def seasonal_decomposition_features(df):
    result = seasonal_decompose(df.set_index('ds')['y'], model='additive')
    df['trend'] = result.trend
    df['seasonality'] = result.seasonal
    df['residual'] = result.resid
    return df


def train_model(df_train, models):
    logger.info("Training model")
    return MLForecast(
        models=models,
        freq='D',
        lags=[1, 7, 14, 28],
        lag_transforms={
            1: [
                (rolling_mean, 3),
                (rolling_mean, 7),
                (rolling_mean, 14),
                (rolling_mean, 28),
                (rolling_std, 7),
                (rolling_std, 14),
                (rolling_std, 28),
                (diff, 1),
                (diff, 7),
                (diff, 15),
                (diff, 28)
            ],
        },
        date_features=[
            'month',
            'day',
            'week',
            'dayofyear',
            'quarter',
            'dayofweek',
        ],
        num_threads=12
        ).fit(
            df_train,
            id_col='unique_id',
            time_col='ds',
            target_col='y',
            static_features=[],
            prediction_intervals=PredictionIntervals(n_windows=3, method="conformal_distribution"),
            fitted=True
        )

def score_data(df, model, levels):
    logger.info("Scoring data")

    # Update the model with the new data
    data = df[['ds', 'y', 'unique_id']].dropna()
    model.update(data)

    # Make predictions
    forecast_df = model.predict(h=HORIZON, level=levels)

    # Merge the predictions with the original data
    forecast_df = pd.concat([df, forecast_df], axis=0).drop_duplicates(subset=['ds'], keep='last')

    return forecast_df

def save_predictions(df, date):

    logger.info("Saving predictions to Gold layer")
    predictions_blob_path = f"{CONTAINER_NAME}/{FOLDER}/predictions/predictions_{date}.parquet"

    # Save predictions to local temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        local_predictions_path = tmp_file.name
        df.to_parquet(local_predictions_path, index=False)

    # Upload the local file to Azure Blob Storage
    abfs.put(local_predictions_path, predictions_blob_path)

    logger.info(f"Predictions saved to Gold layer: abfs://{predictions_blob_path}")

def load_predictions():
    logger.info("Loading predictions")
    predictions_blob_path = f"{CONTAINER_NAME}/{FOLDER}/predictions/"

    # find the file latest file
    files = abfs.ls(predictions_blob_path)
    latest_file = max(files, key=lambda x: x.split("/")[-1])
    predictions_blob_path = latest_file

    logger.info(f"Loading predictions from: {predictions_blob_path}")
    pqdata = ds.dataset(predictions_blob_path, filesystem=abfs)

    return (
        pqdata
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
        .assign(
            ds=lambda x: pd.to_datetime(x["ds"]),
        )
    )

def generate_plot(data, preds, models=[], levels=[]):
    logger.info("Generating plot")
    return plot_series(data, preds, level=levels, models=models, engine='plotly', palette='tab10')

def create_plotly_figure(df, models, confidence_levels):
    df['ds'] = pd.to_datetime(df['ds'])

    fig = go.Figure()

    # Adding the actual values
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines+markers',
        name='Real',
        line=dict(color='black', width=2),
        marker=dict(size=5)
    ))

    # Adding model predictions and confidence intervals
    for model in models:
        # Plotting the predicted values
        fig.add_trace(go.Scatter(
            x=df['ds'],
            y=df[model],
            mode='lines',
            name=f'{model}',
            line=dict(width=2)
        ))

        for conf in confidence_levels:
            lo_col = f'{model}-lo-{conf}'
            hi_col = f'{model}-hi-{conf}'

            # Plotting the confidence intervals correctly
            fig.add_trace(go.Scatter(
                x=df['ds'],
                y=df[hi_col],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df['ds'],
                y=df[lo_col],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(width=0),
                showlegend=False
            ))

    # Updating layout
    fig.update_layout(
        title="Previsões dos Modelos com Intervalos de Confiança",
        xaxis_title="Data",
        yaxis_title="Demanda (MW)",
        legend_title="Legenda",
        hovermode="x"
    )

    return fig