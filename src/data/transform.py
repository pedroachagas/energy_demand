import pandas as pd
import duckdb
import pendulum
from src.utils.logging_utils import logger

def transform_to_bronze(data):
    logger.info("Transforming data to Bronze format")
    # In this case, we're not doing much transformation for the bronze layer
    # Just converting the raw data to a DataFrame
    df = pd.DataFrame(data)
    return df

def transform_to_silver(df):
    logger.info("Transforming data to Silver format")
    conn = duckdb.connect(":memory:")
    conn.register("bronze_data", df)

    silver_df = conn.execute("""
    SELECT
        CAST(din_referenciautc AS TIMESTAMP) AS data,
        CAST(val_cargaglobal AS DOUBLE) AS carga_mw
    FROM bronze_data
    WHERE din_referenciautc < ?
    """, [pendulum.now().to_date_string()]).df()

    return silver_df

def transform_to_gold(df):
    logger.info("Transforming data to Gold format")
    conn = duckdb.connect(":memory:")
    conn.register("silver_data", df)

    gold_df = conn.execute("""
    SELECT
        DATE_TRUNC('day', data) AS date,
        SUM(carga_mw) AS daily_carga_mw
    FROM silver_data
    GROUP BY DATE_TRUNC('day', data)
    ORDER BY date
    """).df()

    return gold_df