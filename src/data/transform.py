import pandas as pd
import duckdb
import pendulum
from src.utils.logging_utils import logger
from typing import Any, Dict, List

from src.data.schemas.bronze_schema import schema as bronze_schema
from src.data.schemas.silver_schema import schema as silver_schema
from src.data.schemas.gold_schema import schema as gold_schema
from pandera import check_output

@check_output(bronze_schema)
def transform_to_bronze(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts raw data to a Bronze DataFrame.
    """
    logger.info("Transforming data to Bronze format")
    df = pd.DataFrame(data)
    return df

@check_output(silver_schema)
def transform_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms Bronze DataFrame to Silver by casting columns and filtering.
    """
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

@check_output(gold_schema)
def transform_to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates Silver DataFrame into Gold format by summing daily values.
    """
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