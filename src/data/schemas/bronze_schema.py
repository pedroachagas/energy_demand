from pandera import DataFrameSchema, Column, Check, Index, MultiIndex
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import pendulum

from src.config.config import config
from src.utils.date_utils import check_time_interval


schema = DataFrameSchema(
    columns={
        "cod_areacarga": Column(
            dtype="object",
            checks=[
                Check.isin(['SP']),
                Check.str_length(2)
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "din_atualizacao": Column(
            dtype="object",
            checks=[
                Check.str_matches(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$'),
                Check(lambda x: x.str.len() == 24, error="din_atualizacao must be 24 characters long"),
                Check(lambda x: pd.to_datetime(x, utc=True) <= pd.Timestamp.now(tz='UTC'), error="Future dates are not allowed"),
                Check(lambda x: pd.to_datetime(x, utc=True) >= pd.Timestamp(config.START_DATE, tz='UTC'), error=f"Dates before {config.START_DATE} are not allowed")
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
        ),
        "dat_referencia": Column(
            dtype="object",
            checks=[
                Check.str_matches(r'^\d{4}-\d{2}-\d{2}$'),
                Check.str_length(10)
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "din_referenciautc": Column(
            dtype="object",
            checks=[
                Check.str_matches(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$'),
                Check(lambda x: pd.to_datetime(x, utc=True) <= pd.Timestamp.now(tz='UTC'), error="Future dates are not allowed"),
                Check(lambda x: pd.to_datetime(x, utc=True) >= pd.Timestamp(config.START_DATE, tz='UTC'), error=f"Dates before {config.START_DATE} are not allowed"),
                Check(lambda x: check_time_interval(x, timedelta(minutes=30)), element_wise=False, error="Time interval should be 30 minutes"),
                ],
            nullable=False,
            unique=True,
            coerce=False,
            required=True,
            regex=False,

        ),
        "val_cargaglobal": Column(
            dtype="float64",
            checks=[
                Check.less_than_or_equal_to(50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "val_cargaglobalcons": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=9837.16),
                Check.less_than_or_equal_to(50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "val_cargaglobalsmmgd": Column(
            dtype="float64",
            checks=[
                Check.less_than_or_equal_to(50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "val_cargasupervisionada": Column(
            dtype="float64",
            checks=[
                Check.less_than_or_equal_to(50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "val_carganaosupervisionada": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=137.9281),
                Check.less_than_or_equal_to(50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "val_cargammgd": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
        "val_consistencia": Column(
            dtype="float64",
            checks=[
                Check.less_than_or_equal_to(50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,

        ),
    },
    checks=None,
    index=Index(
        dtype="int64",
        checks=[
            Check.greater_than_or_equal_to(min_value=0.0),
        ],
        nullable=False,
        coerce=False,
        name=None,
        description=None,
        title=None,
    ),
    dtype=None,
    coerce=True,
    strict=True,
    name=None,
    ordered=False,
    unique=None,
    report_duplicates="all",
    unique_column_names=True,
    add_missing_columns=False,
    title=None,
    description=None,
)

