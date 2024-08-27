from datetime import datetime, timedelta
from pandas import Timestamp
from pandera import DataFrameSchema, Column, Check, Index

from src.config.config import config
from src.utils.date_utils import check_time_interval

schema = DataFrameSchema(
    columns={
        "data": Column(
            dtype="datetime64[ns]",
            checks=[
                Check.greater_than_or_equal_to(Timestamp(config.START_DATE)),
                Check.less_than_or_equal_to((datetime.now())),
                Check(lambda x: check_time_interval(x, timedelta(minutes=30)), element_wise=False, error="Time interval should be 30 minutes"),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "carga_mw": Column(
            dtype="float64",
            checks=[
                Check.less_than_or_equal_to(max_value=50_000),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
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
