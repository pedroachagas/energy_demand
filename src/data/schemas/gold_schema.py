from datetime import datetime, timedelta
from pandas import Timestamp
from pandera import DataFrameSchema, Column, Check, Index

from src.config.config import config
from src.utils.date_utils import check_time_interval

LOWER_BOUND = 100_000
UPPER_BOUND = 1_100_000

schema = DataFrameSchema(
    columns={
        "date": Column(
            dtype="datetime64[ns]",
            checks=[
                Check.greater_than_or_equal_to(Timestamp(config.START_DATE)),
                Check.less_than_or_equal_to((datetime.now())),
                Check(lambda x: check_time_interval(x, timedelta(days=1)), element_wise=False, error="Time interval should be 1 day"),
            ],
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "daily_carga_mw": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(LOWER_BOUND),
                Check.less_than_or_equal_to(UPPER_BOUND),
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
            Check.greater_than_or_equal_to(0.0),
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
    unique_column_names=False,
    add_missing_columns=False,
    title=None,
    description=None,
)
