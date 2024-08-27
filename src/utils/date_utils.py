import pendulum
import pandas as pd
from datetime import timedelta
from typing import Tuple, List, Optional

def parse_date(date_string: str) -> pendulum.Date:
    """
    Parse a date string into a pendulum.Date object.

    Args:
        date_string (str): Date string in the format 'YYYY-MM-DD'.

    Returns:
        pendulum.Date: Parsed date object.
    """
    parsed = pendulum.parse(date_string)
    if isinstance(parsed, pendulum.DateTime):
        return parsed.date()  # Explicitly return the date part
    elif isinstance(parsed, pendulum.Date):
        return parsed
    else:
        raise ValueError("Invalid date string")

def format_date(date: pendulum.Date) -> str:
    """
    Format a pendulum.Date object into a string.

    Args:
        date (pendulum.Date): Date object to format.

    Returns:
        str: Formatted date string in the format 'YYYYMMDD'.
    """
    return date.format('YYYYMMDD')

def get_date_range(start_date: str, end_date: Optional[str] = None) -> Tuple[pendulum.Date, pendulum.Date]:
    """
    Get a date range from start_date to end_date (or yesterday if not provided).

    Args:
        start_date (str): Start date string in the format 'YYYY-MM-DD'.
        end_date (Optional[str]): End date string in the format 'YYYY-MM-DD'. Defaults to None.

    Returns:
        Tuple[pendulum.Date, pendulum.Date]: Tuple containing start and end dates.
    """
    start = parse_date(start_date)
    end = parse_date(end_date) if end_date else pendulum.yesterday().date()
    return start, end

def generate_date_chunks(start_date: pendulum.Date, end_date: pendulum.Date, chunk_size: int = 90) -> List[Tuple[pendulum.Date, pendulum.Date]]:
    """
    Generate a list of date chunks between start_date and end_date.

    Args:
        start_date (pendulum.Date): Start date.
        end_date (pendulum.Date): End date.
        chunk_size (int, optional): Size of each chunk in days. Defaults to 90.

    Returns:
        List[Tuple[pendulum.Date, pendulum.Date]]: List of tuples containing start and end dates for each chunk.
    """
    chunks = []
    current_start = start_date

    while current_start <= end_date:
        chunk_end = min(current_start.add(days=chunk_size - 1), end_date)
        chunks.append((current_start, chunk_end))
        current_start = chunk_end.add(days=1)

    return chunks

def is_valid_date_range(start_date: pendulum.Date, end_date: pendulum.Date) -> bool:
    """
    Check if the given date range is valid (start_date <= end_date and end_date <= today).

    Args:
        start_date (pendulum.Date): Start date.
        end_date (pendulum.Date): End date.

    Returns:
        bool: True if the date range is valid, False otherwise.
    """
    today = pendulum.today().date()
    return start_date <= end_date <= today

def check_time_interval(series: pd.Series, interval: timedelta) -> bool:
    times = pd.to_datetime(series, utc=True)
    sorted_times = times.sort_values()
    time_diff = sorted_times.diff().dropna()
    condition = time_diff == interval
    return bool(condition.all())