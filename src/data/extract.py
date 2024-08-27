import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry # type: ignore
import pendulum
from pendulum import DateTime
from src.utils.logging_utils import logger
from typing import List, Dict, Any

def fetch_data(start_date: str, end_date: str, area_code: str) -> List[Dict[str, Any]]:
    """
    Fetches data from the API for a given date range and area code.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        area_code (str): The area code for the data fetch.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the fetched data.
    """
    logger.info("Starting data fetch")

    # Force the parse result to be treated as DateTime.
    start = pendulum.parse(start_date)
    if isinstance(start, pendulum.DateTime):
        start_dt: DateTime = start
    else:
        raise TypeError(f"Expected a DateTime object, but got {type(start)}")

    end = pendulum.parse(end_date)
    if isinstance(end, pendulum.DateTime):
        end_dt: DateTime = end
    else:
        raise TypeError(f"Expected a DateTime object, but got {type(end)}")

    logger.info(f"Fetching data from {start_dt.to_date_string()} to {end_dt.to_date_string()} for area code {area_code}")

    all_data: List[Dict[str, Any]] = []
    current_start: DateTime = start_dt

    while current_start < end_dt:
        current_end: DateTime = min(current_start.add(days=90), end_dt)

        url = (f'https://apicarga.ons.org.br/prd/cargaverificada?dat_inicio='
               f'{current_start.to_date_string()}&dat_fim={current_end.to_date_string()}'
               f'&cod_areacarga={area_code}')
        logger.info(url)

        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))

        try:
            response = session.get(url)
            response.raise_for_status()
            chunk_data: List[Dict[str, Any]] = response.json()

            expected_records: int = (current_end.add(days=1) - current_start).in_hours() * 2
            if len(chunk_data) != expected_records:
                raise ValueError(
                    f"Data integrity check failed. Expected {expected_records} records, "
                    f"but received {len(chunk_data)} for period {current_start.to_date_string()} to {current_end.to_date_string()}"
                )

            all_data.extend(chunk_data)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

        current_start = current_end.add(days=1)

    logger.info("Data fetch completed")
    return all_data
