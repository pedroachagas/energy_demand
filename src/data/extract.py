import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pendulum
from src.utils.logging_utils import logger

def fetch_data(start_date, end_date, area_code):
    logger.info("Starting data fetch")
    start = pendulum.parse(start_date)
    end = pendulum.parse(end_date)
    logger.info(f"Fetching data from {start.to_date_string()} to {end.to_date_string()} for area code {area_code}")

    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start.add(days=90), end)

        url = f'https://apicarga.ons.org.br/prd/cargaverificada?dat_inicio={current_start.to_date_string()}&dat_fim={current_end.to_date_string()}&cod_areacarga={area_code}'
        logger.info(url)

        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))

        try:
            response = session.get(url)
            response.raise_for_status()
            chunk_data = response.json()

            expected_records = (current_end.add(days=1) - current_start).in_hours() * 2
            if len(chunk_data) != expected_records:
                raise ValueError(f"Data integrity check failed. Expected {expected_records} records, but received {len(chunk_data)} for period {current_start.to_date_string()} to {current_end.to_date_string()}")

            all_data.extend(chunk_data)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

        current_start = current_end.add(days=1)

    logger.info("Data fetch completed")
    return all_data