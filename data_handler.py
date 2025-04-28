import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional
import threading
from queue import Queue
import time
from polygon.rest import RESTClient

class DataHandler:
    def __init__(self, api_key: str):
        """Initialize with Polygon API key"""
        if not api_key:
            raise ValueError("Polygon API key must be provided")

        self.api_key = api_key
        self.client = RESTClient(api_key)
        self.timeframes = {
            'daily': 'day',
            'weekly': 'week',
            'monthly': 'month',
            'minute': 'minute',
            'hour': 'hour',
            'quarter': 'quarter',
            'year': 'year'
        }

        # Setup caching
        self._cache: Dict[Tuple[str, str, str], Tuple[pd.DataFrame, float]] = {}
        self._cache_lock = threading.Lock()
        self._cache_expiry = 3600  # Cache expires after 1 hour (increased to reduce API calls)

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.5  # Minimum 0.5 second between requests

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, symbol: str, timeframe: str, period: str = '6m') -> pd.DataFrame:
        """Fetch historical data from Polygon API"""
        try:
            # Format and validate symbol
            formatted_symbol = self._format_symbol(symbol)
            cache_key = (formatted_symbol, timeframe, period)

            # Check cache first
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                return cached_data

            # Calculate date range
            end_date = datetime.now()
            if period.endswith('y'):
                years = int(period[:-1])
                start_date = end_date - timedelta(days=years*365)
            elif period.endswith('m'):
                months = int(period[:-1])
                start_date = end_date - timedelta(days=months*30)
            else:
                raise ValueError("Invalid period format. Use 'Xy' for years or 'Xm' for months")

            self.logger.info(f"Fetching data for {formatted_symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Apply rate limiting
            self._apply_rate_limit()

            # Fetch data from Polygon with retry logic
            max_retries = 3
            retry_count = 0
            backoff_time = 1.0  # Start with 1 second

            while retry_count < max_retries:
                try:
                    # Fetch data from Polygon
                    aggs = self.client.get_aggs(
                        ticker=formatted_symbol,
                        multiplier=1,
                        timespan=self.timeframes[timeframe],
                        from_=start_date.strftime('%Y-%m-%d'),
                        to=end_date.strftime('%Y-%m-%d'),
                        adjusted=True
                    )
                    # If successful, break the retry loop
                    break
                except Exception as e:
                    # Check if it's a rate limit error (429)
                    if "429" in str(e):
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception(f"Rate limit exceeded after {max_retries} retries")
                        
                        # Log and wait with exponential backoff
                        wait_time = backoff_time * (2 ** (retry_count - 1))
                        self.logger.warning(f"Rate limit hit, retrying in {wait_time:.1f} seconds (retry {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        # Not a rate limit error, re-raise
                        raise

            if not aggs:
                self.logger.warning(f"No data found for {formatted_symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([{
                'Timestamp': pd.Timestamp(agg.timestamp/1000, unit='s').tz_localize(None),  # Convert milliseconds to seconds
                'Open': agg.open,
                'High': agg.high,
                'Low': agg.low,
                'Close': agg.close,
                'Volume': agg.volume
            } for agg in aggs])

            if df.empty:
                return df

            # Set index and sort
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Cache the results
            with self._cache_lock:
                self._cache[cache_key] = (df.copy(), time.time())

            return df

        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _format_symbol(self, symbol: str) -> str:
        """Format the symbol according to Polygon API requirements"""
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        # Remove any whitespace and convert to uppercase
        formatted = symbol.strip().upper()
        return formatted

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the fetched data"""
        if df.empty:
            return False
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        return all(col in df.columns for col in required_columns)

    def _get_cached_data(self, cache_key: Tuple[str, str, str]) -> Optional[pd.DataFrame]:
        """Get data from cache if valid"""
        with self._cache_lock:
            if cache_key in self._cache:
                df, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_expiry:
                    self.logger.info(f"Cache hit for {cache_key}")
                    return df.copy()
                else:
                    del self._cache[cache_key]
        return None
        
    def _apply_rate_limit(self):
        """Apply rate limiting to avoid API throttling"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        # If we've made a request recently, wait until the minimum interval has passed
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            time.sleep(sleep_time)
            
        # Update the last request time
        self._last_request_time = time.time()