import requests
import pandas as pd
from bs4 import BeautifulSoup
import logging
import time
import random
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

class FintelDataHandler:
    def __init__(self, session_cookies=None):
        """
        Initialize with Fintel session cookies for authenticated access.
        
        Args:
            session_cookies: Dict of cookies from an authenticated Fintel session
        """
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://fintel.io"
        
        # Configure proper browser-like headers to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # Authentication status tracking
        self.auth_status = 'unknown'  # 'unknown', 'authenticated', or 'failed'
        
        # Initialize cookies
        self.cookies = session_cookies
        self.cache_dir = os.path.join(os.getcwd(), 'fintel_cache')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Log warning if no cookies provided
        if not session_cookies:
            self.logger.warning("No Fintel session cookies provided. Authentication will fail.")
            self.logger.info("Please add your Fintel session cookies in the sidebar to access institutional data.")
            
    def set_session_cookies(self, cookies: Dict[str, str]):
        """Set session cookies for authenticated access"""
        if not cookies:
            self.logger.warning("Empty cookies provided. Authentication will fail.")
            self.auth_status = 'failed'
            return
            
        # Check for required cookies
        required_cookies = []
        if 'cf_clearance' not in cookies:
            required_cookies.append('cf_clearance')
        if '_fintel_session' not in cookies and 'remember_user_token' not in cookies:
            required_cookies.append('_fintel_session or remember_user_token')
            
        if required_cookies:
            self.logger.warning(f"Missing required cookies: {', '.join(required_cookies)}. Authentication may fail.")
            self.auth_status = 'unknown'
        else:
            self.auth_status = 'pending'  # Will be verified on next data request
            
        # Update cookies
        self.cookies = cookies
        self.logger.info("Session cookies updated")
        
    def _make_request(self, url: str, max_retries: int = 3, backoff_factor: float = 0.5) -> Optional[str]:
        """Make a request with exponential backoff retry logic"""
        retry_count = 0
        
        # Check if cached version exists and is recent (less than 24 hours old)
        cache_path = self._get_cache_path(url)
        if os.path.exists(cache_path):
            cache_age = time.time() - os.path.getmtime(cache_path)
            if cache_age < 24 * 60 * 60:  # 24 hours in seconds
                self.logger.info(f"Using cached data for {url}")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        # Check if cookies are available
        missing_cookies = []
        if not self.cookies:
            self.logger.warning("No Fintel session cookies provided. Authentication will fail.")
            self.logger.info("Please add your Fintel session cookies in the sidebar to access institutional data.")
            return None
            
        # Check for critical cookies
        if 'cf_clearance' not in self.cookies:
            missing_cookies.append('cf_clearance')
        if '_fintel_session' not in self.cookies and 'remember_user_token' not in self.cookies:
            missing_cookies.append('_fintel_session or remember_user_token')
            
        if missing_cookies:
            self.logger.warning(f"Missing critical cookies: {', '.join(missing_cookies)}. Request may fail.")
            # We'll still try the request, but it might fail
        
        while retry_count < max_retries:
            try:
                # Add a small random delay to reduce the chance of rate limiting
                time.sleep(random.uniform(1.0, 3.0))
                
                self.logger.info(f"Making request to {url}")
                response = requests.get(url, headers=self.headers, cookies=self.cookies, timeout=10)
                
                if response.status_code == 200:
                    # Check if we actually got a login page instead of the data
                    if "Sign in" in response.text and "Login" in response.text and "/users/sign_in" in response.text:
                        self.logger.error("Authentication failed. Please update your Fintel.io session cookies.")
                        self.auth_status = 'failed'
                        return None
                    
                    # Check for Cloudflare challenge page
                    if "Just a moment" in response.text and "cf_chl_" in response.text:
                        self.logger.error("Cloudflare challenge detected. Please update your cf_clearance cookie.")
                        self.auth_status = 'failed'
                        return None
                    
                    # Check for JavaScript identifiers that might cause parse errors
                    js_identifiers = ['ade1cb7b', 'cf_chl_opt']
                    for identifier in js_identifiers:
                        if identifier in response.text:
                            self.logger.warning(f"Detected Cloudflare JavaScript ({identifier}). Cleaning response.")
                            # Only save the part of the response that doesn't contain the problematic JS
                            safe_response = self._clean_cloudflare_js(response.text)
                            with open(cache_path, 'w', encoding='utf-8') as f:
                                f.write(safe_response)
                            return safe_response
                        
                    # Authentication was successful
                    self.auth_status = 'authenticated'
                    
                    # Cache the response
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    return response.text
                elif response.status_code == 429:  # Too Many Requests
                    wait_time = backoff_factor * (2 ** retry_count)
                    self.logger.warning(f"Rate limited. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                elif response.status_code == 403:  # Forbidden - likely authentication issue
                    self.logger.error("Authentication error. Please update your Fintel.io session cookies.")
                    return None
                else:
                    self.logger.error(f"Request failed with status code: {response.status_code}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Request error: {str(e)}")
                
            retry_count += 1
            
        self.logger.error(f"Max retries exceeded for URL: {url}")
        return None
    
    def _get_cache_path(self, url: str) -> str:
        """Generate a cache file path from a URL"""
        # Create a filename from the URL (removing special characters)
        filename = url.replace(self.base_url, '').replace('/', '_').replace('\\', '_')
        if filename.startswith('_'):
            filename = filename[1:]
        return os.path.join(self.cache_dir, f"{filename}.html")
        
    def _clean_cloudflare_js(self, html_content: str) -> str:
        """
        Clean Cloudflare JavaScript from the HTML content to prevent parsing errors.
        This handles the case where a partial Cloudflare challenge response is received.
        """
        # If we detect Cloudflare JS, just extract the HTML body content without scripts
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove all script tags
            for script in soup.find_all('script'):
                script.decompose()
                
            # If we can find the main content, just return that
            main_content = soup.find('main') or soup.find('div', {'id': 'content'}) or soup.find('div', {'class': 'container'})
            if main_content:
                return str(main_content)
                
            # If we can't find specific content, return the body without scripts
            body = soup.find('body')
            if body:
                return str(body)
                
            # Last resort: return the cleaned HTML
            return str(soup)
        except Exception as e:
            self.logger.error(f"Error cleaning Cloudflare JS: {str(e)}")
            # Return empty string as a fallback
            return ""
    
    def _save_to_json_cache(self, data: Any, symbol: str, data_type: str) -> None:
        """Save data to JSON cache file"""
        filename = f"{symbol}_{data_type}.json"
        filepath = os.path.join(self.cache_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'data': data,
                'timestamp': datetime.now().isoformat()
            }, f, default=str)
    
    def _load_from_json_cache(self, symbol: str, data_type: str, max_age_hours: int = 24) -> Optional[Any]:
        """Load data from JSON cache if it exists and is recent"""
        filename = f"{symbol}_{data_type}.json"
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if age_hours <= max_age_hours:
                    return cache_data['data']
            except:
                return None
        
        return None
        
    def get_institutional_ownership(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get institutional ownership data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with institutional ownership data or None if data couldn't be fetched
        """
        # Verify authentication status
        if not self.cookies or self.auth_status == 'failed':
            self.logger.warning(f"Cannot get institutional ownership for {symbol} - not authenticated")
            self.auth_status = 'failed'
            return None
            
        # Check cache first
        cached_data = self._load_from_json_cache(symbol, 'institutional_ownership')
        if cached_data is not None:
            return pd.DataFrame(cached_data)
            
        url = f"{self.base_url}/ownership/stock/{symbol.upper()}"
        self.logger.info(f"Fetching institutional ownership data for {symbol} from {url}")
        
        html_content = self._make_request(url)
        if not html_content:
            self.logger.warning(f"Failed to retrieve HTML content for {symbol} institutional ownership")
            return None
            
        try:
            # Parse the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the institutional ownership table
            table = soup.find('table', {'class': 'table-ownership'})
            if not table:
                self.logger.warning(f"No institutional ownership table found for {symbol}")
                return None
                
            # Extract headers
            headers = [th.text.strip() for th in table.find('thead').find_all('th')]
            
            # Extract rows
            rows = []
            for tr in table.find('tbody').find_all('tr'):
                row_data = [td.text.strip() for td in tr.find_all('td')]
                if len(row_data) == len(headers):
                    rows.append(row_data)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Clean and convert data types
            df = self._clean_institutional_data(df)
            
            # Save to cache
            self._save_to_json_cache(df.to_dict('records'), symbol, 'institutional_ownership')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing institutional ownership data for {symbol}: {str(e)}")
            return None
            
    def _clean_institutional_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types in the institutional ownership DataFrame"""
        if df.empty:
            return df
            
        # Try to convert numeric columns
        for col in df.columns:
            # Skip non-numeric columns
            if col in ['Owner', 'Latest File', 'Source']:
                continue
                
            try:
                # Remove % and $ symbols and convert to numeric
                df[col] = df[col].str.replace('%', '').str.replace('$', '').str.replace(',', '').astype(float)
            except:
                # Keep as is if conversion fails
                pass
                
        return df
        
    def get_insider_trading(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get insider trading data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with insider trading data or None if data couldn't be fetched
        """
        # Verify authentication status
        if not self.cookies or self.auth_status == 'failed':
            self.logger.warning(f"Cannot get insider trading data for {symbol} - not authenticated")
            self.auth_status = 'failed'
            return None
            
        # Check cache first
        cached_data = self._load_from_json_cache(symbol, 'insider_trading')
        if cached_data is not None:
            return pd.DataFrame(cached_data)
            
        url = f"{self.base_url}/insiders/stock/{symbol.upper()}"
        self.logger.info(f"Fetching insider trading data for {symbol} from {url}")
        
        html_content = self._make_request(url)
        if not html_content:
            self.logger.warning(f"Failed to retrieve HTML content for {symbol} insider trading")
            return None
            
        try:
            # Parse the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the insider trading table
            table = soup.find('table', {'class': 'table-insiders'})
            if not table:
                self.logger.warning(f"No insider trading table found for {symbol}")
                return None
                
            # Extract headers
            headers = [th.text.strip() for th in table.find('thead').find_all('th')]
            
            # Extract rows
            rows = []
            for tr in table.find('tbody').find_all('tr'):
                row_data = [td.text.strip() for td in tr.find_all('td')]
                if len(row_data) == len(headers):
                    rows.append(row_data)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Clean and convert data types
            df = self._clean_insider_data(df)
            
            # Save to cache
            self._save_to_json_cache(df.to_dict('records'), symbol, 'insider_trading')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing insider trading data for {symbol}: {str(e)}")
            return None
            
    def _clean_insider_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types in the insider trading DataFrame"""
        if df.empty:
            return df
            
        # Try to convert date columns
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                pass
                
        # Try to convert numeric columns
        for col in ['Shares', 'Value', 'Price']:
            if col in df.columns:
                try:
                    # Remove $ and , characters and convert to numeric
                    df[col] = df[col].str.replace('$', '', regex=True).str.replace(',', '', regex=True).astype(float)
                except:
                    pass
                    
        return df
        
    def get_institutional_metrics(self, symbol: str) -> Dict:
        """
        Get key institutional metrics for a symbol.
        
        Returns a dictionary with:
        - institutional_ownership_percent: % of shares owned by institutions
        - institutional_transaction_percent: Net % bought/sold recently (positive = buying)
        - fund_sentiment_score: Fintel's proprietary score (0-100)
        - insider_ownership_percent: % of shares owned by insiders
        - insider_transaction_percent: Net % bought/sold recently (positive = buying)
        
        Note: Returns empty metrics with zeros if authentication fails
        """
        # Verify authentication status
        if not self.cookies or self.auth_status == 'failed':
            self.logger.warning(f"Cannot get institutional metrics for {symbol} - not authenticated")
            self.auth_status = 'failed'
            return {
                'institutional_ownership_percent': 0,
                'institutional_transaction_percent': 0,
                'fund_sentiment_score': 0,
                'insider_ownership_percent': 0,
                'insider_transaction_percent': 0,
                'total_institutional_holders': 0,
                'institutional_value_usd': 0
            }
            
        # Check cache first
        cached_data = self._load_from_json_cache(symbol, 'institutional_metrics')
        if cached_data is not None:
            return cached_data
            
        metrics = {
            'institutional_ownership_percent': 0,
            'institutional_transaction_percent': 0,
            'fund_sentiment_score': 0,
            'insider_ownership_percent': 0,
            'insider_transaction_percent': 0,
            'total_institutional_holders': 0,
            'institutional_value_usd': 0
        }
        
        # Get summary page
        url = f"{self.base_url}/stock/{symbol.upper()}"
        html_content = self._make_request(url)
        
        if not html_content:
            return metrics
            
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find metrics in the summary boxes
            metric_boxes = soup.find_all('div', {'class': 'info-box'})
            
            for box in metric_boxes:
                title = box.find('div', {'class': 'info-box-title'})
                value = box.find('div', {'class': 'info-box-value'})
                
                if not title or not value:
                    continue
                    
                title_text = title.text.strip()
                value_text = value.text.strip()
                
                # Extract percentage value
                try:
                    if '%' in value_text:
                        percent_value = float(value_text.replace('%', ''))
                    else:
                        percent_value = float(value_text)
                except:
                    continue
                    
                # Map to our metrics dictionary
                if 'Institutional Ownership' in title_text:
                    metrics['institutional_ownership_percent'] = percent_value
                elif 'Institutional Transactions' in title_text:
                    metrics['institutional_transaction_percent'] = percent_value
                elif 'Insider Ownership' in title_text:
                    metrics['insider_ownership_percent'] = percent_value
                elif 'Insider Transactions' in title_text:
                    metrics['insider_transaction_percent'] = percent_value
            
            # Find the Fund Sentiment Score
            fund_sentiment = soup.find('div', {'id': 'fund-sentiment'})
            if fund_sentiment:
                score_div = fund_sentiment.find('div', {'class': 'score'})
                if score_div:
                    score_text = score_div.text.strip()
                    try:
                        # Extract the score (format: "89.82 /100")
                        score = float(score_text.split('/')[0].strip())
                        metrics['fund_sentiment_score'] = score
                    except:
                        pass
            
            # Find total institutional holders
            inst_owners = soup.find('div', text='Institutional Owners')
            if inst_owners:
                parent = inst_owners.parent
                if parent:
                    value_div = parent.find('div', {'class': 'number'})
                    if value_div:
                        try:
                            metrics['total_institutional_holders'] = int(value_div.text.strip().replace(',', ''))
                        except:
                            pass
            
            # Find institutional value
            inst_value = soup.find('div', text='Institutional Value (Long)')
            if inst_value:
                parent = inst_value.parent
                if parent:
                    value_div = parent.find('div', {'class': 'number'})
                    if value_div:
                        value_text = value_div.text.strip()
                        try:
                            # Format is like "$2,306,055,132 USD ($1000)"
                            value_str = value_text.split('USD')[0].strip().replace('$', '').replace(',', '')
                            metrics['institutional_value_usd'] = float(value_str)
                        except:
                            pass
                            
            # Save to cache
            self._save_to_json_cache(metrics, symbol, 'institutional_metrics')
                    
        except Exception as e:
            self.logger.error(f"Error extracting institutional metrics for {symbol}: {str(e)}")
            
        return metrics
        
    def get_auth_status(self) -> str:
        """Get the current authentication status"""
        return self.auth_status
        
    def verify_authentication(self, test_symbol: str = "AAPL") -> bool:
        """
        Verify authentication by making a test request for institutional data.
        Returns True if authenticated, False otherwise.
        
        Args:
            test_symbol: Symbol to use for verification (defaults to AAPL)
        """
        if not self.cookies:
            self.auth_status = 'failed'
            return False
            
        # Try to get institutional metrics
        metrics = self.get_institutional_metrics(test_symbol)
        
        # If we got any non-zero data, authentication worked
        if metrics and (metrics.get('fund_sentiment_score', 0) > 0 or 
                       metrics.get('institutional_ownership_percent', 0) > 0):
            self.auth_status = 'authenticated'
            return True
        else:
            self.auth_status = 'failed'
            return False
        
    def get_fund_sentiment_score(self, symbol: str) -> float:
        """Get the Fund Sentiment Score (0-100) for a symbol"""
        metrics = self.get_institutional_metrics(symbol)
        return metrics.get('fund_sentiment_score', 0)
        
    def analyze_top_institutions(self, symbol: str, min_shares_change_pct: float = 5.0) -> Dict:
        """
        Analyze top institutions for a symbol and identify significant recent transactions.
        
        Args:
            symbol: Stock ticker symbol
            min_shares_change_pct: Minimum percentage change in shares to be considered significant
            
        Returns:
            Dictionary with analysis results
        """
        # Verify authentication status
        if not self.cookies or self.auth_status == 'failed':
            self.logger.warning(f"Cannot analyze top institutions for {symbol} - not authenticated")
            return {
                'symbol': symbol, 
                'significant_buyers': [], 
                'significant_sellers': [],
                'top_holders': [],
                'auth_required': True
            }
            
        df = self.get_institutional_ownership(symbol)
        if df is None or df.empty:
            return {
                'symbol': symbol, 
                'significant_buyers': [], 
                'significant_sellers': [],
                'top_holders': []
            }
            
        try:
            # Process the dataframe
            if 'Shares' in df.columns and 'Δ Shares (%)' in df.columns:
                # Find significant buyers
                buyers = df[df['Δ Shares (%)'] >= min_shares_change_pct].sort_values(
                    by='Δ Shares (%)', ascending=False
                ).head(5)
                
                # Find significant sellers
                sellers = df[df['Δ Shares (%)'] <= -min_shares_change_pct].sort_values(
                    by='Δ Shares (%)', ascending=True
                ).head(5)
                
                # Find top holders
                top_holders = df.sort_values(by='Shares', ascending=False).head(5)
                
                # Create result
                return {
                    'symbol': symbol,
                    'significant_buyers': buyers.to_dict('records') if not buyers.empty else [],
                    'significant_sellers': sellers.to_dict('records') if not sellers.empty else [],
                    'top_holders': top_holders.to_dict('records') if not top_holders.empty else []
                }
        except Exception as e:
            self.logger.error(f"Error analyzing top institutions for {symbol}: {str(e)}")
            
        return {
            'symbol': symbol, 
            'significant_buyers': [], 
            'significant_sellers': [],
            'top_holders': []
        }
        
    def analyze_insider_activity(self, symbol: str, min_value: float = 100000) -> Dict:
        """
        Analyze insider activity for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            min_value: Minimum transaction value to include
            
        Returns:
            Dictionary with analysis results
        """
        # Verify authentication status
        if not self.cookies or self.auth_status == 'failed':
            self.logger.warning(f"Cannot analyze insider activity for {symbol} - not authenticated")
            return {
                'symbol': symbol,
                'recent_buys': [],
                'recent_sells': [],
                'total_buy_value': 0,
                'total_sell_value': 0,
                'net_activity': 0,
                'auth_required': True
            }
            
        df = self.get_insider_trading(symbol)
        if df is None or df.empty:
            return {
                'symbol': symbol,
                'recent_buys': [],
                'recent_sells': [],
                'total_buy_value': 0,
                'total_sell_value': 0,
                'net_activity': 0
            }
            
        try:
            # Process the dataframe
            if 'Value' in df.columns and 'Type' in df.columns:
                # Filter by minimum value
                df = df[df['Value'] >= min_value]
                
                # Separate buys and sells
                buys = df[df['Type'].str.contains('Buy', case=False, na=False)]
                sells = df[df['Type'].str.contains('Sell', case=False, na=False)]
                
                # Calculate totals
                total_buy_value = buys['Value'].sum() if not buys.empty else 0
                total_sell_value = sells['Value'].sum() if not sells.empty else 0
                net_activity = total_buy_value - total_sell_value
                
                # Create result
                return {
                    'symbol': symbol,
                    'recent_buys': buys.to_dict('records') if not buys.empty else [],
                    'recent_sells': sells.to_dict('records') if not sells.empty else [],
                    'total_buy_value': total_buy_value,
                    'total_sell_value': total_sell_value,
                    'net_activity': net_activity
                }
        except Exception as e:
            self.logger.error(f"Error analyzing insider activity for {symbol}: {str(e)}")
            
        return {
            'symbol': symbol,
            'recent_buys': [],
            'recent_sells': [],
            'total_buy_value': 0,
            'total_sell_value': 0,
            'net_activity': 0
        }
        
    def get_composite_score(self, symbol: str) -> Dict:
        """
        Calculate a composite score based on institutional metrics, fund sentiment, and insider activity.
        This is our proprietary "Smart Money" indicator.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with scores
        """
        # Verify authentication status
        if not self.cookies or self.auth_status == 'failed':
            self.logger.warning(f"Cannot get composite score for {symbol} - not authenticated")
            return {
                'symbol': symbol,
                'composite_score': 0,
                'fund_sentiment': 0,
                'institutional_ownership': 0,
                'institutional_transactions': 0,
                'insider_score': 0,
                'net_insider_activity': 0,
                'auth_required': True
            }
            
        # Get institutional metrics
        metrics = self.get_institutional_metrics(symbol)
        
        # Get insider activity
        insider_activity = self.analyze_insider_activity(symbol)
        
        # Calculate the composite score
        fund_sentiment_weight = 0.5  # Fund sentiment is 50% of the score
        inst_ownership_weight = 0.2  # Institutional ownership is 20% of the score
        inst_transaction_weight = 0.2  # Recent institutional transactions are 20% of the score
        insider_weight = 0.1  # Insider transactions are 10% of the score
        
        # Normalize each component to a 0-100 scale
        fund_sentiment = metrics.get('fund_sentiment_score', 0)  # Already 0-100
        
        # Institutional ownership (normalize from 0-100% to 0-100 score)
        inst_ownership = min(100, metrics.get('institutional_ownership_percent', 0))
        
        # Institutional transactions (convert from -100 to +100 range to 0-100 score)
        inst_trans_raw = metrics.get('institutional_transaction_percent', 0)
        inst_trans = 50 + (inst_trans_raw / 2)  # Convert from -100/+100 to 0-100 scale
        
        # Insider activity (normalize based on net activity)
        net_insider = insider_activity.get('net_activity', 0)
        insider_score = 50  # Neutral by default
        if net_insider > 0:
            # Positive insider buying, increase score
            insider_score = min(100, 50 + (net_insider / 1000000) * 10)  # +10 points per million $
        elif net_insider < 0:
            # Negative insider selling, decrease score
            insider_score = max(0, 50 + (net_insider / 1000000) * 10)  # -10 points per million $
            
        # Calculate weighted composite score
        composite_score = (
            fund_sentiment * fund_sentiment_weight +
            inst_ownership * inst_ownership_weight +
            inst_trans * inst_transaction_weight +
            insider_score * insider_weight
        )
        
        return {
            'symbol': symbol,
            'composite_score': round(composite_score, 2),
            'fund_sentiment': round(fund_sentiment, 2),
            'institutional_ownership': round(inst_ownership, 2),
            'institutional_transactions': round(inst_trans_raw, 2),  # Original -100/+100 scale
            'insider_score': round(insider_score, 2),
            'net_insider_activity': net_insider
        }