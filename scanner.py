import pandas as pd
from typing import Dict, List, Tuple
from data_handler import DataHandler
from indicators import TechnicalIndicators
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np

class StockScanner:
    def __init__(self, api_key: str, symbols: List[str] = None):
        """Initialize scanner with optional symbols list. If not provided, will fetch active stocks."""
        self.data_handler = DataHandler(api_key)
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        if symbols is None:
            self.symbols = self._get_active_stocks()
        else:
            self.symbols = symbols

        # Initialize progress tracking
        self.progress_data = {
            'total_batches': 0,
            'current_batch': 0,
            'total_stocks': 0,
            'processed_stocks': 0
        }

    def _get_active_stocks(self) -> List[str]:
        """Return a curated list of active stocks to scan"""
        # Start with a focused list of stocks for faster scanning
        return [
            # Large Cap Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            # Crypto & Mining
            "RIOT", "MARA", "MSTR",
            # Mid Cap Tech
            "AMD", "PLTR", "CRWD", "NET", "SNAP",
            # Growth & Momentum
            "SHOP", "SE", "MELI", "DKNG", "RBLX"
        ]

    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch stock data using Fintel data handler"""
        try:
            return self.data_handler.fetch_data(symbol, 'daily')
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def scan_rsi_opportunities(self) -> Dict[str, List[Dict]]:
        """
        Scan for RSI-based buying opportunities
        Categories: Oversold (<30), Neutral (30-70), Overbought (>70)
        """
        results = {
            'Strong Buy (RSI < 30)': [],
            'Neutral (30-50)': [],
            'Watch (50-70)': [],
            'Overbought (>70)': []
        }

        def analyze_rsi(symbol: str) -> Dict:
            try:
                df = self.fetch_stock_data(symbol)
                if df.empty or len(df) < 14:  # Need at least 14 days for RSI
                    return None

                df = TechnicalIndicators.add_indicators(df)
                latest = df.iloc[-1]
                rsi = latest['RSI']

                # Calculate RSI trend
                rsi_trend = df['RSI'].tail(5).diff().mean()
                trend_direction = "↗️ Rising" if rsi_trend > 0 else "↘️ Falling"

                return {
                    'symbol': symbol,
                    'rsi': rsi,
                    'price': latest['Close'],
                    'volume': latest['Volume'],
                    'trend': trend_direction,
                    'macd': latest['MACD'],
                    'whale_pct': latest['Whale_Volume_Pct']
                }
            except Exception as e:
                self.logger.error(f"Error analyzing RSI for {symbol}: {str(e)}")
                return None

        # Parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            analyses = list(executor.map(analyze_rsi, self.symbols))

        # Categorize results
        for analysis in analyses:
            if analysis is None:
                continue

            rsi = analysis['rsi']
            if rsi < 30:
                results['Strong Buy (RSI < 30)'].append(analysis)
            elif rsi < 50:
                results['Neutral (30-50)'].append(analysis)
            elif rsi < 70:
                results['Watch (50-70)'].append(analysis)
            else:
                results['Overbought (>70)'].append(analysis)

        # Sort each category by RSI
        for category in results:
            if 'Strong Buy' in category or 'Neutral' in category:
                # Sort ascending for buy opportunities (lower RSI is better)
                results[category].sort(key=lambda x: x['rsi'])
            else:
                # Sort descending for overbought conditions
                results[category].sort(key=lambda x: x['rsi'], reverse=True)

        return results

    def scan_macd_opportunities(self, macd_periods=(12, 26, 9)) -> Dict[str, List[Dict]]:
        """
        Scan for MACD-based buying opportunities
        Categories: Strong Buy (crossover), Bullish Momentum, Bearish Pressure, Strong Sell
        Only looks at the last 2 weeks of data for recent signals
        """
        results = {
            'Strong Buy (Crossover)': [],
            'Bullish Momentum': [],
            'Bearish Pressure': [],
            'Strong Sell (Crossover)': []  # Renamed to indicate crossover
        }

        def analyze_macd(symbol: str) -> Dict:
            try:
                df = self.fetch_stock_data(symbol)
                if df.empty or len(df) < max(macd_periods):  # Need at least longest period days
                    return None

                # Calculate indicators with custom MACD periods
                df = TechnicalIndicators.add_indicators(df, macd_periods=macd_periods)

                # Get only last 2 weeks of data for crossover detection
                recent_df = df.last('14D')
                if recent_df.empty:
                    recent_df = df.tail(14)  # Fallback to last 14 days if date index not available

                latest = df.iloc[-1]

                # Check for crossovers in recent data
                bullish_crossovers = []
                bearish_crossovers = []

                for i in range(1, len(recent_df)):
                    current_diff = recent_df['MACD'].iloc[i] - recent_df['MACD_Signal'].iloc[i]
                    prev_diff = recent_df['MACD'].iloc[i-1] - recent_df['MACD_Signal'].iloc[i-1]

                    # Bullish crossover (MACD crosses above Signal)
                    if prev_diff < 0 and current_diff > 0:
                        bullish_crossovers.append(recent_df.index[i])

                    # Bearish crossover (MACD crosses below Signal)
                    if prev_diff > 0 and current_diff < 0:
                        bearish_crossovers.append(recent_df.index[i])

                # Calculate MACD momentum
                macd_momentum = df['MACD'].tail(5).diff().mean()
                momentum_direction = "↗️ Rising" if macd_momentum > 0 else "↘️ Falling"

                return {
                    'symbol': symbol,
                    'macd': latest['MACD'],
                    'macd_signal': latest['MACD_Signal'],
                    'price': latest['Close'],
                    'rsi': latest['RSI'],
                    'volume': latest['Volume'],
                    'momentum': momentum_direction,
                    'bullish_crossovers': bullish_crossovers,
                    'bearish_crossovers': bearish_crossovers,
                    'latest_crossover': max(bullish_crossovers + bearish_crossovers) if bullish_crossovers or bearish_crossovers else None,
                    'crossover_type': 'bullish' if bullish_crossovers and (not bearish_crossovers or max(bullish_crossovers) > max(bearish_crossovers)) else 'bearish' if bearish_crossovers else None,
                    'whale_pct': latest['Whale_Volume_Pct']
                }
            except Exception as e:
                self.logger.error(f"Error analyzing MACD for {symbol}: {str(e)}")
                return None

        # Parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            analyses = list(executor.map(analyze_macd, self.symbols))

        # Categorize results
        for analysis in analyses:
            if analysis is None:
                continue

            if analysis['crossover_type'] == 'bullish' and analysis['latest_crossover']:
                results['Strong Buy (Crossover)'].append(analysis)
            elif analysis['crossover_type'] == 'bearish' and analysis['latest_crossover']:
                results['Strong Sell (Crossover)'].append(analysis)
            elif analysis['macd'] > analysis['macd_signal'] and analysis['momentum'] == "↗️ Rising":
                results['Bullish Momentum'].append(analysis)
            elif analysis['macd'] < analysis['macd_signal']:
                results['Bearish Pressure'].append(analysis)

        # Sort categories
        for category in results:
            if 'Buy' in category or 'Bullish' in category:
                # Sort by timestamp (most recent first) then by MACD strength
                results[category].sort(key=lambda x: (
                    pd.Timestamp.min if x.get('latest_crossover') is None else x['latest_crossover'],
                    x['macd'] - x['macd_signal']
                ), reverse=True)
            else:
                # Sort by timestamp (most recent first) then by MACD strength for sell signals
                results[category].sort(key=lambda x: (
                    pd.Timestamp.min if x.get('latest_crossover') is None else x['latest_crossover'],
                    x['macd_signal'] - x['macd']
                ), reverse=True)

        return results

    def analyze_stock(self, symbol: str) -> Dict:
        """Analyze a single stock for signal conditions"""
        try:
            # Fetch data
            df = self.fetch_stock_data(symbol)
            if df.empty or len(df) < 50:  # Ensure we have enough data points
                return {'symbol': symbol, 'error': 'Insufficient data points'}

            # Calculate indicators
            df = TechnicalIndicators.add_indicators(df)

            # Get latest data point
            latest = df.iloc[-1]

            # Calculate slopes using safe window
            macd_vals = df['MACD'].tail(3)
            if len(macd_vals) < 3:
                return {'symbol': symbol, 'error': 'Insufficient data for MACD calculation'}

            x = np.arange(len(macd_vals))
            macd_slope = np.polyfit(x, macd_vals, 1)[0] if len(macd_vals) > 1 else 0

            # Check individual conditions
            conditions = {
                'macd_below_5': latest['MACD'] < 5,
                'whale_accumulation_50': latest['Whale_Volume_Pct'] >= 50,
                'macd_slope_positive': macd_slope > 0,
                'rsi_below_40': latest['RSI'] < 40,
                'volume_above_average': latest['Volume'] > latest['Volume_SMA']
            }

            # Calculate match score (percentage of conditions met)
            met_conditions = sum(conditions.values())
            total_conditions = len(conditions)
            match_score = (met_conditions / total_conditions) * 100

            return {
                'symbol': symbol,
                'conditions_met': met_conditions,
                'total_conditions': total_conditions,
                'match_score': match_score,
                'details': conditions
            }

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

    def scan_stocks(self) -> List[Dict]:
        """
        Scan all stocks and return the top 25 matches sorted by match score
        """
        results = []
        valid_results = []
        errors = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.analyze_stock, self.symbols))

        # Separate valid results and errors
        for result in results:
            if 'error' in result:
                errors.append(result)
            else:
                valid_results.append(result)

        # Sort valid results by match score and take top 25
        valid_results.sort(key=lambda x: x['match_score'], reverse=True)
        top_matches = valid_results[:25] if valid_results else []

        # Log errors for debugging
        if errors:
            self.logger.warning(f"Encountered errors while scanning: {len(errors)} stocks failed")
            for error in errors:
                self.logger.debug(f"Scan error for {error['symbol']}: {error.get('error')}")

        return top_matches

    def scan_macd_divergences(self, window: int = 50, macd_periods: Tuple[int, int, int] = (12, 26, 9)) -> Dict[str, List[Dict]]:
        """
        Scan for MACD divergences (both bullish and bearish)
        Args:
            window: The lookback period for detecting divergences (default: 50 days)
            macd_periods: Tuple of (fast_period, slow_period, signal_period) for MACD calculation
                         Default is (12, 26, 9) for standard settings
                         Can use (8, 17, 9) for short-term or (21, 50, 9) for long-term
        Returns:
            Dictionary with bullish and bearish divergence patterns
        """
        results = {
            'Bullish Divergence': [],  # Price making lower lows, MACD making higher lows
            'Bearish Divergence': [],  # Price making higher highs, MACD making lower highs
        }

        def detect_divergences(df: pd.DataFrame, window: int = 50) -> Tuple[bool, bool]:
            """
            Helper function to detect both types of divergences
            Uses a longer window to capture more meaningful trend changes
            """
            # Get the last n periods
            recent_data = df.tail(window)

            # Find local minima and maxima
            price_mins = []
            price_maxs = []
            macd_mins = []
            macd_maxs = []

            # Use rolling windows to identify local extrema
            for i in range(2, len(recent_data) - 2):
                # Price extrema
                if recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-1] and \
                   recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+1]:
                    price_mins.append((i, recent_data['Low'].iloc[i]))

                if recent_data['High'].iloc[i] > recent_data['High'].iloc[i-1] and \
                   recent_data['High'].iloc[i] > recent_data['High'].iloc[i+1]:
                    price_maxs.append((i, recent_data['High'].iloc[i]))

                # MACD extrema
                if recent_data['MACD'].iloc[i] < recent_data['MACD'].iloc[i-1] and \
                   recent_data['MACD'].iloc[i] < recent_data['MACD'].iloc[i+1]:
                    macd_mins.append((i, recent_data['MACD'].iloc[i]))

                if recent_data['MACD'].iloc[i] > recent_data['MACD'].iloc[i-1] and \
                   recent_data['MACD'].iloc[i] > recent_data['MACD'].iloc[i+1]:
                    macd_maxs.append((i, recent_data['MACD'].iloc[i]))

            # Check for divergences
            bullish_div = False
            bearish_div = False

            # Need at least 2 points to compare trend
            if len(price_mins) >= 2 and len(macd_mins) >= 2:
                # Check last two price and MACD minima
                price_trend = price_mins[-1][1] - price_mins[-2][1]
                macd_trend = macd_mins[-1][1] - macd_mins[-2][1]

                # Bullish divergence: price making lower lows but MACD making higher lows
                if price_trend < 0 and macd_trend > 0:
                    bullish_div = True

            if len(price_maxs) >= 2 and len(macd_maxs) >= 2:
                # Check last two price and MACD maxima
                price_trend = price_maxs[-1][1] - price_maxs[-2][1]
                macd_trend = macd_maxs[-1][1] - macd_maxs[-2][1]

                # Bearish divergence: price making higher highs but MACD making lower highs
                if price_trend > 0 and macd_trend < 0:
                    bearish_div = True

            return bullish_div, bearish_div

        def analyze_divergence(symbol: str) -> Dict:
            """Analyze a single stock for MACD divergences"""
            try:
                df = self.fetch_stock_data(symbol)
                if df.empty or len(df) < 50:  # Need sufficient data points
                    return None

                # Calculate indicators with custom MACD periods
                df = TechnicalIndicators.add_indicators(df, macd_periods=macd_periods)
                latest = df.iloc[-1]

                # Detect divergences
                bullish_div, bearish_div = detect_divergences(df, window)

                return {
                    'symbol': symbol,
                    'price': latest['Close'],
                    'macd': latest['MACD'],
                    'macd_signal': latest['MACD_Signal'],
                    'rsi': latest['RSI'],
                    'volume': latest['Volume'],
                    'bullish_divergence': bullish_div,
                    'bearish_divergence': bearish_div,
                    'date': latest.name,
                    'macd_settings': f"{macd_periods[0]},{macd_periods[1]},{macd_periods[2]}"
                }
            except Exception as e:
                self.logger.error(f"Error analyzing divergence for {symbol}: {str(e)}")
                return None

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            analyses = list(executor.map(analyze_divergence, self.symbols))

        # Filter and categorize results
        for analysis in analyses:
            if analysis is None:
                continue

            if analysis['bullish_divergence']:
                results['Bullish Divergence'].append(analysis)
            if analysis['bearish_divergence']:
                results['Bearish Divergence'].append(analysis)

        # Sort by MACD value for each category
        for category in results:
            results[category].sort(key=lambda x: abs(x['macd'] - x['macd_signal']), reverse=True)

        return results

    def scan_whale_accumulation(self) -> Dict[str, List[Dict]]:
        """
        Scan stocks for whale accumulation using technical patterns and Fintel institutional data.
        Returns top 50 stocks per category, sorted by various metrics.
        """
        results = {
            'Below 30%': [],
            '30-40%': [],
            '40-50%': [],
            '50-60%': [],
            '60-70%': [],
            '70%+': [],
            # New categories for Smart Money signals
            'Smart Money Buying': [],
            'Smart Money Selling': [],
            'High Fund Sentiment': [],
            'Significant Insider Buying': []
        }

        def analyze_whale_activity(symbol: str) -> Dict:
            """Analyze a single stock for whale activity patterns using institutional data"""
            try:
                clean_symbol = symbol.replace('$', '')
                self.logger.info(f"Analyzing whale activity for {clean_symbol}")

                df = self.fetch_stock_data(clean_symbol)
                if df.empty or len(df) < 5:
                    return None

                # Calculate whale percentage based on volume analysis and institutional data
                df = TechnicalIndicators.calculate_whale_accumulation(df, symbol=clean_symbol, use_fintel=True)
                
                latest = df.iloc[-1]

                # Calculate trend
                trend_direction = "↗️" if latest['Close'] > df['Close'].iloc[-2] else "↘️"

                # Create base result dictionary
                result_dict = {
                    'symbol': clean_symbol,
                    'whale_percentage': latest['Whale_Volume_Pct'],
                    'current_price': latest['Close'],
                    'rsi': latest.get('RSI', 0),
                    'volume': latest['Volume'],
                    'trend': trend_direction,
                    'strength_score': latest['Whale_Volume_Pct']  # Default to technical whale percentage
                }
                
                # Add institutional metrics if available
                if 'Smart_Money_Score' in latest:
                    result_dict['smart_money_score'] = latest['Smart_Money_Score']
                    # Use Smart Money Score as strength score if available
                    result_dict['strength_score'] = latest['Smart_Money_Score']
                
                if 'Fund_Sentiment_Score' in latest:
                    result_dict['fund_sentiment_score'] = latest['Fund_Sentiment_Score']
                
                if 'Institutional_Ownership_Pct' in latest:
                    result_dict['institutional_ownership'] = latest['Institutional_Ownership_Pct']
                
                if 'Institutional_Transaction_Pct' in latest:
                    result_dict['institutional_transactions'] = latest['Institutional_Transaction_Pct']
                
                if 'Net_Insider_Activity' in latest:
                    result_dict['net_insider_activity'] = latest['Net_Insider_Activity']
                
                if 'Smart_Money_Signal' in latest:
                    result_dict['smart_money_signal'] = latest['Smart_Money_Signal']
                
                return result_dict

            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                return None

        # Process stocks in batches of 20
        batch_size = 20
        total_stocks = len(self.symbols)
        batches = [self.symbols[i:i + batch_size] for i in range(0, total_stocks, batch_size)]

        all_analyses = []

        # Update progress tracking before starting the analysis
        self.progress_data.update({
            'total_batches': len(batches),
            'total_stocks': total_stocks,
            'current_batch': 0,
            'processed_stocks': 0
        })

        for batch in batches:
            self.progress_data['current_batch'] += 1
            # Use ThreadPoolExecutor with smaller max_workers to avoid rate limiting
            with ThreadPoolExecutor(max_workers=3) as executor:
                batch_analyses = list(executor.map(analyze_whale_activity, batch))
                all_analyses.extend([a for a in batch_analyses if a is not None])
                self.progress_data['processed_stocks'] += len(batch)

        # Categorize results
        for analysis in all_analyses:
            whale_pct = analysis['whale_percentage']

            # Basic categories based on whale percentage from technical analysis
            if whale_pct >= 70:
                results['70%+'].append(analysis)
            elif whale_pct >= 60:
                results['60-70%'].append(analysis)
            elif whale_pct >= 50:
                results['50-60%'].append(analysis)
            elif whale_pct >= 40:
                results['40-50%'].append(analysis)
            elif whale_pct >= 30:
                results['30-40%'].append(analysis)
            else:
                results['Below 30%'].append(analysis)
            
            # Smart Money categories if available
            if 'smart_money_signal' in analysis:
                if analysis['smart_money_signal'] == 'buying':
                    results['Smart Money Buying'].append(analysis)
                elif analysis['smart_money_signal'] == 'selling':
                    results['Smart Money Selling'].append(analysis)
            
            # Fund Sentiment category
            if 'fund_sentiment_score' in analysis and analysis['fund_sentiment_score'] > 80:
                results['High Fund Sentiment'].append(analysis)
            
            # Insider buying category
            if 'net_insider_activity' in analysis and analysis['net_insider_activity'] > 500000:  # $500k of insider buying
                results['Significant Insider Buying'].append(analysis)

        # Sort and limit each category to top 50 stocks
        for category in results:
            # Choose appropriate sort key based on category
            if category == 'High Fund Sentiment':
                # Sort by fund sentiment score
                if results[category] and 'fund_sentiment_score' in results[category][0]:
                    results[category].sort(key=lambda x: x.get('fund_sentiment_score', 0), reverse=True)
            elif category == 'Smart Money Buying':
                # Sort by smart money score
                if results[category] and 'smart_money_score' in results[category][0]:
                    results[category].sort(key=lambda x: x.get('smart_money_score', 0), reverse=True)
            elif category == 'Significant Insider Buying':
                # Sort by net insider activity
                if results[category] and 'net_insider_activity' in results[category][0]:
                    results[category].sort(key=lambda x: x.get('net_insider_activity', 0), reverse=True)
            else:
                # Default sort by strength score
                results[category].sort(key=lambda x: x['strength_score'], reverse=True)
            
            # Limit to top 50 stocks
            results[category] = results[category][:50]
            
        # Remove empty categories
        results = {k: v for k, v in results.items() if v}

        return results

    def analyze_single_stock_whale_data(self, symbol: str) -> Dict:
        """Analyze whale data for a single stock with detailed logging, including institutional data"""
        try:
            clean_symbol = symbol.replace('$', '')
            self.logger.info(f"Analyzing single stock: {clean_symbol}")

            df = self.fetch_stock_data(clean_symbol)
            if df.empty:
                self.logger.warning(f"No data found for {clean_symbol}")
                return {'error': 'No data found'}

            # Calculate indicators with Fintel integration
            df = TechnicalIndicators.add_indicators(df, symbol=clean_symbol, use_fintel=True)

            latest = df.iloc[-1]
            whale_pct = latest['Whale_Volume_Pct']

            self.logger.info(f"Analysis results for {clean_symbol}:")
            self.logger.info(f"Whale percentage: {whale_pct:.2f}%")
            self.logger.info(f"Volume: {latest['Volume']}")
            
            # Create base result dictionary
            result = {
                'symbol': clean_symbol,
                'whale_percentage': whale_pct,
                'volume': latest['Volume'],
                'price': latest['Close'],
                'rsi': latest.get('RSI', 0),
                'macd': latest.get('MACD', 0),
                'macd_signal': latest.get('MACD_Signal', 0),
                'volume_z_score': latest.get('Volume_Z_Score', 0),
            }
            
            # Add institutional metrics if available
            if 'Smart_Money_Score' in latest:
                result['smart_money_score'] = latest['Smart_Money_Score']
                self.logger.info(f"Smart Money Score: {latest['Smart_Money_Score']:.2f}")
            
            if 'Fund_Sentiment_Score' in latest:
                result['fund_sentiment_score'] = latest['Fund_Sentiment_Score']
                self.logger.info(f"Fund Sentiment Score: {latest['Fund_Sentiment_Score']:.2f}")
            
            if 'Institutional_Ownership_Pct' in latest:
                result['institutional_ownership'] = latest['Institutional_Ownership_Pct']
                self.logger.info(f"Institutional Ownership: {latest['Institutional_Ownership_Pct']:.2f}%")
            
            if 'Institutional_Transaction_Pct' in latest:
                result['institutional_transactions'] = latest['Institutional_Transaction_Pct']
                self.logger.info(f"Recent Institutional Activity: {latest['Institutional_Transaction_Pct']:.2f}%")
            
            if 'Net_Insider_Activity' in latest:
                result['net_insider_activity'] = latest['Net_Insider_Activity']
                if latest['Net_Insider_Activity'] > 0:
                    activity_str = f"${latest['Net_Insider_Activity']/1000000:.2f}M net buying"
                else:
                    activity_str = f"${-latest['Net_Insider_Activity']/1000000:.2f}M net selling"
                self.logger.info(f"Net Insider Activity: {activity_str}")
            
            if 'Smart_Money_Signal' in latest:
                result['smart_money_signal'] = latest['Smart_Money_Signal']
                self.logger.info(f"Smart Money Signal: {latest['Smart_Money_Signal'].upper()}")
                
            # Add additional signals
            signals = []
            
            # Technical signals
            if whale_pct > 60:
                signals.append("Strong whale accumulation")
            
            # Smart Money signals
            if 'Smart_Money_Signal' in latest:
                if latest['Smart_Money_Signal'] == 'buying':
                    signals.append("Smart money buying signal")
                elif latest['Smart_Money_Signal'] == 'selling':
                    signals.append("Smart money selling signal")
            
            # Fund Sentiment signals
            if 'Fund_Sentiment_Score' in latest:
                if latest['Fund_Sentiment_Score'] > 80:
                    signals.append("Very high fund sentiment")
                elif latest['Fund_Sentiment_Score'] > 65:
                    signals.append("Above average fund sentiment")
                elif latest['Fund_Sentiment_Score'] < 35:
                    signals.append("Low fund sentiment")
            
            # Insider Activity signals
            if 'Net_Insider_Activity' in latest:
                if latest['Net_Insider_Activity'] > 1000000:  # >$1M buying
                    signals.append("Strong insider buying")
                elif latest['Net_Insider_Activity'] < -1000000:  # >$1M selling
                    signals.append("Strong insider selling")
            
            # Add signals to result
            result['signals'] = signals
            
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {'error': str(e)}

    def calculate_whale_accumulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to calculate whale accumulation"""
        # Use the established implementation from TechnicalIndicators
        return TechnicalIndicators.calculate_whale_accumulation(df)