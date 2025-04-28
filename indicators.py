import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def identify_candle_pattern(df):
        """Identify candle patterns and trends"""
        df['Candle_Color'] = 'neutral'  # Default color

        # Determine candle color based on price movement
        df.loc[df['Close'] > df['Open'], 'Candle_Color'] = 'red'  # Bullish
        df.loc[df['Close'] < df['Open'], 'Candle_Color'] = 'yellow'  # Bearish

        # Detect persistent trends
        df['Trend'] = 'neutral'
        window = 5  # Look back period for trend

        # Uptrend detection (dark blue)
        up_trend = (df['Close'].rolling(window=window).mean().diff() > 0) & \
                  (df['Close'] > df['Close'].rolling(window=window).mean())
        df.loc[up_trend, 'Trend'] = 'dark_blue'

        # Downtrend detection (light blue)
        down_trend = (df['Close'].rolling(window=window).mean().diff() < 0) & \
                    (df['Close'] < df['Close'].rolling(window=window).mean())
        df.loc[down_trend, 'Trend'] = 'light_blue'

        return df

    @staticmethod
    def calculate_whale_accumulation(df, symbol=None, use_fintel=True):
        """
        Enhanced whale accumulation detection using volume analysis and Fintel institutional data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock ticker symbol (needed for Fintel data)
            use_fintel: Whether to use Fintel data
            
        Returns:
            DataFrame with whale accumulation metrics
        """
        # Calculate baseline metrics
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_STD'] = df['Volume'].rolling(window=20).std()

        # 1. Volume Analysis
        df['Volume_Z_Score'] = (df['Volume'] - df['Volume_SMA']) / df['Volume_STD']

        # 2. Price Impact Analysis
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Price_Impact'] = df['Price_Change'] * df['Volume'] / df['Volume_SMA']

        # 3. Order Flow Analysis
        df['Buy_Volume'] = df['Volume'].where(df['Close'] > df['Open'], 0)
        df['Sell_Volume'] = df['Volume'].where(df['Close'] <= df['Open'], 0)

        # Calculate whale activity scores - Lower thresholds for more sensitivity
        df['Whale_Score'] = (
            (df['Volume_Z_Score'] > 0.8).astype(int) +  # Even lower threshold for volume
            (abs(df['Volume_Price_Impact']) > 0.2).astype(int) +  # More sensitive to price impact
            (df['Volume'] > df['Volume_SMA'] * 0.7).astype(int)  # More lenient volume check
        )

        # Get institutional data if symbol is provided and Fintel is enabled
        institutional_metrics = None
        composite_score = None
        
        if use_fintel and symbol:
            try:
                import streamlit as st
                from fintel_data_handler import FintelDataHandler
                
                # Check if we already have a Fintel handler in the session state
                if 'fintel_handler' in st.session_state and st.session_state.fintel_handler:
                    fintel = st.session_state.fintel_handler
                else:
                    # Initialize FintelDataHandler
                    fintel = FintelDataHandler()
                    
                    # Store in session state for reuse
                    st.session_state.fintel_handler = fintel
                
                # Get institutional metrics
                institutional_metrics = fintel.get_institutional_metrics(symbol)
                
                # Update authentication status based on the response
                if institutional_metrics:
                    # Check if we got valid data
                    if institutional_metrics.get('fund_sentiment_score', 0) > 0 or \
                       institutional_metrics.get('institutional_ownership_percent', 0) > 0:
                        # Auth successful
                        st.session_state.fintel_auth_status = 'authenticated'
                    else:
                        # Auth failed or limited data
                        st.session_state.fintel_auth_status = 'failed'
                else:
                    # No data received
                    st.session_state.fintel_auth_status = 'failed'
                
                # Get composite score (combines fund sentiment, institutional ownership, and insider activity)
                composite_score = fintel.get_composite_score(symbol)
                
                # Adjust whale scores based on institutional data
                if institutional_metrics:
                    # Boost score if fund sentiment is high
                    fund_sentiment = institutional_metrics.get('fund_sentiment_score', 0)
                    if fund_sentiment > 75:  # High fund sentiment
                        df['Whale_Score'] += 1
                    
                    # Boost score if recent institutional buying
                    inst_trans = institutional_metrics.get('institutional_transaction_percent', 0)
                    if inst_trans > 2:  # Significant institutional buying
                        buy_days = df['Price_Change'] > 0
                        df.loc[buy_days, 'Whale_Score'] += 1
                    elif inst_trans < -2:  # Significant institutional selling
                        sell_days = df['Price_Change'] < 0
                        df.loc[sell_days, 'Whale_Score'] += 1
            
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error getting Fintel data: {str(e)}")
                
                # Set authentication status to failed on exception
                if 'streamlit' in locals() and 'st' in locals():
                    st.session_state.fintel_auth_status = 'failed'

        # Classify whale activity type
        df['Whale_Type'] = 'retail'  # Default classification
        whale_mask = df['Whale_Score'] >= 1  # Lowered from 2 to 1 for more sensitivity

        # Update classifications based on price change
        df.loc[whale_mask & (df['Price_Change'] > 0), 'Whale_Type'] = 'whale_accumulation'
        df.loc[whale_mask & (df['Price_Change'] < 0), 'Whale_Type'] = 'whale_distribution'

        # Calculate volume classifications
        df['Whale_Volume'] = df['Volume'].where(whale_mask, 0)
        df['Retail_Volume'] = df['Volume'].where(~whale_mask, df['Volume'])

        # Use a short window for dynamic percentage calculation
        window = 5  # Keep short window for responsive detection
        whale_sum = df['Whale_Volume'].rolling(window=window).sum()
        total_volume = df['Volume'].rolling(window=window).sum()

        # Calculate percentages based on total volume
        df['Whale_Volume_Pct'] = (whale_sum / total_volume * 100).fillna(0)
        df['Retail_Volume_Pct'] = 100 - df['Whale_Volume_Pct']

        # Add institutional metrics to the dataframe if available
        if institutional_metrics:
            # Add as constant columns since they're sourced from current data
            df['Institutional_Ownership_Pct'] = institutional_metrics.get('institutional_ownership_percent', 0)
            df['Institutional_Transaction_Pct'] = institutional_metrics.get('institutional_transaction_percent', 0)
            df['Fund_Sentiment_Score'] = institutional_metrics.get('fund_sentiment_score', 0)
            df['Insider_Ownership_Pct'] = institutional_metrics.get('insider_ownership_percent', 0)
            df['Insider_Transaction_Pct'] = institutional_metrics.get('insider_transaction_percent', 0)
        
        # Add composite score if available
        if composite_score:
            df['Smart_Money_Score'] = composite_score.get('composite_score', 0)
            df['Net_Insider_Activity'] = composite_score.get('net_insider_activity', 0)
            
            # Create smart money signal column
            df['Smart_Money_Signal'] = 'neutral'
            
            # Smart money buying signal
            if composite_score.get('composite_score', 0) > 75:
                df['Smart_Money_Signal'] = 'buying'
                # Boost whale score for all days with positive price change
                buy_days = df['Price_Change'] > 0
                df.loc[buy_days, 'Whale_Score'] += 1
                
            # Smart money selling signal
            elif composite_score.get('composite_score', 0) < 35:
                df['Smart_Money_Signal'] = 'selling'
                # Boost whale score for all days with negative price change
                sell_days = df['Price_Change'] < 0
                df.loc[sell_days, 'Whale_Score'] += 1

        return df

    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """Calculate MACD indicator with customizable periods"""
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })

    @staticmethod
    def calculate_multi_rsi(df, periods=[14, 21, 50]):
        """Calculate multiple RSI indicators"""
        # Default RSI for signal generation (14 period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate additional RSI periods
        for period in periods:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def calculate_volume_sma(df, period=20):
        """Calculate Volume SMA"""
        return df['Volume'].rolling(window=period).mean()

    @staticmethod
    def add_indicators(df, macd_periods=(12, 26, 9), symbol=None, use_fintel=True):
        """
        Add all indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            macd_periods: Tuple of (fast, slow, signal) periods for MACD
            symbol: Stock symbol for institutional data lookup
            use_fintel: Whether to use Fintel institutional data
            
        Returns:
            DataFrame with all indicators added
        """
        # Add candle patterns
        df = TechnicalIndicators.identify_candle_pattern(df)

        # Add whale accumulation patterns with institutional data if available
        df = TechnicalIndicators.calculate_whale_accumulation(df, symbol=symbol, use_fintel=use_fintel)

        # Calculate MACD with custom periods
        macd_data = TechnicalIndicators.calculate_macd(df, 
            fast=macd_periods[0], 
            slow=macd_periods[1], 
            signal=macd_periods[2]
        )
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['MACD_Signal']
        df['MACD_Histogram'] = macd_data['MACD_Histogram']

        # Calculate RSI
        df = TechnicalIndicators.calculate_multi_rsi(df)

        # Calculate Volume SMA
        df['Volume_SMA'] = TechnicalIndicators.calculate_volume_sma(df)

        return df