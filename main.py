import streamlit as st
import pandas as pd
import time
import os
from data_handler import DataHandler
from indicators import TechnicalIndicators
from signals import SignalGenerator
from visualization import ChartVisualizer
from backtester import Backtester
from market_insights import MarketInsights
from scanner import StockScanner

def main():
    st.set_page_config(layout="wide")
    st.title("Multi-Timeframe Technical Analysis")

    # Initialize session states for components
    if 'scanner' not in st.session_state:
        st.session_state.scanner = None
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = None
    if 'fintel_handler' not in st.session_state:
        st.session_state.fintel_handler = None

    # Sidebar inputs
    st.sidebar.header("Parameters")
    symbol = st.sidebar.text_input("Symbol", value="AAPL")

    # Add configuration parameters
    st.sidebar.subheader("Indicator Settings")
    rsi_oversold = st.sidebar.slider("RSI Oversold Threshold", 20, 40, 30)
    rsi_overbought = st.sidebar.slider("RSI Overbought Threshold", 60, 80, 70)
    whale_threshold = st.sidebar.slider("Whale Activity Threshold (%)", 30, 70, 50)

    # MACD settings in a collapsible section
    with st.sidebar.expander("MACD Settings"):
        macd_fast = st.number_input("MACD Fast Period", min_value=5, max_value=50, value=12)
        macd_slow = st.number_input("MACD Slow Period", min_value=10, max_value=100, value=26)
        macd_signal = st.number_input("MACD Signal Period", min_value=5, max_value=50, value=9)
    macd_periods = (macd_fast, macd_slow, macd_signal)

    # Get API key from environment variable
    api_key = os.environ.get("POLYGON_API_KEY")
    
    if not api_key:
        st.error("Polygon API key is missing. Please contact the administrator.")
        return

    # Add Fintel integration section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fintel Integration")
    
    # Initialize session state for Fintel cookies
    if 'fintel_cookies' not in st.session_state:
        st.session_state.fintel_cookies = {}
    if 'use_fintel' not in st.session_state:
        st.session_state.use_fintel = True
    if 'fintel_auth_status' not in st.session_state:
        st.session_state.fintel_auth_status = 'unknown'
    if 'fintel_handler' not in st.session_state:
        st.session_state.fintel_handler = None
    if 'verify_fintel_auth' not in st.session_state:
        st.session_state.verify_fintel_auth = False
    
    # Toggle for using Fintel data
    st.session_state.use_fintel = st.sidebar.checkbox("Use Fintel institutional data", value=st.session_state.use_fintel)
    
    # Show auth status indicator
    if st.session_state.use_fintel:
        if st.session_state.fintel_auth_status == 'authenticated':
            st.sidebar.success("✅ Authenticated with Fintel")
        elif st.session_state.fintel_auth_status == 'failed':
            st.sidebar.error("❌ Authentication failed - add your Fintel cookies below")
        else:
            st.sidebar.info("⚠️ Fintel authentication status unknown")
    
    # Fintel cookies input
    with st.sidebar.expander("Fintel Login (Optional)"):
        st.markdown("Enter your Fintel session cookies to access institutional data:")
        
        # Show help box for authentication failures
        if st.session_state.fintel_auth_status == 'failed':
            st.warning("""
            ⚠️ **Authentication Failed**
            
            The system was unable to authenticate with Fintel.io. Please follow the instructions below to add your session cookies.
            
            **Benefits of Fintel authentication:**
            - Access to institutional ownership data
            - Fund sentiment scores
            - Insider trading activity
            - Smart Money indicators
            """)
        
        # Helper text
        st.markdown("""
        **How to get cookies**:
        1. Log in to Fintel.io in your browser
        2. Open developer tools (F12 or right-click > Inspect)
        3. Go to Application tab > Cookies > https://fintel.io
        4. Copy the values for these cookies:
           - `cf_clearance` (required - Cloudflare protection)
           - `_fintel_session` (recommended - login session)
           - `remember_user_token` (alternative to _fintel_session)
        """)
        
        # Cookie inputs
        cf_clearance = st.text_input("Cloudflare Clearance (cf_clearance) - Required", type="password")
        session_cookie = st.text_input("Session Cookie (_fintel_session)", type="password")
        remember_cookie = st.text_input("Remember Cookie (remember_user_token)", type="password")
        
        if st.button("Save Cookies"):
            if cf_clearance:
                st.session_state.fintel_cookies['cf_clearance'] = cf_clearance
            if session_cookie:
                st.session_state.fintel_cookies['_fintel_session'] = session_cookie
            if remember_cookie:
                st.session_state.fintel_cookies['remember_user_token'] = remember_cookie
                
            # Check if required cookies are present
            if 'cf_clearance' not in st.session_state.fintel_cookies:
                st.error("❌ Missing required Cloudflare cookie (cf_clearance). Authentication will likely fail.")
            
            # Set authentication status to pending verification
            st.session_state.fintel_auth_status = 'pending'
            
            # This will be updated to the correct status after verification
            
            # Ensure Fintel handler is initialized with new cookies
            if 'fintel_handler' in st.session_state and st.session_state.fintel_handler:
                st.session_state.fintel_handler.set_session_cookies(st.session_state.fintel_cookies)
                # Update status from handler to keep in sync
                st.session_state.fintel_auth_status = st.session_state.fintel_handler.get_auth_status()
            else:
                from fintel_data_handler import FintelDataHandler
                st.session_state.fintel_handler = FintelDataHandler(st.session_state.fintel_cookies)
                # Update status from handler to keep in sync
                st.session_state.fintel_auth_status = st.session_state.fintel_handler.get_auth_status()
            
            # Queue verification on next data load
            st.session_state.verify_fintel_auth = True
            
            st.success("Fintel cookies saved! Authentication will be verified with the next data load.")
    
    # Add attribution
    st.sidebar.markdown("---")
    st.sidebar.markdown("Data provided by [Polygon.io](https://polygon.io) and [Fintel.io](https://fintel.io)")

    try:
        # Initialize components with API key
        if not st.session_state.data_handler or api_key != getattr(st.session_state.data_handler, 'api_key', None):
            st.session_state.data_handler = DataHandler(api_key)
            st.session_state.scanner = StockScanner(api_key)
        
        # Initialize Fintel handler with session cookies if available
        if st.session_state.use_fintel:
            from fintel_data_handler import FintelDataHandler
            if not st.session_state.fintel_handler:
                st.session_state.fintel_handler = FintelDataHandler(st.session_state.fintel_cookies)
                # Sync the auth status with the handler's status
                st.session_state.fintel_auth_status = st.session_state.fintel_handler.get_auth_status()
            elif st.session_state.fintel_cookies:
                # Update cookies if available
                st.session_state.fintel_handler.set_session_cookies(st.session_state.fintel_cookies)
                st.session_state.fintel_auth_status = st.session_state.fintel_handler.get_auth_status()

        # Initialize other components with configuration parameters
        signal_generator = SignalGenerator(rsi_oversold, rsi_overbought, whale_threshold)
        backtester = Backtester()

        # Fetch data for different timeframes
        timeframes = ['daily', 'weekly', 'monthly']
        data = {}
        signals = {}

        with st.spinner('Fetching market data...'):
            # Handle Fintel authentication verification if requested
            if st.session_state.verify_fintel_auth and st.session_state.use_fintel:
                with st.spinner('Verifying Fintel authentication...'):
                    try:
                        # Using a test symbol for verification
                        test_symbol = "AAPL"  # A common stock that will have institutional data
                        
                        # Make a test request through fintel_handler
                        from fintel_data_handler import FintelDataHandler
                        if not st.session_state.fintel_handler:
                            st.session_state.fintel_handler = FintelDataHandler(st.session_state.fintel_cookies)
                        elif st.session_state.fintel_cookies:
                            st.session_state.fintel_handler.set_session_cookies(st.session_state.fintel_cookies)
                        
                        # Use the new verification method
                        if st.session_state.fintel_handler.verify_authentication(test_symbol):
                            st.session_state.fintel_auth_status = 'authenticated'
                            st.sidebar.success("✅ Successfully authenticated with Fintel!")
                        else:
                            st.session_state.fintel_auth_status = 'failed'
                            st.sidebar.error("❌ Failed to authenticate with Fintel. Check your cookies.")
                            
                        # Update auth status from handler
                        st.session_state.fintel_auth_status = st.session_state.fintel_handler.get_auth_status()
                        
                    except Exception as e:
                        st.session_state.fintel_auth_status = 'failed'
                        st.sidebar.error(f"❌ Error authenticating with Fintel: {str(e)}")
                    
                    # Reset verification flag
                    st.session_state.verify_fintel_auth = False
            
            # Process each timeframe
            for timeframe in timeframes:
                try:
                    # Fetch and validate data
                    df = st.session_state.data_handler.fetch_data(symbol, timeframe)
                    if not st.session_state.data_handler.validate_data(df):
                        st.error(f"Invalid data for {timeframe} timeframe")
                        return

                    # Calculate indicators with custom MACD periods and Fintel data if enabled
                    df = TechnicalIndicators.add_indicators(
                        df, 
                        macd_periods=macd_periods,
                        symbol=symbol if st.session_state.use_fintel else None,
                        use_fintel=st.session_state.use_fintel
                    )
                    data[timeframe] = df

                    # Generate signals
                    signals[timeframe] = signal_generator.generate_signals(df)
                except Exception as e:
                    st.error(f"Error processing {timeframe} data: {str(e)}")
                    return

        # Combine signals from different timeframes
        combined_signals = signal_generator.combine_timeframe_signals(
            signals['daily'], signals['weekly'], signals['monthly']
        )

        # Create tabs for different timeframes
        tabs = st.tabs(timeframes)

        for tab, timeframe in zip(tabs, timeframes):
            with tab:
                df = data[timeframe]

                # Create and display chart
                fig = ChartVisualizer.create_analysis_chart(
                    df,
                    signals[timeframe]
                )
                st.plotly_chart(fig, use_container_width=True)

        # Display combined signals analysis
        st.header("Multi-Timeframe Signal Analysis")
        strong_signals = combined_signals[combined_signals['Strong_Signal'] != 0]

        if not strong_signals.empty:
            st.write("Strong Signals (Confirmed across all timeframes):")
            signal_df = pd.DataFrame({
                'Date': strong_signals.index,
                'Signal': strong_signals['Strong_Signal'].map({1: 'Buy', -1: 'Sell'})
            })
            st.dataframe(signal_df)
        else:
            st.write("No strong signals currently")

        # Stock Scanner Section
        st.header("Market Scanner")
        scanner_tab1, scanner_tab2, scanner_tab3, scanner_tab4, scanner_tab5 = st.tabs([
            "Signal Scanner",
            "RSI Scanner",
            "MACD Scanner",
            "Whale Accumulation Scanner",
            "MACD Divergence Scanner"
        ])

        # Signal Scanner Tab
        with scanner_tab1:
            if st.button("Scan for Signal Matches", key="signal_scanner"):
                with st.spinner("Scanning for signal matches..."):
                    try:
                        top_matches = st.session_state.scanner.scan_stocks()
                        if top_matches:
                            st.subheader("Top Signal Matches")
                            results_df = pd.DataFrame([{
                                'Symbol': match['symbol'],
                                'Match Score': f"{match['match_score']:.1f}%",
                                'Conditions Met': f"{match['conditions_met']}/{match['total_conditions']}"
                            } for match in top_matches])
                            st.dataframe(results_df)

                            for match in top_matches:
                                with st.expander(f"{match['symbol']} - Details"):
                                    conditions = match['details']
                                    for condition, met in conditions.items():
                                        icon = "✅" if met else "❌"
                                        st.write(f"{icon} {condition.replace('_', ' ').title()}")
                        else:
                            st.warning("No matching signals found in the current scan.")
                    except Exception as e:
                        st.error(f"Error during signal scan: {str(e)}")

        # RSI Scanner Tab
        with scanner_tab2:
            if st.button("Scan RSI Levels", key="rsi_scanner"):
                with st.spinner("Scanning RSI levels..."):
                    try:
                        rsi_results = st.session_state.scanner.scan_rsi_opportunities()
                        rsi_categories = ['Strong Buy (RSI < 30)', 'Neutral (30-50)',
                                        'Watch (50-70)', 'Overbought (>70)']
                        rsi_tabs = st.tabs(rsi_categories)

                        for tab, category in zip(rsi_tabs, rsi_categories):
                            with tab:
                                stocks = rsi_results[category]
                                if stocks:
                                    df = pd.DataFrame([{
                                        'Symbol': stock['symbol'],
                                        'RSI': f"{stock['rsi']:.1f}",
                                        'Price': f"${stock['price']:.2f}",
                                        'Trend': stock['trend'],
                                        'MACD': f"{stock['macd']:.2f}",
                                        'Whale %': f"{stock['whale_pct']:.1f}%",
                                        'Volume': f"{stock['volume']:,.0f}"
                                    } for stock in stocks])
                                    st.dataframe(df)
                                else:
                                    st.info(f"No stocks found in {category} category")
                    except Exception as e:
                        st.error(f"Error during RSI scan: {str(e)}")

        # MACD Scanner Tab
        with scanner_tab3:
            if st.button("Scan MACD Signals", key="macd_scanner"):
                with st.spinner("Scanning MACD signals..."):
                    try:
                        macd_results = st.session_state.scanner.scan_macd_opportunities(macd_periods=macd_periods)
                        macd_categories = ['Strong Buy (Crossover)', 'Bullish Momentum',
                                         'Bearish Pressure', 'Strong Sell (Crossover)']
                        macd_tabs = st.tabs(macd_categories)

                        for tab, category in zip(macd_tabs, macd_categories):
                            with tab:
                                stocks = macd_results[category]
                                if stocks:
                                    df = pd.DataFrame([{
                                        'Symbol': stock['symbol'],
                                        'MACD': f"{stock['macd']:.2f}",
                                        'Signal': f"{stock['macd_signal']:.2f}",
                                        'Price': f"${stock['price']:.2f}",
                                        'RSI': f"{stock['rsi']:.1f}",
                                        'Momentum': stock['momentum'],
                                        'Whale %': f"{stock['whale_pct']:.1f}%",
                                        'Volume': f"{stock['volume']:,.0f}",
                                        'Latest Signal': stock['latest_crossover'].strftime('%Y-%m-%d') if stock.get('latest_crossover') else 'N/A'
                                    } for stock in stocks])
                                    st.dataframe(df)
                                else:
                                    st.info(f"No stocks found in {category} category")
                    except Exception as e:
                        st.error(f"Error during MACD scan: {str(e)}")

        # Whale Scanner Tab
        with scanner_tab4:
            if st.button("Scan for Whale Accumulation", key="whale_scanner"):
                with st.spinner("Initializing whale activity scanner..."):
                    try:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Start the scan
                        whale_results = st.session_state.scanner.scan_whale_accumulation()

                        # Create tabs for different categories including Smart Money indicators
                        tab_categories = []
                        
                        # Traditional whale percentage categories
                        technical_categories = ['Below 30%', '30-40%', '40-50%', '50-60%', '60-70%', '70%+']
                        
                        # Add Smart Money categories if available in results and using Fintel data
                        institutional_categories = []
                        if st.session_state.use_fintel:
                            for category in ['Smart Money Buying', 'Smart Money Selling', 'High Fund Sentiment', 'Significant Insider Buying']:
                                if category in whale_results:
                                    institutional_categories.append(category)
                        
                        # Combine all categories
                        tab_categories = technical_categories + institutional_categories
                        
                        # Create tabs for all categories
                        range_tabs = st.tabs(tab_categories)

                        for tab, category in zip(range_tabs, tab_categories):
                            with tab:
                                if category in whale_results and whale_results[category]:
                                    stocks = whale_results[category]
                                    
                                    # Create DataFrame with appropriate columns based on category
                                    columns = {
                                        'Symbol': 'symbol',
                                        'Whale %': lambda s: f"{s['whale_percentage']:.1f}%",
                                        'Strength': lambda s: f"{s['strength_score']:.0f}/100",
                                        'Price': lambda s: f"${s['current_price']:.2f}",
                                        'RSI': lambda s: f"{s['rsi']:.1f}%",
                                        'Volume': lambda s: f"{s['volume']:,.0f}",
                                        'Trend': 'trend'
                                    }
                                    
                                    # Add institutional metrics for smart money categories
                                    if category in ['Smart Money Buying', 'Smart Money Selling', 
                                                   'High Fund Sentiment', 'Significant Insider Buying']:
                                        if 'smart_money_score' in stocks[0]:
                                            columns['Smart Money'] = lambda s: f"{s.get('smart_money_score', 0):.1f}/100"
                                        if 'fund_sentiment_score' in stocks[0]:
                                            columns['Fund Sentiment'] = lambda s: f"{s.get('fund_sentiment_score', 0):.1f}/100"
                                        if 'institutional_ownership' in stocks[0]:
                                            columns['Inst. Own%'] = lambda s: f"{s.get('institutional_ownership', 0):.1f}%"
                                        if 'net_insider_activity' in stocks[0]:
                                            columns['Insider $'] = lambda s: f"${s.get('net_insider_activity', 0)/1000000:.1f}M"
                                    
                                    # Create DataFrame by applying the column functions
                                    df_data = []
                                    for stock in stocks:
                                        row = {}
                                        for col_name, accessor in columns.items():
                                            if callable(accessor):
                                                try:
                                                    row[col_name] = accessor(stock)
                                                except:
                                                    row[col_name] = "N/A"
                                            else:
                                                row[col_name] = stock.get(accessor, "N/A")
                                        df_data.append(row)
                                    
                                    df = pd.DataFrame(df_data)
                                    st.dataframe(df)

                                    # Add detailed analysis expandable section
                                    for stock in stocks:
                                        with st.expander(f"📊 Detailed Analysis - {stock['symbol']}"):
                                            cols = st.columns(3)
                                            with cols[0]:
                                                st.metric("Strength Score",
                                                         f"{stock['strength_score']:.0f}/100")
                                            with cols[1]:
                                                st.metric("Current Whale %",
                                                         f"{stock['whale_percentage']:.1f}%")
                                            with cols[2]:
                                                st.metric("Volume",
                                                         f"{stock['volume']:,.0f}")

                                            st.subheader("Trend Analysis")
                                            st.write(f"Current Trend: {stock['trend']}")
                                else:
                                    st.info(f"No stocks found in {category} category")
                    except Exception as e:
                        st.error(f"Error during whale accumulation scan: {str(e)}")

        # MACD Divergence Scanner Tab
        with scanner_tab5:
            if st.button("Scan for MACD Divergences", key="divergence_scanner"):
                with st.spinner("Scanning for MACD divergences..."):
                    try:
                        divergence_results = st.session_state.scanner.scan_macd_divergences(macd_periods=macd_periods)
                        div_tabs = st.tabs(['Bullish Divergence', 'Bearish Divergence'])

                        for tab, category in zip(div_tabs, divergence_results.keys()):
                            with tab:
                                stocks = divergence_results[category]
                                if stocks:
                                    df = pd.DataFrame([{
                                        'Symbol': stock['symbol'],
                                        'Price': f"${stock['price']:.2f}",
                                        'MACD': f"{stock['macd']:.2f}",
                                        'Signal': f"{stock['macd_signal']:.2f}",
                                        'RSI': f"{stock['rsi']:.1f}",
                                        'Volume': f"{stock['volume']:,.0f}",
                                        'Date': stock['date'].strftime('%Y-%m-%d')
                                    } for stock in stocks])
                                    st.dataframe(df)
                                else:
                                    st.info(f"No stocks showing {category.lower()}")
                    except Exception as e:
                        st.error(f"Error during MACD divergence scan: {str(e)}")

        # Quick Insights Section
        st.header("Quick Insights")

        # Add refresh button for insights
        if st.button("Generate Market Insights"):
            try:
                # Generate insights for daily timeframe
                daily_df = data['daily']

                # Market Summary
                st.subheader("Market Summary")
                market_summary = MarketInsights.generate_market_summary(daily_df)
                st.text(market_summary)

                # Trading Signals
                st.subheader("Current Trading Signals")
                trading_signals = MarketInsights.get_trading_signals(daily_df)
                for signal in trading_signals:
                    st.markdown(f"• {signal}")
            except Exception as e:
                st.error(f"Error generating market insights: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()