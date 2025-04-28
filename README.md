# MarketSense: Advanced Stock Market Intelligence Platform

## Project Overview

MarketSense is a sophisticated market intelligence platform designed to provide traders and investors with deep insights into stock market behavior, with a particular focus on institutional (whale) activity detection. The platform leverages multiple data sources and advanced technical analysis to help users identify high-probability trading opportunities across different timeframes.

## Key Features

### Multi-Timeframe Analysis
- **Seamless Switching**: Analyze daily, weekly, and monthly charts in a tabbed interface to identify trend alignment across timeframes
- **Combined Signal Strength**: Proprietary algorithm that identifies when signals align across multiple timeframes, giving higher conviction trade setups

### Advanced Technical Indicators
- **Custom MACD Configuration**: Adjustable MACD parameters with visual crossover signals and divergence detection
- **Multi-RSI Overlay**: View 14, 21, and 50-period RSI simultaneously to identify convergence/divergence patterns
- **Volume Analysis**: Identifies unusual volume patterns with statistical Z-score measurements

### Institutional Activity Detection ("Whale Tracking")
- **Smart Money Indicator**: Proprietary algorithm that combines volume analysis, price impact, and institutional ownership data
- **Real-Time Institutional Data**: Integration with Fintel.io for institutional ownership metrics and fund sentiment scores
- **Insider Activity Analysis**: Tracks significant insider buying and selling patterns

### Market Scanner
- **Signal Scanner**: Scans the market for stocks matching your custom signal criteria
- **RSI Scanner**: Identifies oversold and overbought stocks sorted by technical strength
- **MACD Scanner**: Finds stocks with recent MACD crossovers and momentum shifts
- **Whale Accumulation Scanner**: Detects unusual institutional activity across all stocks
- **MACD Divergence Scanner**: Locates potential reversal candidates using MACD divergence analysis

### Trading Signal Generation
- **Buy/Sell Signals**: Clearly marked entry and exit points based on technical and institutional factors
- **Warning Level System**: Color-coded warning levels to anticipate upcoming signal changes
- **No-Sell Zones**: Identification of periods where selling is not advised due to institutional accumulation

### Interactive Data Visualization
- **Custom Chart Design**: Clean, information-rich charts with tooltips showing detailed metrics
- **Signal Annotation**: Visual markers for buy/sell signals and whale activity directly on charts
- **Technical Overlay Options**: Customizable indicator display options

## Technical Implementation

### Data Pipeline
- **Polygon.io API Integration**: Real-time and historical price data with intelligent caching and rate-limiting
- **Fintel.io Data Retrieval**: Authentication system for accessing premium institutional ownership data
- **Multi-Source Fallback Strategy**: Graceful degradation to technical-only analysis when institutional data is unavailable

### Authentication System
- **Cookie-Based Authentication**: Secure session handling for Fintel.io data access
- **Cloudflare Protection Support**: Complete implementation of Cloudflare bypass for reliable data retrieval
- **Status Verification**: Real-time verification of authentication status with clear user feedback

### Performance Optimization
- **Intelligent Caching**: Local storage of recently accessed data to reduce API usage
- **Rate Limit Handling**: Built-in exponential backoff retry mechanism for API rate limit management
- **Asynchronous Processing**: Background processing of scanner tasks to maintain UI responsiveness

## Use Cases

- **Technical Traders**: Identify high-probability setups with multi-timeframe confirmation
- **Institutional Followers**: Track "smart money" movements to ride institutional momentum
- **Swing Traders**: Find optimal entry and exit points with warning level anticipation
- **Fundamental Investors**: Correlate institutional activity with company fundamentals
- **Algorithmic Traders**: Export signals for integration with automated trading systems

## Getting Started

### Prerequisites
- Python 3.9+ 
- Polygon.io API key for price data
- Optional: Fintel.io account for institutional data access (enhances whale detection)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market-sense.git
cd market-sense
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

4. Enter your Polygon.io API key in the sidebar settings

5. (Optional) For institutional data, add your Fintel.io cookies in the Authentication section

## Future Development

- Portfolio tracking and performance measurement
- Machine learning signal optimization
- Real-time alerts for signal triggers
- Expanded fundamental data integration
- Custom scanner rule creation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by [Polygon.io](https://polygon.io) and [Fintel.io](https://fintel.io)
- Built with [Streamlit](https://streamlit.io/)

---

*MarketSense brings institutional-grade market intelligence to individual traders and investors, helping level the playing field against large financial institutions by exposing their activity patterns and providing advanced technical analysis tools across multiple timeframes.*