import pandas as pd
from typing import Dict, List

class MarketInsights:
    @staticmethod
    def analyze_whale_activity(df: pd.DataFrame) -> Dict:
        """Analyze whale activity patterns"""
        recent_whale_pct = df['Whale_Volume_Pct'].iloc[-1]
        whale_trend = df['Whale_Volume_Pct'].tail(5).mean() - df['Whale_Volume_Pct'].tail(10).mean()
        
        return {
            'current_whale_percentage': recent_whale_pct,
            'whale_trend': whale_trend
        }
    
    @staticmethod
    def analyze_technical_indicators(df: pd.DataFrame) -> Dict:
        """Analyze technical indicators for market insights"""
        current_rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        recent_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume_SMA'].iloc[-1]
        
        return {
            'rsi_level': current_rsi,
            'macd_status': 'bullish' if macd > macd_signal else 'bearish',
            'volume_vs_average': (recent_volume / avg_volume - 1) * 100
        }
    
    @staticmethod
    def generate_market_summary(df: pd.DataFrame) -> str:
        """Generate a comprehensive market summary"""
        whale_data = MarketInsights.analyze_whale_activity(df)
        tech_data = MarketInsights.analyze_technical_indicators(df)
        
        # Price movement analysis
        price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100)
        price_trend = "upward" if price_change > 0 else "downward"
        
        # Generate insights
        insights = [
            f"Current Price Movement: {price_trend} ({price_change:.2f}%)",
            f"Whale Activity: {whale_data['current_whale_percentage']:.1f}% of volume",
            f"Whale Trend: {'Increasing' if whale_data['whale_trend'] > 0 else 'Decreasing'} whale presence",
            f"Technical Indicators:",
            f"  - RSI: {'Overbought' if tech_data['rsi_level'] > 70 else 'Oversold' if tech_data['rsi_level'] < 30 else 'Neutral'} at {tech_data['rsi_level']:.1f}",
            f"  - MACD: {tech_data['macd_status'].title()} momentum",
            f"  - Volume: {tech_data['volume_vs_average']:.1f}% compared to average"
        ]
        
        return "\n".join(insights)

    @staticmethod
    def get_trading_signals(df: pd.DataFrame) -> List[str]:
        """Generate current trading signals based on indicators"""
        signals = []
        
        # RSI signals
        current_rsi = df['RSI'].iloc[-1]
        if current_rsi < 30:
            signals.append("RSI indicates oversold conditions - potential buying opportunity")
        elif current_rsi > 70:
            signals.append("RSI indicates overbought conditions - consider taking profits")
            
        # MACD signals
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_Signal'].iloc[-1]
        if current_macd > current_signal and current_macd > 0:
            signals.append("MACD shows strong bullish momentum")
        elif current_macd < current_signal and current_macd < 0:
            signals.append("MACD indicates bearish pressure")
            
        # Volume analysis
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume_SMA'].iloc[-1]
        if current_volume > avg_volume * 1.5:
            signals.append("Unusually high volume - increased market interest")
        
        # Whale activity
        if df['Whale_Volume_Pct'].iloc[-1] > 50:
            signals.append("Significant whale accumulation detected")
            
        return signals if signals else ["No significant signals at the moment"]
