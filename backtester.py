import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
    def run_backtest(self, df, signals):
        """Run basic backtest on the signals"""
        positions = pd.DataFrame(index=signals.index)
        positions['Position'] = signals['Signal']
        
        # Calculate returns
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['Returns'] = df['Close'].pct_change()
        portfolio['Strategy_Returns'] = portfolio['Returns'] * positions['Position'].shift(1)
        
        # Calculate cumulative returns
        portfolio['Cumulative_Market_Returns'] = (1 + portfolio['Returns']).cumprod()
        portfolio['Cumulative_Strategy_Returns'] = (1 + portfolio['Strategy_Returns']).cumprod()
        
        # Calculate metrics
        total_returns = portfolio['Cumulative_Strategy_Returns'].iloc[-1] - 1
        market_returns = portfolio['Cumulative_Market_Returns'].iloc[-1] - 1
        
        sharpe_ratio = np.sqrt(252) * (portfolio['Strategy_Returns'].mean() / 
                                      portfolio['Strategy_Returns'].std())
        
        max_drawdown = (portfolio['Cumulative_Strategy_Returns'] / 
                       portfolio['Cumulative_Strategy_Returns'].cummax() - 1).min()
        
        results = {
            'Total Returns': f"{total_returns:.2%}",
            'Market Returns': f"{market_returns:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Portfolio Value': f"${self.initial_capital * (1 + total_returns):,.2f}"
        }
        
        return results, portfolio
