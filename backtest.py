import numpy as np
import pandas as pd
from models import RegimeKMeans, RegimeGMM, RegimeHMM, RegimeCUSUM
import matplotlib.pyplot as plt

class BaselineStrategy:
    """
    Implements a simple Moving Average Crossover strategy as a control group.
    Buys SPY when the fast MA crosses above the slow MA.
    Moves to cash when the feat MA crosses below the slow MA.
    """
    def __init__(self, fast_window: int = 50, slow_window: int = 200):
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def run(self, raw_prices: pd.Series) -> pd.DataFrame:
        """
        Executes the vectorized backtest on the provided price series.
        """
        df = raw_prices.to_frame(name='Price')
        # log returns
        df['Market_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
        # moving averages
        df['Fast_MA'] = df['Price'].rolling(window=self.fast_window).mean()
        df['Slow_MA'] = df['Price'].rolling(window=self.slow_window).mean()
        # 1 = Long, 0 = Cash
        df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
        # trade tomorrow based on today's closing signal
        df['Position'] = df['Signal'].shift(1)
        df['Strategy_Returns'] = df['Position'] * df['Market_Returns']
        
        return df.dropna()


class StateDrivenStrategy:
    """
    Overlays a continuous-state ML regime filter (K-Means, GMM, HMM).
    """
    def __init__(self, engine, fast_window: int = 50, slow_window: int = 200):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.engine = engine
    
    def run(self, raw_prices: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        df = raw_prices.to_frame(name='Price')
        df['Market_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
        df['Fast_MA'] = df['Price'].rolling(window=self.fast_window).mean()
        df['Slow_MA'] = df['Price'].rolling(window=self.slow_window).mean()
        df['MA_Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
        # fit engine and dynmaically id high-volatility crash regime
        self.engine.fit(features)
        regime_preds = self.engine.predict(features)
        df['Regime'] = regime_preds
        regime_vix_variance = features.groupby(regime_preds)['^VIX'].var()
        crash_regime = regime_vix_variance.idxmax()
        # trade only if MA says buy and we're not in the crash regime
        df['ML_Signal'] = np.where(
            (df['MA_Signal'] == 1) & (df['Regime'] != crash_regime), 1, 0
        )
        df['Position'] = df['ML_Signal'].shift(1)
        df['Strategy_Returns'] = df['Position'] * df['Market_Returns']

        return df.dropna()


class EventDrivenStrategy:
    """
    Overlays a sparse event-driven filter (CUSUM).
    Uses a time-decay memory latch to stay in cash after a structural break.
    """
    def __init__(self, engine, fast_window: int = 50, slow_window: int = 200, cooldown: int = 21):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.engine = engine
        self.cooldown = cooldown # trading days to stay in cash after a break
        
    def run(self, raw_prices: pd.Series, vix_series: pd.Series) -> pd.DataFrame:
        df = raw_prices.to_frame(name='Price')
        df['Market_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
        df['Fast_MA'] = df['Price'].rolling(window=self.fast_window).mean()
        df['Slow_MA'] = df['Price'].rolling(window=self.slow_window).mean()
        df['MA_Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
        # get sparse events (+1, -1, 0)
        events = self.engine.fit_predict(vix_series)
        df['Event'] = events.abs() # treat upward and downward volatility breaks equally
        # if an event fired in the last 21 days, In_Cooldown = 1
        df['In_Cooldown'] = df['Event'].rolling(window=self.cooldown, min_periods=1).max()
        # trade only if MA says buy AND the CUSUM cooldown is 0
        df['ML_Signal'] = np.where(
            (df['MA_Signal'] == 1) & (df['In_Cooldown'] == 0), 1, 0
        )
        df['Position'] = df['ML_Signal'].shift(1)
        df['Strategy_Returns'] = df['Position'] * df['Market_Returns']
        
        return df.dropna()

def calculate_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> tuple:
    """
    Calculates annualized Sharpe Ratio and Maximum Drawdown.
    Assuming 252 trading days in a year for daily data.
    """
    # Annualized Sharpe Ratio
    excess_returns = returns_series - (risk_free_rate / 252)
    sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    # Maximum Drawdown
    cumulative_returns = (1 + returns_series).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return sharpe, max_drawdown



if __name__ == "__main__":
    # 1. load data
    raw_data = pd.read_csv("data/raw_macro_data.csv", index_col=0, parse_dates=True)
    stationary_features = pd.read_csv("data/stationary_features.csv", index_col=0, parse_dates=True)
    # align dates
    common_index = raw_data.index.intersection(stationary_features.index)
    spy_prices = raw_data.loc[common_index, 'SPY']
    stationary_features = stationary_features.loc[common_index]
    vix_series = stationary_features['^VIX']

    # 2. initialize Models
    kmeans_engine = RegimeKMeans(n_regimes=2)
    gmm_engine = RegimeGMM(n_regimes=2)
    hmm_engine = RegimeHMM(n_regimes=2)
    cusum_engine = RegimeCUSUM(threshold_multi=10.0)

    # 3. run Strategies
    base_res = BaselineStrategy().run(spy_prices)
    kmeans_res = StateDrivenStrategy(kmeans_engine).run(spy_prices, stationary_features)
    gmm_res = StateDrivenStrategy(gmm_engine).run(spy_prices, stationary_features)
    hmm_res = StateDrivenStrategy(hmm_engine).run(spy_prices, stationary_features)
    cusum_res = EventDrivenStrategy(cusum_engine).run(spy_prices, vix_series)\
    
    # 4. calculate and Print Metrics
    metrics_map = {
        "Buy & Hold SPY": base_res['Market_Returns'],
        "Baseline MA Cross": base_res['Strategy_Returns'],
        "K-Means Overlay": kmeans_res['Strategy_Returns'],
        "GMM Overlay": gmm_res['Strategy_Returns'],
        "HMM Overlay": hmm_res['Strategy_Returns'],
        "CUSUM Overlay": cusum_res['Strategy_Returns']
    }
    print("\n--- Backtest Risk & Return Metrics ---")
    print(f"{'Strategy':<20} | {'Final Wealth':<15} | {'Sharpe':<10} | {'Max Drawdown'}")
    print("-" * 65)
    for name, returns in metrics_map.items():
        # terminal wealth from $1
        wealth = np.exp(returns.cumsum()).iloc[-1]
        # risk metrics
        sharpe, max_dd = calculate_metrics(returns)
        # format the output table
        print(f"{name:<20} | ${wealth:<14.2f} | {sharpe:<10.2f} | {max_dd*100:<10.2f}%")

    # 5. generate the final portfolio visualization
    print("\nGenerating structural break visualization...")
    plt.figure(figsize=(16, 8))
    plt.plot(spy_prices.index, spy_prices.values, label='SPY Price', color='black', linewidth=1.2)
    # iterate through the recorded structural breaks and plot them
    for date, event_type in cusum_engine.breaks:
        if event_type == 'Upward Break':
            plt.axvline(x=date, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
        elif event_type == 'Downward Break':
            plt.axvline(x=date, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    # titles, labels, and legends
    plt.title('Symmetric CUSUM Filter: Macroeconomic Structural Breaks (10.0x Threshold)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('SPY Price (USD)', fontsize=12)
    # custom legend handles to avoid duplicate labels
    import matplotlib.lines as mlines
    green_line = mlines.Line2D([], [], color='green', linestyle='--', label='Upward Volatility Break')
    red_line = mlines.Line2D([], [], color='red', linestyle='--', label='Downward Volatility Break')
    black_line = mlines.Line2D([], [], color='black', label='SPY Asset Price')
    plt.legend(handles=[black_line, green_line, red_line], loc='upper left', fontsize=11)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cusum_structural_breaks.png', dpi=300, bbox_inches='tight')
    plt.show()