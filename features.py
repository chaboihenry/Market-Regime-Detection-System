import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def get_weights_ffd(d: float, threshold: float = 1e-4) -> np.ndarray:
    """
    Generates fractional differentiation weights.
    Stops calculation when absolute weight drops below the threshold
    """
    w = [1.0]
    k = 1
    while True:
        # iterative weight formula (LPD)
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < threshold:
            break
        w.append(w_)
        k += 1
    
    # Notice the weights are in reverse order, so most recent data gets largest weight
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """
    Applies the fixed-width fractional weights to a time series
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    # storage of the stationary values
    diff_series = {}
    # roll through the series anc calculate the dot product
    for i in range(width -1, len(series)):
        window = series.iloc[i - width + 1: i + 1]
        diff_series[series.index[i]] = np.dot(weights.T, window)[0]

    return pd.Series(diff_series)

def find_min_d(series: pd.Series, max_d: float = 1.0, step: float = 0.05) -> float:
    """
    Finds the minimum differentiation value 'd' that makes the series stationary
    using the Augmented Dickey-Fuller test.
    """
    for d in np.arange(0.0, max_d + step, step):
        diff_series = frac_diff_ffd(series, d)
        if len(diff_series) < 100: # Ensure we have enough data points for ADF test
            break
            # run ADF test (adfuller returns a tuple; index 1 is the p-value)
        p_value = adfuller(diff_series, maxlag=1, regression='c', autolag=None)[1]
        print(f"Testing d={d:.2f} | p-value={p_value:4f}")
        if p_value < 0.05: # Stationarity threshold
            print(f"Series is stationary at d={d:.2f} with p-value={p_value:.4f}")
            return d
    print(f"Series is not stationary up to d={max_d:.2f}")
    return max_d

if __name__ == "__main__":
    # load macroeconomic data
    raw_data = pd.read_csv('data/raw_macro_data.csv', index_col=0, parse_dates=True)
    stationary_features = {}
    optimal_d_values = {}

    for col in raw_data.columns:
        print(f"---- Optimizing {col} ----")
        # log-transform the series to stabilize variance
        log_series = np.log(raw_data[col])
        # calculate optimal d
        d_val = find_min_d(log_series)
        optimal_d_values[col] = d_val
        # generate fully transformed series using optimal d
        transformed_series = frac_diff_ffd(log_series, d_val)
        stationary_features[col] = transformed_series

    # drop rows where the FFD window produced NaNs, then save
    features_df = pd.DataFrame(stationary_features)
    features_df.dropna(inplace=True)
    features_df.to_csv("data/stationary_features.csv")

    print("Final Optimal d-values")
    print(optimal_d_values)
    print(f"\nStationary feature matrix saved. Shape: {features_df.shape}")