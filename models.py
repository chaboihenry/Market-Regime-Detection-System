import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

class RegimeKMeans:
    """
    Distance-based regime detection using K-Means. 
    Serves as baseline to demostrate the flaw of spherecial clustering.
    """
    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame):
        """
        Fits the K-Means algorithm.
        """
        self.model.fit(features)
        self.is_fitted = True
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """ 
        Returns distance-based regime assignments.
        """ 
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        return pd.Series(self.model.predict(features), index=features.index, name='Regime_KMeans')


class RegimeGMM:
    """
    Density-based regime detection using Gaussian Mixture Models.
    Utilizes full covariance matrices for elliptical financial data.
    """
    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            init_params='kmeans',
            max_iter=100,
            random_state=random_state
        )
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame):
        """
        Fits the Expectation Maximization (EM) Algorithm.
        """
        self.model.fit(features)
        self.is_fitted = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Returns density-based regime assignments.
        """
        if not self.is_fitted:
            raise ValueError("Model must be Fitted")
        return pd.Series(self.model.predict(features), index=features.index, name='Regime_GMM')


class RegimeHMM:
    """
    Temporal regime detection using Hidden Markov Models.
    Applies the Baum-Welch algorithm to estimate state transitions.
    """
    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=random_state
        )
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame):
        """
        Fits the Baum-Welch algorithm and extracts the transition matrix.
        """
        #hmmlearn requires numpy array format
        self.model.fit(features.values)
        self.is_fitted = True
        print("\n--- HMM Estimated Transition Matrix ---")
        print(np.round(self.model.transmat_, 4))
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Returns the most likely sequence of hidden states.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted.")
        return pd.Series(self.model.predict(features.values), index=features.index, name='Regime_HMM')


class RegimeCUSUM:
    """
    Symmetric CUSUM Filter derived from Advances in Financial Machine Learning.
    Detects structural breaks using accumulated absolute price differences.
    """
    def __init__(self, threshold_multi: float = 10.0):
        # threshold defines 'how many standard deviations the cumulative sum
        # must drift before triggering a regime break'. Default to 10.0, empirically
        # derived from the optimization loop
        self.threshold_multi = threshold_multi
        self.breaks = []

    def fit_predict(self, series: pd.Series) -> pd.Series:
        """
        Runs the CUSUM filter over a single 1D time series (like VIX or SPY vol).
        Calculates an absolute threshold based on the standard deviation of daily differences. Accumulates
        consecutive deviations and triggers a structural break event (1 or -1) when threshold is breached.
        """
        # caluculate absolute threshold 'h' based on series' daily differences
        diffs = series.diff().dropna()
        h = diffs.std() * self.threshold_multi
        print(f"Calculated CUSUM absolute threshold (h): {h:.4f}")
        
        s_pos = 0.0
        s_neg = 0.0
        events = pd.Series(0, index=series.index, name='CUSUM_Breaks')
        
        for i in range(1, len(series)):
            diff_val = series.iloc[i] - series.iloc[i-1]
            # accumulate the raw differences
            s_pos = max(0, s_pos + diff_val)
            s_neg = min(0, s_neg + diff_val)
            # trigger events if 'h' is breached
            if s_pos > h:
                s_pos, s_neg = 0.0, 0.0 # reset
                events.iloc[i] = 1
                self.breaks.append((series.index[i], 'Upward Break'))
            elif s_neg < -h:
                s_pos, s_neg = 0.0, 0.0 # reset
                events.iloc[i] = -1
                self.breaks.append((series.index[i], 'Downward Break'))
        
        print(f"Total Structural Breaks Detected: {len(self.breaks)}")
        return events


def optimize_cusum_threshold(series: pd.Series, max_mult: float = 10.0, step: float = 0.5):
        """
        Iterates through threshold multipliers to observe the decay of structural breaks. 
        This empirically measures the fat tails of the asset's volatility.
        """
        print(f"--- Optimizing CUSUM Threshold for {series.name} ---")
        # calculate the base 1-sigma volatility of daily differences
        diffs = series.diff().dropna()
        base_std = diffs.std()
        print(f"Base Daily Difference Volatility (1-Sigma): {base_std:.4f}\n")

        for mult in np.arange(1.0, max_mult + step, step):
            h = base_std * mult
            s_pos, s_neg = 0.0, 0.0
            break_count = 0

            for i in range(1, len(series)):
                diff_val = series.iloc[i] - series.iloc[i-1]
                s_pos = max(0, s_pos + diff_val)
                s_neg = min(0, s_neg + diff_val)
                # reset both memory sums when a break occurs
                if s_pos > h:
                    s_pos, s_neg = 0.0, 0.0
                    break_count += 1
                elif s_neg < -h:
                    s_pos, s_neg= 0.0, 0.0
                    break_count += 1
            
            print(f"Multiplier: {mult:4.1f}x | Threshold (h): {h:.4f} | Total Breaks: {break_count}")


if __name__ == "__main__":
    # Load stationary features
    data = pd.read_csv("data/stationary_features.csv", index_col=0, parse_dates=True)
    # # Initialize the 3 continuous-state engines
    # kmeans_engine = RegimeKMeans(n_regimes=2)
    # gmm_engine = RegimeGMM(n_regimes=2)
    # hmm_engine = RegimeHMM(n_regimes=2)
    # # fit the models
    # kmeans_engine.fit(data)
    # gmm_engine.fit(data)
    # hmm_engine.fit(data)
    # # generate predictions
    # kmeans_preds = kmeans_engine.predict(data)
    # gmm_preds = gmm_engine.predict(data)
    # hmm_preds = hmm_engine.predict(data)
    # # evaluate state distributions
    # print("\n--- State Distributions ---")
    # print(f"K-Means:\n{kmeans_preds.value_counts().to_dict()}")
    # print(f"GMM:\n:{gmm_preds.value_counts().to_dict()}")
    # print(f"HMM:\n{hmm_preds.value_counts().to_dict()}")
    # # check agreement rate between GMM & HMM
    # agreement = (gmm_preds == hmm_preds).mean() * 100
    # print(f"\nGMM vs HMM agreement rate: {agreement:.2f}%")

    # # run optimization on the VIX to find true macro fracture point
    # optimize_cusum_threshold(data['^VIX'])
    vix_series = data['^VIX']
    cusum_engine = RegimeCUSUM(threshold_multi=10.0)
    cusum_events = cusum_engine.fit_predict(vix_series)
    print(f"\n--- Structural Breaks at 10.0x Multiplier ---")
    for date, event_type in cusum_engine.breaks:
        print(f"Date: {date.date()} | Event: {event_type}")
