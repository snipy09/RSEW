"""
Feature engineering module for creating rolling window features.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import sys
import os

# Handle imports
try:
    from utils import calculate_log_returns, calculate_rolling_metric
except ImportError:
    from .utils import calculate_log_returns, calculate_rolling_metric


class FeatureEngineer:
    """Create engineered features for SVD analysis."""

    def __init__(self, rolling_window: int = 60):
        """
        Initialize FeatureEngineer.

        Parameters:
        -----------
        rolling_window : int, default=60
            Rolling window size in days
        """
        self.rolling_window = rolling_window
        self.features_df = None

    @staticmethod
    def calculate_rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling volatility.

        Parameters:
        -----------
        returns : np.ndarray
            Log returns series
        window : int
            Window size

        Returns:
        --------
        volatility : np.ndarray
            Rolling volatility
        """
        vol = calculate_rolling_metric(
            returns,
            window,
            lambda x: np.std(x)
        )
        return vol

    @staticmethod
    def calculate_rolling_correlation(prices1: np.ndarray, prices2: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling correlation between two price series.

        Parameters:
        -----------
        prices1, prices2 : np.ndarray
            Price series
        window : int
            Window size

        Returns:
        --------
        correlation : np.ndarray
            Rolling correlation
        """
        returns1 = calculate_log_returns(prices1)
        returns2 = calculate_log_returns(prices2)

        corr = np.full(len(returns1) - window + 1, np.nan)
        for i in range(len(returns1) - window + 1):
            corr[i] = np.corrcoef(returns1[i:i+window],
                                  returns2[i:i+window])[0, 1]

        return corr

    def create_features(self, prices: np.ndarray, volumes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix from price and volume data.

        Parameters:
        -----------
        prices : np.ndarray
            Close prices (1D array)
        volumes : np.ndarray, optional
            Trading volumes

        Returns:
        --------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        feature_dates : np.ndarray
            Date indices corresponding to features
        """
        # Calculate log returns
        log_returns = calculate_log_returns(prices)

        # Calculate rolling volatilities
        vol_10 = self.calculate_rolling_volatility(log_returns, 10)
        vol_20 = self.calculate_rolling_volatility(log_returns, 20)
        vol_50 = self.calculate_rolling_volatility(log_returns, 50)

        # Adjust for log_returns being 1 element shorter
        n_prices = len(prices)
        n_samples = n_prices - self.rolling_window + 1

        # Initialize feature list (we'll build along the rolling window)
        features_list = []

        for i in range(n_samples):
            # Get window of log returns
            returns_window = log_returns[i:i + self.rolling_window]

            # Calculate features for this window
            feature_vec = []

            # Feature 1: Mean log return
            feature_vec.append(np.mean(returns_window))

            # Feature 2: Volatility of returns in window
            feature_vec.append(np.std(returns_window))

            # Feature 3: Skewness
            mean_ret = np.mean(returns_window)
            feature_vec.append(np.mean(
                (returns_window - mean_ret) ** 3) / (np.std(returns_window) ** 3 + 1e-8))

            # Feature 4: Kurtosis
            feature_vec.append(np.mean(
                (returns_window - mean_ret) ** 4) / (np.std(returns_window) ** 4 + 1e-8))

            # Feature 5: Autocorrelation lag-1
            if len(returns_window) > 1:
                acf_1 = np.corrcoef(
                    returns_window[:-1], returns_window[1:])[0, 1]
                feature_vec.append(np.nan_to_num(acf_1))
            else:
                feature_vec.append(0.0)

            # Feature 6: Maximum drawdown in window
            cum_returns = np.cumprod(1 + returns_window)
            max_dd = np.min(cum_returns) / np.max(cum_returns) - \
                1 if len(cum_returns) > 0 else 0
            feature_vec.append(max_dd)

            # Feature 7-9: Volatility ratios at different lookback periods
            if i + 10 <= n_prices - self.rolling_window:
                idx = i + self.rolling_window
                if idx < len(vol_10):
                    feature_vec.append(
                        vol_10[idx] if not np.isnan(vol_10[idx]) else 0)
                else:
                    feature_vec.append(0)
            else:
                feature_vec.append(0)

            if i + 20 <= n_prices - self.rolling_window:
                idx = i + self.rolling_window
                if idx < len(vol_20):
                    feature_vec.append(
                        vol_20[idx] if not np.isnan(vol_20[idx]) else 0)
                else:
                    feature_vec.append(0)
            else:
                feature_vec.append(0)

            if i + 50 <= n_prices - self.rolling_window:
                idx = i + self.rolling_window
                if idx < len(vol_50):
                    feature_vec.append(
                        vol_50[idx] if not np.isnan(vol_50[idx]) else 0)
                else:
                    feature_vec.append(0)
            else:
                feature_vec.append(0)

            # Feature 10: Volume change if available
            if volumes is not None and i + self.rolling_window < len(volumes):
                vol_change = (volumes[i + self.rolling_window] - np.mean(
                    volumes[i:i + self.rolling_window])) / (np.mean(volumes[i:i + self.rolling_window]) + 1e-8)
                feature_vec.append(vol_change)
            else:
                feature_vec.append(0)

            features_list.append(feature_vec)

        features = np.array(features_list)

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature dates are the last date in each window
        feature_dates = np.arange(self.rolling_window - 1, len(prices))

        return features, feature_dates

    def create_simple_features(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create simplified feature set (fewer features for faster computation).

        Parameters:
        -----------
        prices : np.ndarray
            Close prices

        Returns:
        --------
        features : np.ndarray
            Simplified feature matrix
        feature_dates : np.ndarray
            Date indices
        """
        log_returns = calculate_log_returns(prices)
        n_prices = len(prices)
        n_samples = n_prices - self.rolling_window + 1

        features_list = []

        for i in range(n_samples):
            returns_window = log_returns[i:i + self.rolling_window]

            feature_vec = [
                np.mean(returns_window),      # Mean return
                np.std(returns_window),       # Volatility
                np.std(returns_window) /
                (np.abs(np.mean(returns_window)) + 1e-8),  # Sharpe-like
                np.max(returns_window) - np.min(returns_window),  # Range
            ]

            features_list.append(feature_vec)

        features = np.array(features_list)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        feature_dates = np.arange(self.rolling_window - 1, len(prices))

        return features, feature_dates


def engineer_features(
    prices: np.ndarray,
    volumes: np.ndarray = None,
    rolling_window: int = 60,
    simple: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to engineer features.

    Parameters:
    -----------
    prices : np.ndarray
        Price series
    volumes : np.ndarray, optional
        Volume series
    rolling_window : int, default=60
        Rolling window size
    simple : bool, default=False
        Use simplified feature set

    Returns:
    --------
    features : np.ndarray
        Feature matrix
    feature_dates : np.ndarray
        Feature date indices
    """
    engineer = FeatureEngineer(rolling_window=rolling_window)

    if simple:
        features, feature_dates = engineer.create_simple_features(prices)
    else:
        features, feature_dates = engineer.create_features(prices, volumes)

    return features, feature_dates
