"""
Utility functions for the Regime Shift Early Warning System.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Union


def standardize_features(data: np.ndarray, method: str = 'zscore') -> Tuple[np.ndarray, Union[StandardScaler, MinMaxScaler]]:
    """
    Standardize features using z-score or min-max normalization.

    Parameters:
    -----------
    data : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    method : str, default='zscore'
        Standardization method: 'zscore' or 'minmax'

    Returns:
    --------
    standardized_data : np.ndarray
        Standardized features
    scaler : object
        Fitted scaler for inverse transformation
    """
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    standardized = scaler.fit_transform(data)
    return standardized, scaler


def normalize_scores(scores: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize a 1D array of scores.

    Parameters:
    -----------
    scores : np.ndarray
        1D array of scores to normalize
    method : str, default='zscore'
        Normalization method: 'zscore' or 'minmax'

    Returns:
    --------
    normalized_scores : np.ndarray
        Normalized scores
    """
    if method == 'zscore':
        mean = np.nanmean(scores)
        std = np.nanstd(scores)
        return (scores - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.nanmin(scores)
        max_val = np.nanmax(scores)
        return (scores - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")


def exponential_weights(window_size: int, decay: float = 0.5) -> np.ndarray:
    """
    Generate exponential weights for recency bias.

    Parameters:
    -----------
    window_size : int
        Number of weights to generate
    decay : float, default=0.5
        Decay rate (0-1). Higher = more recent data weighted higher

    Returns:
    --------
    weights : np.ndarray
        Exponential weights normalized to sum to 1
    """
    weights = np.exp(np.arange(window_size) * decay / window_size)
    return weights / np.sum(weights)


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate log returns from price series.

    Parameters:
    -----------
    prices : np.ndarray
        Price series

    Returns:
    --------
    log_returns : np.ndarray
        Log returns (1D array, length = len(prices) - 1)
    """
    return np.diff(np.log(prices))


def calculate_rolling_metric(data: np.ndarray, window: int, metric_func, *args, **kwargs) -> np.ndarray:
    """
    Calculate rolling metric over time series.

    Parameters:
    -----------
    data : np.ndarray
        Time series data
    window : int
        Rolling window size
    metric_func : callable
        Function to compute metric on each window
    *args, **kwargs
        Additional arguments for metric_func

    Returns:
    --------
    metrics : np.ndarray
        Rolling metrics (length = len(data) - window + 1)
    """
    n_samples = len(data)
    metrics = np.full(n_samples - window + 1, np.nan)

    for i in range(n_samples - window + 1):
        window_data = data[i:i + window]
        metrics[i] = metric_func(window_data, *args, **kwargs)

    return metrics


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Safe division that avoids division by zero.

    Parameters:
    -----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    fill_value : float, default=0.0
        Value to use when denominator is zero

    Returns:
    --------
    result : np.ndarray
        Result of division with safe handling
    """
    result = np.divide(numerator, denominator,
                       out=np.full_like(numerator, fill_value, dtype=float),
                       where=denominator != 0)
    return result


def print_section(title: str, char: str = '=', width: int = 60):
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")
