"""
Signal generation and evaluation module for Regime Shift Early Warning System.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


class SignalGenerator:
    """Generate, filter, and evaluate early warning signals from instability metrics."""

    def __init__(self, instability_scores: np.ndarray, threshold: float = 0.5):
        """
        Initialize SignalGenerator.

        Parameters:
        -----------
        instability_scores : np.ndarray
            Instability scores over time
        threshold : float, default=0.5
            Threshold for signal generation
        """
        self.instability_scores = np.asarray(instability_scores, dtype=float)
        self.threshold = threshold
        self.signals = None

    def generate_signals(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Generate binary warning signals based on normalized threshold.
        """
        if threshold is None:
            threshold = self.threshold

        min_val = np.nanmin(self.instability_scores)
        max_val = np.nanmax(self.instability_scores)
        denom = (max_val - min_val) if (max_val - min_val) > 1e-8 else 1.0
        scores_norm = np.clip((self.instability_scores - min_val) / denom, 0.0, 1.0)

        signals = (scores_norm > threshold).astype(int)
        self.signals = signals
        return signals

    def generate_smoothed_signals(self, window: int = 5, threshold: Optional[float] = None) -> np.ndarray:
        """
        Generate signals with rolling smoothing to reduce noise and false positives.
        """
        if threshold is None:
            threshold = self.threshold

        series = pd.Series(self.instability_scores).fillna(method='ffill').fillna(0)
        scores_smooth = series.rolling(window=window, min_periods=1).mean().values

        min_val = np.nanmin(scores_smooth)
        max_val = np.nanmax(scores_smooth)
        denom = (max_val - min_val) if (max_val - min_val) > 1e-8 else 1.0
        scores_norm = np.clip((scores_smooth - min_val) / denom, 0.0, 1.0)

        signals = (scores_norm > threshold).astype(int)
        self.signals = signals
        return signals

    def generate_adaptive_signals(self, lookback: int = 40, num_std: float = 1.25) -> np.ndarray:
        """
        Generate signals using an adaptive rolling quantile & volatility envelope.
        Triggers when instability exceeds rolling mean + num_std * rolling std.
        """
        series = pd.Series(self.instability_scores).fillna(method='ffill').fillna(0)
        rolling_mean = series.rolling(window=lookback, min_periods=10).mean()
        rolling_std = series.rolling(window=lookback, min_periods=10).std().fillna(1e-4)

        adaptive_threshold = rolling_mean + (num_std * rolling_std)
        signals = (series > adaptive_threshold).astype(int).values
        self.signals = signals
        return signals

    @staticmethod
    def calculate_drawdowns(cumulative_returns: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate drawdown series and maximum drawdown from cumulative returns series.
        """
        wealth_index = 1.0 + np.asarray(cumulative_returns, dtype=float)
        peak = np.maximum.accumulate(wealth_index)
        drawdown = (wealth_index - peak) / (peak + 1e-9)
        max_dd = float(np.min(drawdown))
        return drawdown, max_dd

    @staticmethod
    def calculate_performance_metrics(returns: np.ndarray, risk_free_rate: float = 0.045) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted quantitative metrics.
        """
        ret = np.asarray(returns, dtype=float)
        ret = ret[~np.isnan(ret)]
        if len(ret) < 2:
            return {
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "var_95": 0.0
            }

        daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
        excess_returns = ret - daily_rf

        ann_return = float(np.mean(ret) * 252.0)
        ann_vol = float(np.std(ret, ddof=1) * np.sqrt(252.0)) + 1e-8

        sharpe = float((ann_return - risk_free_rate) / ann_vol)

        downside = ret[ret < 0]
        downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252.0)) if len(downside) > 1 else ann_vol
        sortino = float((ann_return - risk_free_rate) / (downside_vol + 1e-8))

        cum_ret = np.cumprod(1.0 + ret) - 1.0
        _, max_dd = SignalGenerator.calculate_drawdowns(cum_ret)
        calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0

        var_95 = float(np.percentile(ret, 5))

        return {
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 2),
            "var_95": round(var_95, 4)
        }
