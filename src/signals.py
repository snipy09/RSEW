"""
Signal generation and evaluation module.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import precision_score, recall_score, f1_score


class SignalGenerator:
    """Generate and evaluate early warning signals."""

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
        self.instability_scores = instability_scores
        self.threshold = threshold
        self.signals = None

    def generate_signals(self, threshold: float = None) -> np.ndarray:
        """
        Generate binary warning signals based on threshold.

        Parameters:
        -----------
        threshold : float, optional
            Override threshold. If None, use self.threshold

        Returns:
        --------
        signals : np.ndarray
            Binary signals (1 = warning, 0 = normal)
        """
        if threshold is None:
            threshold = self.threshold

        # Normalize instability scores to [0, 1] for threshold comparison
        scores_norm = (self.instability_scores - np.min(self.instability_scores)) / (
            np.max(self.instability_scores) -
            np.min(self.instability_scores) + 1e-8
        )

        signals = (scores_norm > threshold).astype(int)
        self.signals = signals

        return signals

    def generate_smoothed_signals(self, window: int = 5, threshold: float = None) -> np.ndarray:
        """
        Generate signals with smoothing to reduce noise.

        Parameters:
        -----------
        window : int, default=5
            Smoothing window
        threshold : float, optional
            Override threshold

        Returns:
        --------
        signals : np.ndarray
            Smoothed binary signals
        """
        if threshold is None:
            threshold = self.threshold

        # Use rolling mean of instability scores
        scores_smooth = pd.Series(self.instability_scores).rolling(
            window=window,
            center=True,
            min_periods=1
        ).mean().values

        # Normalize
        scores_norm = (scores_smooth - np.min(scores_smooth)) / (
            np.max(scores_smooth) - np.min(scores_smooth) + 1e-8
        )

        signals = (scores_norm > threshold).astype(int)
        self.signals = signals

        return signals

    def identify_signal_periods(self) -> np.ndarray:
        """
        Identify continuous periods of warning signals.

        Returns:
        --------
        periods : np.ndarray
            Start and end indices of warning periods
        """
        if self.signals is None:
            raise ValueError("Signals not generated yet")

        # Find transitions
        diff = np.diff(self.signals.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if self.signals[0] == 1:
            starts = np.concatenate([[0], starts])
        if self.signals[-1] == 1:
            ends = np.concatenate([ends, [len(self.signals)]])

        periods = np.column_stack((starts, ends))
        return periods

    def evaluate_signals(
        self,
        regime_transitions: np.ndarray,
        lead_time_threshold: int = 30
    ) -> Dict:
        """
        Evaluate signal quality against detected regime transitions.

        Parameters:
        -----------
        regime_transitions : np.ndarray
            Indices of regime transitions
        lead_time_threshold : int, default=30
            Maximum lead time to consider (days)

        Returns:
        --------
        metrics : Dict
            Evaluation metrics
        """
        if self.signals is None:
            raise ValueError("Signals not generated yet")

        metrics = {}

        # Create ground truth: regime transition occurred within lead_time_threshold
        ground_truth = np.zeros(len(self.signals), dtype=int)
        for transition in regime_transitions:
            if transition < len(ground_truth):
                # Mark period before transition as "regime shift coming"
                start_idx = max(0, transition - lead_time_threshold)
                ground_truth[start_idx:transition] = 1

        # Calculate metrics
        metrics['precision'] = precision_score(
            ground_truth, self.signals, zero_division=0)
        metrics['recall'] = recall_score(
            ground_truth, self.signals, zero_division=0)
        metrics['f1'] = f1_score(ground_truth, self.signals, zero_division=0)

        # Calculate lead time
        signal_indices = np.where(self.signals == 1)[0]
        lead_times = []

        for transition in regime_transitions:
            # Find signals before this transition
            pre_signals = signal_indices[signal_indices < transition]
            if len(pre_signals) > 0:
                last_signal = pre_signals[-1]
                lead_time = transition - last_signal
                if 0 < lead_time <= lead_time_threshold:
                    lead_times.append(lead_time)

        if lead_times:
            metrics['avg_lead_time'] = np.mean(lead_times)
            metrics['median_lead_time'] = np.median(lead_times)
            metrics['min_lead_time'] = np.min(lead_times)
            metrics['max_lead_time'] = np.max(lead_times)
            metrics['n_successful_signals'] = len(lead_times)
        else:
            metrics['avg_lead_time'] = 0
            metrics['median_lead_time'] = 0
            metrics['min_lead_time'] = 0
            metrics['max_lead_time'] = 0
            metrics['n_successful_signals'] = 0

        # False alarm rate
        false_alarms = np.sum((self.signals == 1) & (ground_truth == 0))
        total_signal_days = np.sum(self.signals)
        metrics['false_alarm_rate'] = false_alarms / (total_signal_days + 1e-8)

        return metrics

    def optimize_threshold(
        self,
        regime_transitions: np.ndarray,
        lead_time_threshold: int = 30,
        thresholds: np.ndarray = None
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold based on signal quality.

        Parameters:
        -----------
        regime_transitions : np.ndarray
            Regime transition indices
        lead_time_threshold : int, default=30
            Lead time threshold
        thresholds : np.ndarray, optional
            Thresholds to test. Defaults to [0.1, 0.2, ..., 0.9]

        Returns:
        --------
        optimal_threshold : float
            Best threshold
        results : Dict
            Results for all thresholds
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)

        results = {}

        for threshold in thresholds:
            # Generate signals with this threshold
            self.generate_signals(threshold=threshold)

            # Evaluate
            metrics = self.evaluate_signals(
                regime_transitions, lead_time_threshold)
            results[threshold] = metrics

        # Find optimal (maximize F1 score)
        f1_scores = {t: results[t]['f1'] for t in thresholds}
        optimal_threshold = max(f1_scores, key=f1_scores.get)

        return optimal_threshold, results

    def get_signal_statistics(self) -> Dict:
        """Get statistics about generated signals."""
        if self.signals is None:
            raise ValueError("Signals not generated yet")

        periods = self.identify_signal_periods()
        durations = periods[:, 1] - periods[:, 0]

        stats = {
            'total_signal_days': np.sum(self.signals),
            'signal_percentage': 100 * np.sum(self.signals) / len(self.signals),
            'n_signal_periods': len(periods),
            'avg_signal_duration': np.mean(durations) if len(durations) > 0 else 0,
            'min_signal_duration': np.min(durations) if len(durations) > 0 else 0,
            'max_signal_duration': np.max(durations) if len(durations) > 0 else 0,
        }

        return stats


def generate_early_warning_signals(
    instability_scores: np.ndarray,
    regime_transitions: np.ndarray,
    threshold: float = 0.5,
    optimize: bool = False
) -> Tuple[np.ndarray, SignalGenerator, Dict]:
    """
    Convenience function to generate signals.

    Parameters:
    -----------
    instability_scores : np.ndarray
        Instability scores
    regime_transitions : np.ndarray
        Regime transition indices
    threshold : float, default=0.5
        Signal threshold
    optimize : bool, default=False
        Whether to optimize threshold

    Returns:
    --------
    signals : np.ndarray
        Generated signals
    generator : SignalGenerator
        SignalGenerator object
    metrics : Dict
        Evaluation metrics
    """
    generator = SignalGenerator(instability_scores, threshold=threshold)

    if optimize:
        optimal_threshold, results = generator.optimize_threshold(
            regime_transitions)
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        generator.generate_signals(threshold=optimal_threshold)
    else:
        generator.generate_signals()

    signals = generator.get_signals() if hasattr(
        generator, 'get_signals') else generator.signals
    metrics = generator.evaluate_signals(regime_transitions)

    return signals, generator, metrics
