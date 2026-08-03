"""
Regime detection module using Hidden Markov Models.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None


class RegimeDetector:
    """Detect market regimes using HMM or KMeans."""

    def __init__(self, n_regimes: int = 3, method: str = 'hmm'):
        """
        Initialize RegimeDetector.

        Parameters:
        -----------
        n_regimes : int, default=3
            Number of regimes
        method : str, default='hmm'
            Method: 'hmm' or 'kmeans'
        """
        self.n_regimes = n_regimes
        self.method = method

        if method == 'hmm':
            if hmm is None:
                print("hmmlearn not found. Falling back to KMeans.")
                self.method = 'kmeans'
            else:
                self.model = hmm.GaussianHMM(
                    n_components=n_regimes, random_state=42)
        elif method == 'kmeans':
            self.model = KMeans(n_clusters=n_regimes, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.regimes = None
        self.regime_probs = None

    def fit_hmm(self, features: np.ndarray) -> np.ndarray:
        """
        Fit HMM to features and return regime labels.

        Parameters:
        -----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features)

        Returns:
        --------
        regimes : np.ndarray
            Regime labels (0, 1, 2, ...)
        """
        print(f"Fitting HMM with {self.n_regimes} regimes...")

        # Fit HMM
        self.model.fit(features)

        # Predict regimes
        regimes = self.model.predict(features)
        self.regimes = regimes

        # Get prediction probabilities
        self.regime_probs = self.model.predict_proba(features)

        print(f"HMM fitted. Unique regimes: {np.unique(regimes)}")

        return regimes

    def fit_kmeans(self, features: np.ndarray) -> np.ndarray:
        """
        Fit KMeans to features and return regime labels.

        Parameters:
        -----------
        features : np.ndarray
            Feature matrix

        Returns:
        --------
        regimes : np.ndarray
            Regime labels
        """
        print(f"Fitting KMeans with {self.n_regimes} clusters...")

        # Fit KMeans
        self.model.fit(features)

        # Predict regimes
        regimes = self.model.labels_
        self.regimes = regimes

        # Calculate pseudo-probabilities (distance-based)
        distances = self.model.transform(features)
        # Convert distances to probabilities (inverse of distance)
        probs = 1 / (distances + 1e-8)
        probs = probs / probs.sum(axis=1, keepdims=True)
        self.regime_probs = probs

        print(f"KMeans fitted. Unique regimes: {np.unique(regimes)}")

        return regimes

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit regime model to features.

        Parameters:
        -----------
        features : np.ndarray
            Feature matrix

        Returns:
        --------
        regimes : np.ndarray
            Regime labels
        """
        if self.method == 'hmm':
            return self.fit_hmm(features)
        else:
            return self.fit_kmeans(features)

    def get_regimes(self) -> np.ndarray:
        """Get regime labels."""
        return self.regimes

    def get_regime_probabilities(self) -> np.ndarray:
        """Get regime probabilities."""
        return self.regime_probs

    def get_regime_transitions(self) -> np.ndarray:
        """
        Get detected regime transitions (indices where regime changes).

        Returns:
        --------
        transitions : np.ndarray
            Indices where regimes transition
        """
        if self.regimes is None:
            raise ValueError("Model not fitted yet")

        transitions = np.where(np.diff(self.regimes) != 0)[0] + 1
        return transitions

    def get_regime_duration_stats(self) -> dict:
        """
        Get statistics about regime durations.

        Returns:
        --------
        stats : dict
            Duration statistics for each regime
        """
        if self.regimes is None:
            raise ValueError("Model not fitted yet")

        stats = {}
        for regime in np.unique(self.regimes):
            regime_mask = self.regimes == regime
            regime_periods = np.split(np.where(regime_mask)[0],
                                      np.where(np.diff(np.where(regime_mask)[0]) != 1)[0] + 1)
            durations = [len(p) for p in regime_periods if len(p) > 0]

            stats[f'regime_{regime}'] = {
                'mean_duration': np.mean(durations) if durations else 0,
                'min_duration': np.min(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'n_periods': len(durations),
                'total_days': np.sum(durations)
            }

        return stats

    def characterize_regimes(self, features: np.ndarray) -> dict:
        """
        Characterize each regime by mean feature values.

        Parameters:
        -----------
        features : np.ndarray
            Original feature matrix

        Returns:
        --------
        characterization : dict
            Mean feature values for each regime
        """
        characterization = {}

        for regime in np.unique(self.regimes):
            regime_mask = self.regimes == regime
            regime_features = features[regime_mask]
            characterization[f'regime_{regime}'] = {
                'mean': np.mean(regime_features, axis=0),
                'std': np.std(regime_features, axis=0),
                'n_samples': np.sum(regime_mask)
            }

        return characterization


def detect_regimes(
    features: np.ndarray,
    n_regimes: int = 3,
    method: str = 'hmm'
) -> Tuple[np.ndarray, RegimeDetector]:
    """
    Convenience function to detect regimes.

    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    n_regimes : int, default=3
        Number of regimes
    method : str, default='hmm'
        Detection method

    Returns:
    --------
    regimes : np.ndarray
        Regime labels
    detector : RegimeDetector
        Fitted RegimeDetector object
    """
    detector = RegimeDetector(n_regimes=n_regimes, method=method)
    regimes = detector.fit(features)
    return regimes, detector
