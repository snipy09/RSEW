"""
Instability metrics module for detecting regime shifts.
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os

# Handle imports
try:
    from utils import normalize_scores
    from svd_module import RollingSVD
except ImportError:
    from .utils import normalize_scores
    from .svd_module import RollingSVD


class InstabilityMetrics:
    """Calculate instability metrics from rolling SVD results."""

    def __init__(self, rolling_svd: RollingSVD):
        """
        Initialize InstabilityMetrics.

        Parameters:
        -----------
        rolling_svd : RollingSVD
            Fitted RollingSVD object
        """
        self.rolling_svd = rolling_svd
        self.singular_value_changes = None
        self.subspace_drifts = None
        self.explained_variance_ratios = None
        self.combined_instability = None

    def calculate_singular_value_change(self) -> np.ndarray:
        """
        Calculate change in singular values (ΔΣ_t).

        ΔΣ_t = ||Σ_t - Σ_{t-1}||_2

        Returns:
        --------
        sv_changes : np.ndarray
            Singular value changes over time
        """
        sv_array = self.rolling_svd.get_singular_values()

        sv_changes = np.zeros(len(sv_array))
        sv_changes[0] = 0  # First value is undefined

        for t in range(1, len(sv_array)):
            # L2 norm of difference
            sv_changes[t] = np.linalg.norm(sv_array[t] - sv_array[t-1])

        self.singular_value_changes = sv_changes
        return sv_changes

    def calculate_subspace_drift(self) -> np.ndarray:
        """
        Calculate subspace drift (D_t).

        D_t = 1 - |V_t · V_{t-1}| (average absolute cosine similarity)

        Returns:
        --------
        subspace_drifts : np.ndarray
            Subspace drift over time
        """
        vt_matrices = self.rolling_svd.get_vt_matrices()

        drifts = np.zeros(len(vt_matrices))
        drifts[0] = 0  # First value is undefined

        for t in range(1, len(vt_matrices)):
            V_t = vt_matrices[t].T  # Convert to (n_features, n_components)
            V_prev = vt_matrices[t-1].T

            # Calculate cosine similarity between subspaces
            # Use Frobenius norm of dot product normalized
            dot_product = np.abs(np.dot(V_t.T, V_prev))  # (n_comp, n_comp)

            # Average of absolute values
            avg_similarity = np.mean(np.abs(dot_product.diagonal()))

            # Drift = 1 - similarity
            drifts[t] = 1 - np.clip(avg_similarity, 0, 1)

        self.subspace_drifts = drifts
        return drifts

    def calculate_explained_variance_ratio(self) -> np.ndarray:
        """
        Calculate explained variance ratio of first component (R_t).

        R_t = σ_1^2 / Σ(σ_i^2)

        Returns:
        --------
        evr : np.ndarray
            Explained variance ratio over time
        """
        evr = self.rolling_svd.get_primary_component_variance()
        self.explained_variance_ratios = evr
        return evr

    def calculate_variance_concentration(self) -> np.ndarray:
        """
        Calculate how concentrated variance is in first component.
        Higher concentration (>0.5) indicates dominant regime, lower (<0.5) indicates transition.

        Returns:
        --------
        concentration : np.ndarray
            Variance concentration metric
        """
        evr = self.calculate_explained_variance_ratio()

        # Concentration = 1 - entropy
        # Use first component variance as proxy
        concentration = evr

        return concentration

    def calculate_combined_instability(
        self,
        weights: Dict[str, float] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Calculate combined instability score.

        Instability(t) = w1 * ΔΣ_t + w2 * D_t + w3 * (1 - R_t)

        Parameters:
        -----------
        weights : Dict[str, float], optional
            Weights for metrics: {'sv_change', 'subspace_drift', 'variance_concentration'}
            Defaults to equal weights (1/3 each)
        normalize : bool, default=True
            Normalize individual metrics before combining

        Returns:
        --------
        instability : np.ndarray
            Combined instability score
        """
        # Get individual metrics
        sv_change = self.calculate_singular_value_change()
        subspace_drift = self.calculate_subspace_drift()
        # Lower EVR = more instability
        variance_conc = 1 - self.calculate_explained_variance_ratio()

        # Normalize if requested
        if normalize:
            sv_change_norm = normalize_scores(sv_change, method='zscore')
            subspace_drift_norm = normalize_scores(
                subspace_drift, method='zscore')
            variance_conc_norm = normalize_scores(
                variance_conc, method='zscore')
        else:
            sv_change_norm = sv_change
            subspace_drift_norm = subspace_drift
            variance_conc_norm = variance_conc

        # Set default weights if not provided
        if weights is None:
            weights = {
                'sv_change': 1/3,
                'subspace_drift': 1/3,
                'variance_concentration': 1/3
            }

        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        # Combine metrics
        instability = (
            weights['sv_change'] * sv_change_norm +
            weights['subspace_drift'] * subspace_drift_norm +
            weights['variance_concentration'] * variance_conc_norm
        )

        # Handle NaN and inf
        instability = np.nan_to_num(
            instability, nan=0.0, posinf=0.0, neginf=0.0)

        self.combined_instability = instability
        return instability

    def get_metrics_summary(self) -> Dict[str, np.ndarray]:
        """
        Get summary of all metrics.

        Returns:
        --------
        metrics : Dict[str, np.ndarray]
            Dictionary with all calculated metrics
        """
        metrics = {
            'singular_value_change': self.singular_value_changes if self.singular_value_changes is not None else self.calculate_singular_value_change(),
            'subspace_drift': self.subspace_drifts if self.subspace_drifts is not None else self.calculate_subspace_drift(),
            'explained_variance_ratio': self.explained_variance_ratios if self.explained_variance_ratios is not None else self.calculate_explained_variance_ratio(),
            'combined_instability': self.combined_instability if self.combined_instability is not None else self.calculate_combined_instability()
        }

        return metrics


def calculate_all_instability_metrics(
    rolling_svd: RollingSVD,
    weights: Dict[str, float] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function to calculate all instability metrics.

    Parameters:
    -----------
    rolling_svd : RollingSVD
        Fitted RollingSVD object
    weights : Dict[str, float], optional
        Weights for combined score

    Returns:
    --------
    combined_instability : np.ndarray
        Combined instability score
    metrics : Dict[str, np.ndarray]
        Individual metrics
    """
    metrics_obj = InstabilityMetrics(rolling_svd)
    combined = metrics_obj.calculate_combined_instability(weights=weights)
    metrics = metrics_obj.get_metrics_summary()

    return combined, metrics
