"""
Rolling SVD module for latent factor analysis.
"""

import numpy as np
from scipy.linalg import svd
from typing import Tuple, List
import sys
import os

# Handle imports
try:
    from utils import standardize_features
except ImportError:
    from .utils import standardize_features


class RollingSVD:
    """Compute rolling SVD on feature matrices."""

    def __init__(self, window_size: int = 60, n_components: int = 3):
        """
        Initialize RollingSVD.

        Parameters:
        -----------
        window_size : int, default=60
            Rolling window size
        n_components : int, default=3
            Number of singular vectors to keep
        """
        self.window_size = window_size
        self.n_components = n_components

        # Storage for SVD results
        self.singular_values_list = []
        self.u_matrices = []
        self.vt_matrices = []
        self.explained_variance_ratios = []

    def compute_svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SVD on a matrix.

        Parameters:
        -----------
        X : np.ndarray
            Input matrix of shape (n_samples, n_features)

        Returns:
        --------
        U : np.ndarray
            Left singular vectors
        singular_values : np.ndarray
            Singular values
        Vt : np.ndarray
            Right singular vectors (transposed)
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute SVD
        U, s, Vt = svd(X_centered, full_matrices=False)

        # Keep only top n_components
        U = U[:, :self.n_components]
        s = s[:self.n_components]
        Vt = Vt[:self.n_components, :]

        return U, s, Vt

    def rolling_svd(self, features: np.ndarray) -> None:
        """
        Apply rolling SVD to feature matrix.

        Parameters:
        -----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        n_samples = features.shape[0]

        # Standardize features
        features_std, _ = standardize_features(features, method='zscore')

        print(
            f"Computing rolling SVD with window={self.window_size}, components={self.n_components}...")

        # Apply rolling SVD
        for i in range(n_samples - self.window_size + 1):
            # Get window
            X_window = features_std[i:i + self.window_size]

            # Compute SVD
            U, s, Vt = self.compute_svd(X_window)

            # Store results
            self.singular_values_list.append(s)
            self.u_matrices.append(U)
            self.vt_matrices.append(Vt)

            # Calculate explained variance ratio
            total_variance = np.sum(s ** 2)
            evr = (s ** 2) / (total_variance + 1e-8)
            self.explained_variance_ratios.append(evr)

        print(f"Computed {len(self.singular_values_list)} SVD decompositions")

    def get_singular_values(self) -> np.ndarray:
        """Get all singular values as array."""
        return np.array(self.singular_values_list)

    def get_explained_variance_ratios(self) -> np.ndarray:
        """Get explained variance ratios."""
        return np.array(self.explained_variance_ratios)

    def get_u_matrices(self) -> List[np.ndarray]:
        """Get U matrices."""
        return self.u_matrices

    def get_vt_matrices(self) -> List[np.ndarray]:
        """Get Vt matrices."""
        return self.vt_matrices

    def get_primary_component_variance(self) -> np.ndarray:
        """
        Get variance explained by first principal component.

        Returns:
        --------
        primary_variance : np.ndarray
            Variance of first component over time
        """
        svd_values = self.get_singular_values()
        primary_variance = (svd_values[:, 0] ** 2) / \
            (np.sum(svd_values ** 2, axis=1) + 1e-8)
        return primary_variance

    def get_subspace_basis(self, time_idx: int) -> np.ndarray:
        """
        Get the subspace basis (first k principal components) at a specific time.

        Parameters:
        -----------
        time_idx : int
            Time index

        Returns:
        --------
        basis : np.ndarray
            Subspace basis of shape (n_features, n_components)
        """
        if time_idx >= len(self.vt_matrices):
            raise IndexError(f"Time index {time_idx} out of range")

        # The columns of V form the basis
        # Vt is (n_components, n_features), so V = Vt.T is (n_features, n_components)
        return self.vt_matrices[time_idx].T

    def get_reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for each rolling window.

        Parameters:
        -----------
        features : np.ndarray
            Original feature matrix

        Returns:
        --------
        reconstruction_errors : np.ndarray
            Reconstruction error at each time step
        """
        features_std, _ = standardize_features(features, method='zscore')

        reconstruction_errors = []

        for i in range(len(self.u_matrices)):
            X_window = features_std[i:i + self.window_size]
            X_centered = X_window - np.mean(X_window, axis=0)

            U = self.u_matrices[i]
            s = self.singular_values_list[i]

            # Reconstruct
            X_reconstructed = U @ np.diag(s) @ self.vt_matrices[i]

            # Error
            error = np.linalg.norm(X_centered - X_reconstructed, 'fro')
            reconstruction_errors.append(error)

        return np.array(reconstruction_errors)


def compute_rolling_svd(
    features: np.ndarray,
    window_size: int = 60,
    n_components: int = 3
) -> RollingSVD:
    """
    Convenience function to compute rolling SVD.

    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    window_size : int, default=60
        Rolling window size
    n_components : int, default=3
        Number of components to keep

    Returns:
    --------
    rolling_svd : RollingSVD
        Fitted RollingSVD object
    """
    rolling_svd_obj = RollingSVD(
        window_size=window_size, n_components=n_components)
    rolling_svd_obj.rolling_svd(features)
    return rolling_svd_obj
