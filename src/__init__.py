"""
Regime Shift Early Warning System (RSEWS) - Package initialization
"""

__version__ = "1.0.0"
__author__ = "Quantitative Analysis Team"

from src.data_loader import DataLoader, fetch_market_data
from src.features import FeatureEngineer, engineer_features
from src.svd_module import RollingSVD, compute_rolling_svd
from src.instability import InstabilityMetrics, calculate_all_instability_metrics
from src.regime_model import RegimeDetector, detect_regimes
from src.signals import SignalGenerator, generate_early_warning_signals
from src.visualization import RegimeVisualizer, create_analysis_plots

__all__ = [
    'DataLoader',
    'fetch_market_data',
    'FeatureEngineer',
    'engineer_features',
    'RollingSVD',
    'compute_rolling_svd',
    'InstabilityMetrics',
    'calculate_all_instability_metrics',
    'RegimeDetector',
    'detect_regimes',
    'SignalGenerator',
    'generate_early_warning_signals',
    'RegimeVisualizer',
    'create_analysis_plots',
]
