"""
Visualization module for plotting analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import warnings

warnings.filterwarnings('ignore')


class RegimeVisualizer:
    """Create visualizations for regime analysis."""

    def __init__(self, figsize: tuple = (16, 12), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.

        Parameters:
        -----------
        figsize : tuple, default=(16, 12)
            Figure size
        style : str, default='seaborn-v0_8-darkgrid'
            Matplotlib style
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            pass

        # Color palette for regimes
        self.regime_colors = {
            0: '#1f77b4',  # Blue
            1: '#ff7f0e',  # Orange
            2: '#2ca02c',  # Green
            3: '#d62728',  # Red
            4: '#9467bd',  # Purple
            5: '#8c564b'   # Brown
        }

        self.signal_color = '#ff0000'
        self.alpha_regime = 0.3
        self.alpha_signal = 0.2

    def plot_comprehensive_analysis(
        self,
        dates: np.ndarray,
        prices: np.ndarray,
        regimes: np.ndarray,
        instability_scores: np.ndarray,
        signals: np.ndarray,
        regime_transitions: np.ndarray,
        metrics: dict = None,
        save_path: str = None
    ):
        """
        Create comprehensive 4-panel analysis plot.

        Parameters:
        -----------
        dates : np.ndarray
            Dates corresponding to data points
        prices : np.ndarray
            Price data
        regimes : np.ndarray
            Regime labels
        instability_scores : np.ndarray
            Instability scores
        signals : np.ndarray
            Warning signals
        regime_transitions : np.ndarray
            Indices of regime transitions
        metrics : dict, optional
            Evaluation metrics to display
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(4, 1, figsize=self.figsize, sharex=True)

        # Align data lengths
        n_regimes = len(regimes)
        n_prices = len(prices)
        n_instability = len(instability_scores)
        n_signals = len(signals)

        min_len = min(n_prices, n_regimes, n_instability, n_signals)

        dates_subset = dates[-min_len:]
        prices_subset = prices[-min_len:]
        regimes_subset = regimes[-min_len:]
        instability_subset = instability_scores[-min_len:]
        signals_subset = signals[-min_len:]

        # --- Panel 1: Price with regimes ---
        ax1 = axes[0]
        ax1.plot(dates_subset, prices_subset, 'k-', linewidth=2, label='Price')

        # Shade regimes
        for regime in np.unique(regimes_subset):
            mask = regimes_subset == regime
            ax1.fill_between(
                dates_subset,
                prices_subset.min(),
                prices_subset.max(),
                where=mask,
                alpha=self.alpha_regime,
                color=self.regime_colors.get(regime, '#cccccc'),
                label=f'Regime {regime}'
            )

        # Mark transitions
        for transition_idx in regime_transitions:
            if transition_idx < len(dates_subset):
                ax1.axvline(dates_subset[transition_idx], color='red',
                            linestyle='--', alpha=0.7, linewidth=1.5)

        ax1.set_ylabel('Price', fontsize=11, fontweight='bold')
        ax1.set_title('Market Price with Detected Regimes and Transitions',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Instability Score ---
        ax2 = axes[1]
        ax2.plot(dates_subset, instability_subset, 'b-',
                 linewidth=2, label='Instability Score')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

        # Highlight high instability
        high_instability = instability_subset > np.percentile(
            instability_subset, 75)
        ax2.fill_between(
            dates_subset,
            instability_subset.min(),
            instability_subset.max(),
            where=high_instability,
            alpha=0.2,
            color='orange',
            label='High Instability'
        )

        # Mark transitions
        for transition_idx in regime_transitions:
            if transition_idx < len(dates_subset):
                ax2.axvline(dates_subset[transition_idx], color='red',
                            linestyle='--', alpha=0.7, linewidth=1.5)

        ax2.set_ylabel('Instability', fontsize=11, fontweight='bold')
        ax2.set_title('Latent Factor Instability Over Time',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # --- Panel 3: Warning Signals ---
        ax3 = axes[2]
        signal_periods = np.where(signals_subset == 1)[0]
        if len(signal_periods) > 0:
            ax3.scatter(
                dates_subset[signal_periods],
                np.ones(len(signal_periods)),
                color=self.signal_color,
                s=50,
                alpha=0.7,
                label='Warning Signal',
                marker='v'
            )

        ax3.fill_between(
            dates_subset,
            0,
            signals_subset,
            alpha=self.alpha_signal,
            color=self.signal_color,
            label='Signal Period'
        )

        # Mark transitions
        for transition_idx in regime_transitions:
            if transition_idx < len(dates_subset):
                ax3.axvline(dates_subset[transition_idx], color='red', linestyle='--',
                            alpha=0.7, linewidth=1.5, label='Regime Transition')

        ax3.set_ylim([-0.1, 1.1])
        ax3.set_ylabel('Signal', fontsize=11, fontweight='bold')
        ax3.set_title('Early Warning Signals (1=Warning, 0=Normal)',
                      fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # --- Panel 4: Signal Quality Metrics ---
        ax4 = axes[3]
        ax4.axis('off')

        # Create metrics text
        if metrics:
            metrics_text = (
                f"Signal Evaluation Metrics\n"
                f"{'='*40}\n"
                f"Precision: {metrics.get('precision', 0):.3f}\n"
                f"Recall: {metrics.get('recall', 0):.3f}\n"
                f"F1 Score: {metrics.get('f1', 0):.3f}\n"
                f"False Alarm Rate: {metrics.get('false_alarm_rate', 0):.3%}\n"
                f"\n"
                f"Lead Time Analysis\n"
                f"{'='*40}\n"
                f"Avg Lead Time: {metrics.get('avg_lead_time', 0):.1f} days\n"
                f"Median Lead Time: {metrics.get('median_lead_time', 0):.1f} days\n"
                f"Min Lead Time: {metrics.get('min_lead_time', 0):.1f} days\n"
                f"Max Lead Time: {metrics.get('max_lead_time', 0):.1f} days\n"
                f"Successful Signals: {metrics.get('n_successful_signals', 0)}\n"
            )
        else:
            metrics_text = "Metrics not available"

        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comprehensive plot to {save_path}")

        return fig, axes

    def plot_individual_metrics(
        self,
        dates: np.ndarray,
        metrics_dict: dict,
        regime_transitions: np.ndarray = None,
        save_path: str = None
    ):
        """
        Plot individual instability metrics.

        Parameters:
        -----------
        dates : np.ndarray
            Dates
        metrics_dict : dict
            Dictionary with metric names and values
        regime_transitions : np.ndarray, optional
            Regime transitions to mark
        save_path : str, optional
            Path to save
        """
        n_metrics = len(metrics_dict)
        fig, axes = plt.subplots(
            n_metrics, 1, figsize=(14, 3*n_metrics), sharex=True)

        if n_metrics == 1:
            axes = [axes]

        for idx, (name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx]

            # Align lengths
            min_len = min(len(dates), len(values))
            dates_subset = dates[-min_len:]
            values_subset = values[-min_len:]

            ax.plot(dates_subset, values_subset, linewidth=2, label=name)
            ax.fill_between(dates_subset, values_subset, alpha=0.3)

            if regime_transitions is not None:
                for trans in regime_transitions:
                    if trans < len(dates_subset):
                        ax.axvline(
                            dates_subset[trans], color='red', linestyle='--', alpha=0.5)

            ax.set_ylabel(name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')

        fig.suptitle('Individual Instability Metrics',
                     fontsize=14, fontweight='bold', y=1.001)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics plot to {save_path}")

        return fig, axes

    def plot_regime_characteristics(
        self,
        characterization: dict,
        save_path: str = None
    ):
        """
        Plot regime characteristics as heatmap.

        Parameters:
        -----------
        characterization : dict
            Regime characteristics
        save_path : str, optional
            Path to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data
        regimes = list(characterization.keys())
        n_features = len(characterization[regimes[0]]['mean'])

        data = np.zeros((len(regimes), n_features))
        for i, regime in enumerate(regimes):
            data[i] = characterization[regime]['mean']

        # Normalize
        data_norm = (data - data.min(axis=0)) / \
            (data.max(axis=0) - data.min(axis=0) + 1e-8)

        im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(len(regimes)))
        ax.set_xticklabels([f'F{i}' for i in range(n_features)])
        ax.set_yticklabels(regimes)

        ax.set_xlabel('Features', fontsize=11, fontweight='bold')
        ax.set_ylabel('Regimes', fontsize=11, fontweight='bold')
        ax.set_title('Regime Characteristics (Normalized)',
                     fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', rotation=270, labelpad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved regime characteristics plot to {save_path}")

        return fig, ax


def create_analysis_plots(
    dates: np.ndarray,
    prices: np.ndarray,
    regimes: np.ndarray,
    instability_scores: np.ndarray,
    signals: np.ndarray,
    regime_transitions: np.ndarray,
    metrics: dict = None,
    output_dir: str = 'results'
):
    """
    Create all analysis plots.

    Parameters:
    -----------
    dates, prices, regimes, instability_scores, signals : arrays
        Analysis data
    regime_transitions : np.ndarray
        Regime transition indices
    metrics : dict, optional
        Evaluation metrics
    output_dir : str, default='results'
        Output directory
    """
    visualizer = RegimeVisualizer()

    # Comprehensive plot
    visualizer.plot_comprehensive_analysis(
        dates, prices, regimes, instability_scores, signals,
        regime_transitions, metrics,
        save_path=f"{output_dir}/comprehensive_analysis.png"
    )

    print(f"Plots saved to {output_dir}/")
