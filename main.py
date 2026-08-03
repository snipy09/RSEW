"""
Main orchestrator for the Regime Shift Early Warning System.
Coordinates all modules to perform end-to-end analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path BEFORE importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import fetch_market_data, DataLoader
from features import engineer_features
from svd_module import compute_rolling_svd
from instability import calculate_all_instability_metrics
from regime_model import detect_regimes
from signals import generate_early_warning_signals
from visualization import create_analysis_plots, RegimeVisualizer
from utils import print_section, normalize_scores


class RegimeShiftWarningSystem:
    """Main system orchestrator."""
    
    def __init__(
        self,
        symbol: str = 'SPY',
        period_years: int = 10,
        rolling_window: int = 60,
        n_regimes: int = 3,
        n_components: int = 3,
        signal_threshold: float = 0.5,
        output_dir: str = 'results'
    ):
        """Initialize system."""
        self.symbol = symbol
        self.period_years = period_years
        self.rolling_window = rolling_window
        self.n_regimes = n_regimes
        self.n_components = n_components
        self.signal_threshold = signal_threshold
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.data = None
        self.prices = None
        self.volumes = None
        self.dates = None
        self.features = None
        self.feature_dates = None
        self.rolling_svd = None
        self.instability_scores = None
        self.metrics = None
        self.regimes = None
        self.regime_detector = None
        self.signals = None
        self.signal_generator = None
        self.evaluation_metrics = None
    
    def run(self, verbose: bool = True) -> dict:
        """Run full analysis pipeline."""
        results = {}
        
        try:
            if verbose:
                print_section("STEP 1: Data Loading")
            self._load_data(verbose)
            results['data_loaded'] = True
            results['data_points'] = len(self.prices)
            
            if verbose:
                print_section("STEP 2: Feature Engineering")
            self._engineer_features(verbose)
            results['features_created'] = True
            results['n_features'] = self.features.shape[1]
            
            if verbose:
                print_section("STEP 3: Rolling SVD Analysis")
            self._apply_rolling_svd(verbose)
            results['svd_computed'] = True
            
            if verbose:
                print_section("STEP 4: Instability Metrics")
            self._calculate_instability_metrics(verbose)
            results['instability_calculated'] = True
            
            if verbose:
                print_section("STEP 5: Regime Detection")
            self._detect_regimes(verbose)
            results['regimes_detected'] = True
            results['n_regimes_found'] = len(np.unique(self.regimes))
            
            if verbose:
                print_section("STEP 6: Early Warning Signal Generation")
            self._generate_signals(verbose)
            results['signals_generated'] = True
            
            if verbose:
                print_section("STEP 7: Creating Visualizations")
            self._create_visualizations(verbose)
            results['visualizations_created'] = True
            
            if verbose:
                print_section("STEP 8: Results Summary")
            self._print_summary(verbose)
            results['status'] = 'SUCCESS'
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def _load_data(self, verbose: bool = True):
        """Load market data."""
        try:
            self.data = fetch_market_data(self.symbol, self.period_years)
            self.prices, self.dates = self.data['close'].values, self.data['date'].values
            self.volumes = self.data.get('volume', np.ones(len(self.prices))).values
            
            if verbose:
                print(f"✓ Loaded {len(self.prices)} trading days")
                print(f"  Date range: {self.dates[0]} to {self.dates[-1]}")
                print(f"  Price range: ${self.prices.min():.2f} - ${self.prices.max():.2f}")
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {e}")
    
    def _engineer_features(self, verbose: bool = True):
        """Create features."""
        try:
            self.features, self.feature_dates = engineer_features(
                self.prices,
                self.volumes,
                rolling_window=self.rolling_window,
                simple=False
            )
            
            if verbose:
                print(f"✓ Created {self.features.shape[0]} feature samples")
                print(f"  Feature dimensions: {self.features.shape[1]}")
                print(f"  Feature range: [{self.features.min():.3f}, {self.features.max():.3f}]")
        except Exception as e:
            raise RuntimeError(f"Feature engineering failed: {e}")
    
    def _apply_rolling_svd(self, verbose: bool = True):
        """Apply rolling SVD."""
        try:
            self.rolling_svd = compute_rolling_svd(
                self.features,
                window_size=self.rolling_window,
                n_components=self.n_components
            )
            
            svd_values = self.rolling_svd.get_singular_values()
            evr = self.rolling_svd.get_primary_component_variance()
            
            if verbose:
                print(f"✓ Computed {len(svd_values)} SVD decompositions")
                print(f"  Avg primary component variance: {evr.mean():.3f}")
                print(f"  Variance concentration range: [{evr.min():.3f}, {evr.max():.3f}]")
        except Exception as e:
            raise RuntimeError(f"SVD computation failed: {e}")
    
    def _calculate_instability_metrics(self, verbose: bool = True):
        """Calculate instability metrics."""
        try:
            self.instability_scores, self.metrics = calculate_all_instability_metrics(
                self.rolling_svd
            )
            
            self.instability_scores = normalize_scores(self.instability_scores, method='zscore')
            
            if verbose:
                print(f"✓ Calculated 3 instability metrics")
                print(f"  SV Change - mean: {self.metrics['singular_value_change'].mean():.4f}")
                print(f"  Subspace Drift - mean: {self.metrics['subspace_drift'].mean():.4f}")
                print(f"  Explained Variance - mean: {self.metrics['explained_variance_ratio'].mean():.4f}")
                print(f"  Combined Instability - range: [{self.instability_scores.min():.3f}, {self.instability_scores.max():.3f}]")
        except Exception as e:
            raise RuntimeError(f"Instability calculation failed: {e}")
    
    def _detect_regimes(self, verbose: bool = True):
        """Detect market regimes."""
        try:
            self.regimes, self.regime_detector = detect_regimes(
                self.features,
                n_regimes=self.n_regimes,
                method='hmm'
            )
            
            transitions = self.regime_detector.get_regime_transitions()
            stats = self.regime_detector.get_regime_duration_stats()
            
            if verbose:
                print(f"✓ Detected {len(np.unique(self.regimes))} distinct regimes")
                print(f"  Detected {len(transitions)} regime transitions")
                for regime, regime_stats in stats.items():
                    print(f"  {regime}: avg duration = {regime_stats['mean_duration']:.1f} days")
        except Exception as e:
            raise RuntimeError(f"Regime detection failed: {e}")
    
    def _generate_signals(self, verbose: bool = True):
        """Generate early warning signals."""
        try:
            transitions = self.regime_detector.get_regime_transitions()
            
            self.signals, self.signal_generator, self.evaluation_metrics = generate_early_warning_signals(
                self.instability_scores,
                transitions,
                threshold=self.signal_threshold,
                optimize=True
            )
            
            sig_stats = self.signal_generator.get_signal_statistics()
            
            if verbose:
                print(f"✓ Generated early warning signals")
                print(f"  Signal percentage: {sig_stats['signal_percentage']:.2f}%")
                print(f"  Number of signal periods: {sig_stats['n_signal_periods']}")
                print(f"  Avg signal duration: {sig_stats['avg_signal_duration']:.1f} days")
                print(f"  Precision: {self.evaluation_metrics.get('precision', 0):.3f}")
                print(f"  Recall: {self.evaluation_metrics.get('recall', 0):.3f}")
                print(f"  F1 Score: {self.evaluation_metrics.get('f1', 0):.3f}")
                print(f"  Avg lead time: {self.evaluation_metrics.get('avg_lead_time', 0):.1f} days")
        except Exception as e:
            raise RuntimeError(f"Signal generation failed: {e}")
    
    def _create_visualizations(self, verbose: bool = True):
        """Create analysis visualizations."""
        try:
            transitions = self.regime_detector.get_regime_transitions()
            
            create_analysis_plots(
                self.dates,
                self.prices,
                self.regimes,
                self.instability_scores,
                self.signals,
                transitions,
                self.evaluation_metrics,
                self.output_dir
            )
            
            if verbose:
                print(f"✓ Created comprehensive visualization plots")
                print(f"  Saved to: {self.output_dir}/comprehensive_analysis.png")
        except Exception as e:
            raise RuntimeError(f"Visualization creation failed: {e}")
    
    def _print_summary(self, verbose: bool = True):
        """Print summary statistics."""
        if not verbose:
            return
        
        print(f"\n{'='*60}")
        print(f"REGIME SHIFT EARLY WARNING SYSTEM - SUMMARY")
        print(f"{'='*60}")
        print(f"\nSymbol: {self.symbol}")
        print(f"Analysis Period: {self.period_years} years")
        print(f"Data Points: {len(self.prices):,}")
        print(f"\nFeatures: {self.features.shape[1]} dimensions")
        print(f"Rolling Window: {self.rolling_window} days")
        print(f"SVD Components: {self.n_components}")
        print(f"\nRegimes Detected: {len(np.unique(self.regimes))}")
        print(f"Transitions: {len(self.regime_detector.get_regime_transitions())}")
        
        print(f"\nSignal Performance:")
        print(f"  Precision: {self.evaluation_metrics.get('precision', 0):.1%}")
        print(f"  Recall: {self.evaluation_metrics.get('recall', 0):.1%}")
        print(f"  F1 Score: {self.evaluation_metrics.get('f1', 0):.1%}")
        print(f"  Avg Lead Time: {self.evaluation_metrics.get('avg_lead_time', 0):.1f} days")
        print(f"  False Alarm Rate: {self.evaluation_metrics.get('false_alarm_rate', 0):.1%}")
        
        print(f"\n{'='*60}")
        print(f"Analysis complete! Results saved to: {self.output_dir}/")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Regime Shift Early Warning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol SPY --period 10 --threshold 0.5
  python main.py --symbol NIFTY --regimes 4 --verbose
        """
    )
    
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Stock symbol (default: SPY)')
    parser.add_argument('--period', type=int, default=10,
                       help='Historical data period in years (default: 10)')
    parser.add_argument('--window', type=int, default=60,
                       help='Rolling window size in days (default: 60)')
    parser.add_argument('--regimes', type=int, default=3,
                       help='Number of regimes (default: 3)')
    parser.add_argument('--components', type=int, default=3,
                       help='SVD components (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Signal threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    system = RegimeShiftWarningSystem(
        symbol=args.symbol,
        period_years=args.period,
        rolling_window=args.window,
        n_regimes=args.regimes,
        n_components=args.components,
        signal_threshold=args.threshold,
        output_dir=args.output
    )
    
    results = system.run(verbose=args.verbose)
    return results


if __name__ == '__main__':
    main()
