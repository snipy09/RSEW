"""
Historical Lead-Time & Drawdown Protection Evaluator for RegimeGuard (RSEW)
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import DataLoader
from features import FeatureEngineer
from svd_module import RollingSVD
from instability import InstabilityMetrics
from regime_model import RegimeDetector
from signals import SignalGenerator

def run_evaluation():
    print("\n" + "="*75)
    print(" REGIMEGUARD (RSEW): HISTORICAL DRAWDOWN PROTECTION AUDIT")
    print("="*75)

    loader = DataLoader(symbol='SPY', period_years=5)
    data = loader.fetch_data()
    
    feature_engineer = FeatureEngineer(rolling_window=60)
    feature_matrix, _ = feature_engineer.create_features(data['close'].values, data['volume'].values if 'volume' in data else None)

    svd_model = RollingSVD(window_size=60, n_components=3)
    svd_model.rolling_svd(feature_matrix)

    metrics_calc = InstabilityMetrics(svd_model)
    instability_index = metrics_calc.calculate_combined_instability()

    regime_detector = RegimeDetector(n_regimes=3, method='hmm')
    regimes = regime_detector.fit(feature_matrix)

    signal_gen = SignalGenerator(instability_index, threshold=0.15)
    signals = signal_gen.generate_smoothed_signals(window=5, threshold=0.15)

    num_points = min(len(data.index[59:]), len(regimes), len(instability_index))
    aligned_closes = data['close'].values[59:59+num_points]
    aligned_returns = data['close'].pct_change().values[59:59+num_points]

    def calc_max_drawdown(rets):
        cum = np.cumprod(1 + np.nan_to_num(rets))
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return np.min(dd)

    strat_returns = np.where(signals[:num_points] == 1, 0.0, aligned_returns)
    
    mdd_benchmark = calc_max_drawdown(aligned_returns)
    mdd_strategy = calc_max_drawdown(strat_returns)
    
    cum_bh = np.cumprod(1 + np.nan_to_num(aligned_returns))[-1] - 1.0
    cum_strat = np.cumprod(1 + np.nan_to_num(strat_returns))[-1] - 1.0

    print(f"  • Test Horizon:              5 Years SPY Data ({num_points} Days)")
    print(f"  • Benchmark Max Drawdown:    {mdd_benchmark*100:.2f}%")
    print(f"  • SVD Strategy Max Drawdown: {mdd_strategy*100:.2f}%")
    print(f"  • Tail Risk Reduction:       {abs(mdd_benchmark - mdd_strategy)*100:.2f}% Max Drawdown Protection")
    print("="*75 + "\n")

if __name__ == '__main__':
    run_evaluation()
