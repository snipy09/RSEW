"""
Vercel Serverless API Endpoint for Regime Shift Early Warning System (RSEWS)
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Any, Dict
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from features import FeatureEngineer
from svd_module import RollingSVD
from instability import InstabilityMetrics
from regime_model import RegimeDetector
from signals import SignalGenerator

app = Flask(__name__)

def sanitize_json(obj: Any) -> Any:
    """Recursively convert NumPy data types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return [sanitize_json(v) for v in obj.tolist()]
    elif isinstance(obj, (float, np.float32, np.float64, np.floating)):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    elif isinstance(obj, (int, np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    else:
        return obj

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'online', 'system': 'Regime Shift Early Warning System v2.1'})

@app.route('/api/analyze', methods=['GET', 'POST'])
def analyze_regimes():
    try:
        if request.method == 'POST':
            req = request.get_json(force=True) or {}
            ticker = req.get('ticker', 'SPY').upper().strip()
            years = int(req.get('years', 3))
            mode = req.get('mode', 'adaptive')
        else:
            ticker = request.args.get('ticker', 'SPY').upper().strip()
            years = int(request.args.get('years', 3))
            mode = request.args.get('mode', 'adaptive')

        # 1. Fetch market data
        loader = DataLoader(symbol=ticker, period_years=years)
        data = loader.fetch_data()

        if data is None or len(data) < 100:
            return jsonify({'error': f'Insufficient data for ticker {ticker}'}), 400

        # 2. Extract latent feature matrix
        feature_engineer = FeatureEngineer(rolling_window=60)
        feature_matrix, _ = feature_engineer.create_features(data['close'].values, data['volume'].values if 'volume' in data else None)

        # 3. Rolling SVD Latent Factor Decomposition
        svd_model = RollingSVD(window_size=60, n_components=3)
        svd_model.rolling_svd(feature_matrix)

        # 4. Instability Metrics (Singular Value Change, Subspace Drift, Reconstruction Error)
        metrics_calc = InstabilityMetrics(svd_model)
        sv_change = metrics_calc.calculate_singular_value_change()
        subspace_drift = metrics_calc.calculate_subspace_drift()
        instability_index = metrics_calc.calculate_combined_instability()

        # 5. Gaussian Hidden Markov Model Regime Classification
        regime_detector = RegimeDetector(n_regimes=3, method='hmm')
        regimes = regime_detector.fit(feature_matrix)

        # 6. Early Warning Signals & Adaptive Thresholding
        signal_gen = SignalGenerator(instability_index, threshold=0.15)
        if mode == 'adaptive':
            signals = signal_gen.generate_adaptive_signals(lookback=40, num_std=1.2)
        else:
            signals = signal_gen.generate_smoothed_signals(window=5, threshold=0.15)

        # Align series lengths
        num_points = min(len(data.index[59:]), len(regimes), len(instability_index))
        aligned_dates = [str(d)[:10] for d in data.index[59:59+num_points]]
        aligned_closes = data['close'].values[59:59+num_points]
        aligned_returns = data['close'].pct_change().values[59:59+num_points]
        aligned_returns = np.nan_to_num(aligned_returns)

        # Backtest Strategy: Shift allocation to Cash (0% return) when signal=1
        strat_returns = np.where(signals[:num_points] == 1, 0.0, aligned_returns)
        cum_buy_hold = np.cumprod(1.0 + aligned_returns) - 1.0
        cum_strat = np.cumprod(1.0 + strat_returns) - 1.0

        # Drawdown calculation
        dd_bh, max_dd_bh = SignalGenerator.calculate_drawdowns(cum_buy_hold)
        dd_strat, max_dd_strat = SignalGenerator.calculate_drawdowns(cum_strat)

        # Risk-adjusted performance metrics
        bh_metrics = SignalGenerator.calculate_performance_metrics(aligned_returns)
        strat_metrics = SignalGenerator.calculate_performance_metrics(strat_returns)

        # Regime distribution counts
        unique, counts = np.unique(regimes[:num_points], return_counts=True)
        regime_dist = {int(k): round(float(v) / num_points * 100.0, 1) for k, v in zip(unique, counts)}

        time_series = []
        for i in range(num_points):
            time_series.append({
                'date': aligned_dates[i],
                'close': float(aligned_closes[i]),
                'regime': int(regimes[i]),
                'instability': float(instability_index[i]) if not np.isnan(instability_index[i]) else 0.0,
                'subspace_drift': float(subspace_drift[i]) if not np.isnan(subspace_drift[i]) else 0.0,
                'signal': int(signals[i]),
                'cum_buy_hold': float(cum_buy_hold[i]),
                'cum_strategy': float(cum_strat[i]),
                'drawdown_bh': float(dd_bh[i]),
                'drawdown_strat': float(dd_strat[i])
            })

        latest_regime = int(regimes[-1])
        latest_signal = int(signals[-1])
        total_warns = int(np.sum(signals[:num_points]))

        strat_final_ret = float(cum_strat[-1])
        bh_final_ret = float(cum_buy_hold[-1])
        alpha = strat_final_ret - bh_final_ret

        response = {
            'ticker': ticker,
            'years': years,
            'current_regime': latest_regime,
            'current_signal': latest_signal,
            'total_warnings': total_warns,
            'strategy_return': strat_final_ret,
            'buy_hold_return': bh_final_ret,
            'outperformance': alpha,
            'regime_distribution': regime_dist,
            'metrics': {
                'strategy': strat_metrics,
                'benchmark': bh_metrics,
                'max_drawdown_reduction': round(abs(max_dd_bh) - abs(max_dd_strat), 4)
            },
            'time_series': time_series
        }

        return jsonify(sanitize_json(response))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
