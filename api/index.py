"""
Vercel Serverless API Endpoint for Regime Shift Early Warning System (RSEWS)
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Any, Dict

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
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'online', 'system': 'Regime Shift Early Warning System v2.0'})

@app.route('/api/analyze', methods=['GET', 'POST'])
def analyze_regimes():
    try:
        if request.method == 'POST':
            req = request.get_json(force=True) or {}
            ticker = req.get('ticker', 'SPY').upper().strip()
            years = int(req.get('years', 3))
        else:
            ticker = request.args.get('ticker', 'SPY').upper().strip()
            years = int(request.args.get('years', 3))

        # 1. Fetch market data
        loader = DataLoader(symbol=ticker, period_years=years)
        data = loader.fetch_data()

        if data is None or len(data) < 120:
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

        # 6. Early Warning Signals & Risk-Off Backtesting
        signal_gen = SignalGenerator(instability_index, threshold=0.15)
        signals = signal_gen.generate_smoothed_signals(window=5, threshold=0.15)

        # Align series lengths
        num_points = min(len(data.index[59:]), len(regimes), len(instability_index))
        aligned_dates = [str(d)[:10] for d in data.index[59:59+num_points]]
        aligned_closes = data['close'].values[59:59+num_points]
        aligned_returns = data['close'].pct_change().values[59:59+num_points]

        # Backtest Strategy: Shift allocation to Cash (0% return) when signal=1
        strat_returns = np.where(signals[:num_points] == 1, 0.0, aligned_returns)
        cum_buy_hold = np.cumprod(1 + np.nan_to_num(aligned_returns)) - 1.0
        cum_strat = np.cumprod(1 + np.nan_to_num(strat_returns)) - 1.0

        time_series = []
        for i in range(num_points):
            time_series.append({
                'date': aligned_dates[i],
                'close': float(aligned_closes[i]),
                'regime': int(regimes[i]),
                'sing_val_change': float(sv_change[i]),
                'subspace_drift': float(subspace_drift[i]),
                'instability_index': float(instability_index[i]),
                'warning_signal': int(signals[i]),
                'buy_hold_cum': float(cum_buy_hold[i]),
                'strategy_cum': float(cum_strat[i])
            })

        response = {
            'success': True,
            'ticker': ticker,
            'total_days': num_points,
            'unique_regimes': np.unique(regimes[:num_points]).tolist(),
            'current_regime': int(regimes[num_points-1]),
            'current_signal': int(signals[num_points-1]),
            'total_warnings': int(np.sum(signals[:num_points])),
            'buy_hold_return': float(cum_buy_hold[-1]),
            'strategy_return': float(cum_strat[-1]),
            'outperformance': float(cum_strat[-1] - cum_buy_hold[-1]),
            'time_series': time_series
        }

        return jsonify(sanitize_json(response))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel entrypoint
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
