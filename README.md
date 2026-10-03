# RSEW — Regime Shift Early Warning System

Detects structural breaks and emerging volatility regimes in financial markets using rolling SVD decomposition and Hidden Markov Models, with live signal evaluation and a Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## Overview

Financial markets undergo **regime shifts** — abrupt, persistent changes in volatility structure, correlation dynamics, and return distributions that precede crises and major drawdowns. Standard indicators react after the fact. RSEW is designed to warn ahead of time.

The system ingests raw OHLCV market data from Yahoo Finance, engineers a 10-dimensional feature space, and applies rolling SVD decomposition to track structural instability continuously. A Hidden Markov Model (with KMeans fallback) classifies the market into one of three latent regimes. Early warning signals are generated, smoothed, and evaluated against known drawdown events.

---

## Key Concepts

### Regime Shifts
A regime shift is a transition between qualitatively distinct market states — calm, stressed, and crisis — characterized by changes in volatility clustering, tail behavior, and cross-asset correlations. The transitions are often non-linear and poorly captured by rolling means or standard momentum signals.

### Rolling SVD Decomposition
Singular Value Decomposition applied over a rolling window decomposes the feature matrix into orthogonal components. Changes in the singular value spectrum — particularly the dominance of the first singular value and the rate of decay — signal structural instability and are used to construct the instability metrics.

### Hidden Markov Models
HMMs treat market regimes as latent states with probabilistic emission and transition dynamics. Unlike threshold-based methods, HMMs infer regime membership from the full observation sequence, making them robust to noise and capable of capturing regime persistence.

---

## Features

| Capability | Description |
|---|---|
| 10-dim feature engineering | Log returns, rolling volatility (10/20/50d), skewness, kurtosis, autocorrelation, max drawdown, volume change |
| Rolling SVD | 60-day sliding window SVD for structural change detection |
| Instability metrics | 3 quantitative metrics derived from singular value dynamics |
| Regime detection | HMM (hmmlearn) with automatic KMeans fallback, 3 latent regimes |
| Warning signals | Binary and exponentially smoothed signals with threshold-based triggering |
| Signal evaluation | Precision, Recall, F1 Score against drawdown events |
| 4-panel dashboard | Streamlit interface with Plotly charts, neon-violet dark theme |
| Vercel deployment | Serverless-ready via `vercel.json` and `api/index.py` |

---

## Project Structure

```
RSEW/
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Fetch and preprocess OHLCV data via yfinance
│   ├── features.py           # 10-dimensional feature engineering
│   ├── svd_module.py         # Rolling SVD decomposition (60-day windows)
│   ├── instability.py        # 3 instability metrics from singular value dynamics
│   ├── regime_model.py       # RegimeDetector: HMM with KMeans fallback
│   ├── signals.py            # SignalGenerator: binary/smoothed signals + evaluation
│   ├── visualization.py      # Analysis plots (Plotly)
│   └── utils.py              # Utility functions
├── dashboard/
│   └── app.py                # Streamlit dashboard (neon-violet dark theme)
├── dashboard.py              # Dashboard entry point
├── main.py                   # RegimeShiftWarningSystem orchestrator (380+ lines)
├── test_run.py               # End-to-end system test
├── evaluate.py               # Retrospective signal evaluation
├── api/
│   └── index.py              # Vercel serverless entry point
├── vercel.json               # Vercel deployment configuration
└── requirements.txt
```

---

## Pipeline Stages

```
Raw OHLCV Data (Yahoo Finance)
        |
        v
[ Data Loader ]  -->  Validated DataFrame, NaN handling, normalization
        |
        v
[ Feature Engineering ]  -->  10-dimensional feature matrix
   log returns, vol_{10,20,50}, skewness, kurtosis,
   autocorrelation, max drawdown, volume change
        |
        v
[ Rolling SVD ]  -->  60-day window SVD
   singular values, variance explained, spectral structure
        |
        v
[ Instability Metrics ]  -->  3 scalar metrics per timestep
        |
        v
[ RegimeDetector (HMM / KMeans) ]  -->  Regime labels {0, 1, 2}
        |
        v
[ SignalGenerator ]  -->  Binary signal + exponentially smoothed signal
        |
        v
[ Evaluation ]  -->  Precision / Recall / F1 vs. drawdown events
        |
        v
[ Dashboard / Reports ]
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/snipy09/RSEW.git
cd RSEW
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

This fetches market data, engineers features, runs SVD decomposition, fits the regime model, generates warning signals, and prints evaluation metrics to stdout.

### Run Test Suite

```bash
python test_run.py
```

### Run Evaluation

```bash
python evaluate.py
```

---

## Dashboard

The Streamlit dashboard provides a live, interactive view of the system's outputs.

```bash
streamlit run dashboard/app.py
```

The interface includes four panels:

| Panel | Content |
|---|---|
| Price + Regimes | Asset price overlaid with regime classifications |
| SVD Instability | Rolling instability metric over time |
| Warning Signals | Binary and smoothed early warning signals |
| Feature Heatmap | 10-dimensional feature matrix heatmap |

The dashboard uses a dark neon-violet theme (`#0B0613` background, violet and magenta accents) rendered via Plotly and Streamlit custom CSS.

---

## Deployment

The system can be deployed as a serverless API on Vercel.

```bash
vercel deploy
```

Entry point: `api/index.py`. Configuration: `vercel.json`.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Market data | yfinance |
| Numerical core | NumPy, SciPy, pandas |
| Machine learning | scikit-learn, hmmlearn |
| Dashboard | Streamlit, Plotly |
| Deployment | Vercel (serverless) |

---

## Feature Dimensions

| Index | Feature | Description |
|---|---|---|
| 0 | Log Return | Daily log return |
| 1 | Rolling Vol 10 | 10-day realized volatility |
| 2 | Rolling Vol 20 | 20-day realized volatility |
| 3 | Rolling Vol 50 | 50-day realized volatility |
| 4 | Skewness | Rolling return skewness |
| 5 | Kurtosis | Rolling excess kurtosis |
| 6 | Autocorrelation | Lag-1 autocorrelation of returns |
| 7 | Max Drawdown | Rolling maximum drawdown |
| 8 | Volume Change | Day-over-day volume change (log) |
| 9 | Vol Ratio | Short-to-long volatility ratio (10d/50d) |

---

## Instability Metrics

| Metric | Signal |
|---|---|
| Spectral dominance | Fraction of variance in the first singular value — rises as system compresses into a single mode of variation |
| Spectral entropy | Shannon entropy of the normalized singular value spectrum — drops as complexity collapses |
| Singular value velocity | Rate of change of the leading singular value — spikes precede structural breaks |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with Python. Designed for quantitative research and portfolio risk monitoring.*
