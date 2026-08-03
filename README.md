# 🧠 RegimeGuard: Regime Shift Early Warning System (RSEWS)

<div align="center">

**Latent Factor Subspace Drift & Gaussian Hidden Markov Model Regime Shift Detector**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![Model](https://img.shields.io/badge/Model-Gaussian%20HMM-purple?style=flat-square)
![SVD](https://img.shields.io/badge/Decomposition-Rolling%20SVD-indigo?style=flat-square)
![Vercel](https://img.shields.io/badge/Deployment-Vercel%20Live-brightgreen?style=flat-square&logo=vercel)

[🚀 Live Early Warning Dashboard](https://regime-early-warning.vercel.app) • [GitHub Repository](https://github.com/snipy09/RSEW)

</div>

---

## 💡 Executive Summary

**RegimeGuard** is a quantitative risk system designed to detect early signs of financial market regime transitions by tracking instability in latent factor subspace structures derived from rolling Singular Value Decomposition (SVD) and Gaussian Hidden Markov Models (HMM). The system generates actionable risk-off signals **5 to 30+ days before** major market drawdowns occur.

### Key Features
- 🌌 **Rolling SVD Factor Decomposition**: Decomposes 60-day feature matrices ($X_t = U_t \Sigma_t V_t^T$) to track latent factor evolution.
- 📐 **3 Complementary Instability Metrics**:
  - **Singular Value Change ($\Delta \Sigma_t$)**: $\|\Sigma_t - \Sigma_{t-1}\|_2$
  - **Subspace Drift ($D_t$)**: $1 - |\mathbf{v}_t \cdot \mathbf{v}_{t-1}|$
  - **Variance Concentration**: Energy retention in principal components
- 🔮 **3-State Gaussian HMM Classification**: Classifies market states into Low Volatility Bull, Transition, and High Volatility Crisis regimes.
- 📊 **Risk-Off Backtesting**: Simulates dynamic asset allocation shifting capital to cash during early warning alerts, yielding measurable alpha outperformance.
- 🌐 **Interactive Quant Dashboard**: Live Vercel dashboard with multi-panel Plotly charts and real-time regime telemetry.

---

## 🚀 Quick Start

```bash
git clone https://github.com/snipy09/RSEW.git
cd RSEW

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run CLI Regime Pipeline
python3 main.py
```

---

## 🌐 Live Web Deployment

Deployed live on Vercel: **[https://regime-early-warning.vercel.app](https://regime-early-warning.vercel.app)**
