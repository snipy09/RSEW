"""
Interactive Streamlit dashboard for the Regime Shift Early Warning System.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make project root importable when running:
# python -m streamlit run dashboard/app.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import RegimeShiftWarningSystem


NEON = {
    "violet": "#8A2BE2",
    "magenta": "#DF00FF",
    "purple": "#9400D3",
    "bg": "#0B0613",
    "panel": "#130A22",
    "text": "#F5EFFF",
    "muted": "#B39DDB",
    "grid": "rgba(223, 0, 255, 0.15)",
}


def apply_custom_theme() -> None:
    """Inject lightweight neon-violet styling."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: radial-gradient(circle at top left, #1A1030 0%, {NEON["bg"]} 55%);
                color: {NEON["text"]};
            }}
            [data-testid="stSidebar"] {{
                background-color: {NEON["panel"]};
                border-right: 1px solid {NEON["violet"]};
            }}
            [data-testid="metric-container"] {{
                background: linear-gradient(140deg, rgba(138,43,226,0.15), rgba(223,0,255,0.12));
                border: 1px solid rgba(223,0,255,0.35);
                border-radius: 12px;
                padding: 10px;
            }}
            .stButton button {{
                background: linear-gradient(90deg, {NEON["violet"]}, {NEON["magenta"]});
                color: white;
                border: none;
                border-radius: 8px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def run_pipeline(
    symbol: str,
    period_years: int,
    rolling_window: int,
    n_regimes: int,
    n_components: int,
    signal_threshold: float,
) -> Dict[str, Any]:
    """Run the quantitative pipeline and return aligned dashboard-ready data."""
    system = RegimeShiftWarningSystem(
        symbol=symbol.upper().strip(),
        period_years=period_years,
        rolling_window=rolling_window,
        n_regimes=n_regimes,
        n_components=n_components,
        signal_threshold=signal_threshold,
        output_dir="results",
    )
    run_result = system.run(verbose=False)
    if run_result.get("status") != "SUCCESS":
        raise RuntimeError(run_result.get("error", "Pipeline failed"))

    transitions = system.regime_detector.get_regime_transitions()
    min_len = min(
        len(system.dates),
        len(system.prices),
        len(system.regimes),
        len(system.instability_scores),
        len(system.signals),
    )
    if min_len == 0:
        raise RuntimeError("No analysis points available. Try a longer period or a more liquid symbol.")

    dates = pd.to_datetime(system.dates[-min_len:])
    prices = np.asarray(system.prices[-min_len:])
    regimes = np.asarray(system.regimes[-min_len:])
    instability = np.asarray(system.instability_scores[-min_len:])
    signals = np.asarray(system.signals[-min_len:])
    transitions = np.asarray(transitions)
    transitions = transitions[transitions < min_len]

    return {
        "dates": dates,
        "prices": prices,
        "regimes": regimes,
        "instability": instability,
        "signals": signals,
        "transitions": transitions,
        "metrics": system.evaluation_metrics or {},
    }


def base_layout(title: str, y_title: str) -> Dict[str, Any]:
    return {
        "template": "plotly_dark",
        "title": title,
        "xaxis_title": "Date",
        "yaxis_title": y_title,
        "paper_bgcolor": NEON["bg"],
        "plot_bgcolor": NEON["panel"],
        "font": {"color": NEON["text"]},
        "xaxis": {"gridcolor": NEON["grid"]},
        "yaxis": {"gridcolor": NEON["grid"]},
        "margin": {"l": 40, "r": 30, "t": 50, "b": 40},
    }


def add_transition_lines(fig: go.Figure, dates: pd.Series, transitions: np.ndarray) -> None:
    date_values = pd.to_datetime(np.asarray(dates))
    for idx in transitions:
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= len(date_values):
            continue
        transition_x = date_values[idx_int]
        fig.add_vline(
            x=transition_x,
            line_dash="dot",
            line_color=NEON["magenta"],
            line_width=1.2,
            opacity=0.7,
        )


def price_regime_figure(data: Dict[str, Any]) -> go.Figure:
    dates = data["dates"]
    prices = data["prices"]
    regimes = data["regimes"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name="Price",
            line={"color": "#EAD7FF", "width": 2},
        )
    )

    colors = [NEON["violet"], NEON["purple"], NEON["magenta"], "#6A0DAD", "#B026FF"]
    for regime in sorted(np.unique(regimes)):
        mask = regimes == regime
        fig.add_trace(
            go.Scatter(
                x=dates[mask],
                y=prices[mask],
                mode="markers",
                marker={"size": 4, "color": colors[int(regime) % len(colors)], "opacity": 0.85},
                name=f"Regime {regime}",
            )
        )

    add_transition_lines(fig, dates, data["transitions"])
    fig.update_layout(**base_layout("Price & Regime States", "Price"))
    return fig


def instability_figure(data: Dict[str, Any], threshold: float) -> go.Figure:
    dates = data["dates"]
    instability = data["instability"]
    norm = (instability - instability.min()) / (instability.max() - instability.min() + 1e-8)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=norm,
            mode="lines",
            name="Instability (normalized)",
            line={"color": NEON["violet"], "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(138, 43, 226, 0.12)",
        )
    )
    fig.add_hline(
        y=threshold,
        line_color=NEON["magenta"],
        line_width=2,
        line_dash="dash",
        annotation_text=f"Threshold {threshold:.2f}",
        annotation_position="top left",
    )
    add_transition_lines(fig, dates, data["transitions"])
    fig.update_layout(**base_layout("Instability Signal", "Normalized Instability"))
    return fig


def warning_signal_figure(data: Dict[str, Any]) -> go.Figure:
    dates = data["dates"]
    signals = data["signals"]
    active_idx = np.where(signals == 1)[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=signals,
            mode="lines",
            line={"shape": "hv", "color": NEON["purple"], "width": 2},
            name="Warning state",
        )
    )
    if len(active_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=dates[active_idx],
                y=signals[active_idx],
                mode="markers",
                marker={"size": 9, "symbol": "triangle-up", "color": NEON["magenta"]},
                name="Warning trigger",
            )
        )

    add_transition_lines(fig, dates, data["transitions"])
    fig.update_layout(**base_layout("Early Warning Activations", "Signal (0/1)"))
    fig.update_yaxes(range=[-0.05, 1.15], tickvals=[0, 1])
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Regime Shift Early Warning Dashboard",
        page_icon=":crystal_ball:",
        layout="wide",
    )
    apply_custom_theme()

    st.title("Regime Shift Early Warning Dashboard")
    st.caption("Interactive monitoring for instability, warning signals, and market regimes.")

    with st.sidebar:
        st.header("Pipeline Controls")
        symbol = st.text_input("Symbol", value="SPY")
        period_years = st.slider("Analysis Period (years)", min_value=2, max_value=20, value=10)
        rolling_window = st.slider("Rolling Window", min_value=20, max_value=180, value=60)
        n_regimes = st.slider("Regimes", min_value=2, max_value=6, value=3)
        n_components = st.slider("Components", min_value=2, max_value=6, value=3)
        signal_threshold = st.slider("Warning Threshold", 0.10, 0.90, 0.50, 0.05)
        run_now = st.button("Run Analysis", use_container_width=True)

    if not run_now:
        st.info("Set parameters in the sidebar and click 'Run Analysis'.")
        return

    with st.spinner("Running quantitative pipeline..."):
        try:
            data = run_pipeline(
                symbol=symbol,
                period_years=period_years,
                rolling_window=rolling_window,
                n_regimes=n_regimes,
                n_components=n_components,
                signal_threshold=signal_threshold,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    metrics = data["metrics"]
    cols = st.columns(4)
    cols[0].metric("Precision", f"{metrics.get('precision', 0.0):.2f}")
    cols[1].metric("Recall", f"{metrics.get('recall', 0.0):.2f}")
    cols[2].metric("F1 Score", f"{metrics.get('f1', 0.0):.2f}")
    cols[3].metric("Avg Lead Time", f"{metrics.get('avg_lead_time', 0.0):.1f} days")

    st.plotly_chart(price_regime_figure(data), use_container_width=True)
    st.plotly_chart(instability_figure(data, signal_threshold), use_container_width=True)
    st.plotly_chart(warning_signal_figure(data), use_container_width=True)


if __name__ == "__main__":
    main()
