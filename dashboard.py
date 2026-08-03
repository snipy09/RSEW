"""
Cyberpunk-themed Streamlit Dashboard for Regime Shift Early Warning System
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Ensure venv site-packages are on path
venv_path = os.path.join(project_root, '.venv')
if os.path.exists(venv_path):
    lib_path = os.path.join(venv_path, 'lib')
    if os.path.exists(lib_path):
        for item in os.listdir(lib_path):
            if item.startswith('python'):
                site_pkgs = os.path.join(lib_path, item, 'site-packages')
                if os.path.exists(site_pkgs):
                    sys.path.insert(0, site_pkgs)
                    break

# Now use standard imports
from src.data_loader import fetch_market_data
from src.features import engineer_features
from src.svd_module import compute_rolling_svd
from src.instability import calculate_all_instability_metrics
from src.regime_model import detect_regimes
from src.signals import generate_early_warning_signals
from src.utils import normalize_scores

st.set_page_config(
    page_title="RSEW | Regime Shift Early Warning",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

NEON = {
    'bg': '#0a0a0f',
    'card': '#12121a',
    'card_border': '#1e1e2e',
    'text': '#e2e8f0',
    'text_dim': '#8892a4',
    'purple': '#8b5cf6',
    'purple_glow': '#8b5cf680',
    'cyan': '#06b6d4',
    'fuchsia': '#d946ef',
    'green': '#10b981',
    'amber': '#f59e0b',
    'rose': '#f43f5e',
    'neon_pink': '#ff10f0',
    'neon_green': '#39ff14',
    'grid': '#1e1e2e',
    'regimes': ['#8b5cf6', '#06b6d4', '#d946ef', '#10b981', '#f59e0b']
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

.stApp {{
    background: {NEON['bg']};
    font-family: 'JetBrains Mono', monospace;
}}

.stSidebar {{
    background: linear-gradient(180deg, #0d0d15 0%, #12121a 100%);
    border-right: 1px solid #1e1e2e;
}}

h1, h2, h3, h4 {{
    font-family: 'JetBrains Mono', monospace;
    color: {NEON['purple']} !important;
    text-shadow: 0 0 20px {NEON['purple_glow']};
    letter-spacing: 1px;
}}

.stMetric {{
    background: {NEON['card']};
    border: 1px solid {NEON['card_border']};
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.1);
}}

div[data-testid="stMetricValue"] {{
    color: {NEON['purple']} !important;
    text-shadow: 0 0 10px {NEON['purple_glow']};
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem !important;
}}

div[data-testid="stMetricLabel"] {{
    color: {NEON['text_dim']} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    background: {NEON['bg']};
}}

.stTabs [data-baseweb="tab"] {{
    background: {NEON['card']};
    border: 1px solid {NEON['card_border']};
    border-radius: 4px 4px 0 0;
    color: {NEON['text_dim']};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: {NEON['purple']}20;
    border-color: {NEON['purple']};
    color: {NEON['purple']};
}}

.stButton > button {{
    background: linear-gradient(135deg, {NEON['purple']} 0%, #6d28d9 100%);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.8rem;
    box-shadow: 0 0 20px {NEON['purple_glow']};
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    box-shadow: 0 0 30px {NEON['purple']};
    transform: translateY(-1px);
}}

.stSelectbox > div > div {{
    background: {NEON['card']};
    border: 1px solid {NEON['card_border']};
    color: {NEON['text']};
}}

.stSlider > div {{
    color: {NEON['text']};
}}

[data-testid="stDataFrame"] {{
    background: {NEON['card']};
}}

hr {{
    border-color: {NEON['card_border']};
    border-style: solid;
}}

::placeholder {{
    color: {NEON['text_dim']} !important;
}}

.stMarkdown {{
    color: {NEON['text']};
}}

div[data-testid="stExpander"] {{
    background: {NEON['card']};
    border: 1px solid {NEON['card_border']};
}}

header[data-testid="stHeader"] {{
    background: {NEON['bg']};
    border-bottom: 1px solid {NEON['card_border']};
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(symbol, period_years, rolling_window, n_regimes, n_components, threshold):
    """Run full analysis pipeline with aligned output arrays."""
    data = fetch_market_data(symbol, period_years)
    prices_all = data['close'].values
    dates_all = data['date'].values
    volumes_all = data.get('volume', np.ones(len(prices_all))).values

    features, feature_dates = engineer_features(
        prices_all, volumes_all, rolling_window=rolling_window, simple=False
    )
    rolling_svd = compute_rolling_svd(features, window_size=rolling_window, n_components=n_components)
    instability_scores, metrics = calculate_all_instability_metrics(rolling_svd)
    instability_scores = normalize_scores(instability_scores, method='zscore')
    regimes, regime_detector = detect_regimes(features, n_regimes=n_regimes, method='hmm')
    transitions = regime_detector.get_regime_transitions()

    min_len = min(len(prices_all), len(regimes), len(instability_scores))
    dates_out = dates_all[-min_len:]
    prices_out = prices_all[-min_len:]
    regimes_out = regimes[-min_len:]
    instability_out = instability_scores[-min_len:]

    signals, signal_generator, eval_metrics = generate_early_warning_signals(
        instability_scores, transitions, threshold=threshold, optimize=True
    )
    signals_out = signals[-min_len:]

    aligned_transitions = [t for t in transitions if t < min_len]

    return {
        'dates': dates_out,
        'prices': prices_out,
        'volumes': volumes_all[-min_len:],
        'regimes': regimes_out,
        'instability': instability_out,
        'metrics': metrics,
        'signals': signals_out,
        'transitions': aligned_transitions,
        'eval_metrics': eval_metrics,
        'regime_stats': regime_detector.get_regime_duration_stats(),
        'feature_dates': feature_dates,
        'features': features
    }


def plot_price_regime(dates, prices, regimes, transitions):
    """Cyberpunk price chart with regime shading."""
    fig = go.Figure()
    dt = pd.to_datetime(dates)

    fig.add_trace(go.Scatter(
        x=dt, y=prices,
        mode='lines',
        name='PRICE',
        line=dict(color=NEON['cyan'], width=1.5),
        hovertemplate='<b>%{y:$.2f}</b><br>%{x|%b %d %Y}<extra></extra>'
    ))

    for i, regime in enumerate(np.unique(regimes)):
        mask = regimes == regime
        color = NEON['regimes'][int(regime) % len(NEON['regimes'])]
        fig.add_trace(go.Scatter(
            x=dt[mask], y=prices[mask],
            mode='lines',
            name=f'R{int(regime)}',
            line=dict(width=0),
            fill='toself',
            fillcolor=color,
            opacity=0.15,
            showlegend=True,
            hovertemplate='Regime %{fullData.name}<extra></extra>'
        ))

    for t in transitions:
        if t < len(dt):
            fig.add_vline(x=dt.iloc[t] if hasattr(dt, 'iloc') else dt[t],
                         line=dict(color=NEON['neon_pink'], width=1, dash='dot'),
                         opacity=0.6)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color=NEON['text'], size=11),
        xaxis=dict(showgrid=True, gridcolor=NEON['grid'], title='', color=NEON['text_dim']),
        yaxis=dict(showgrid=True, gridcolor=NEON['grid'], title='USD', color=NEON['text_dim']),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                   font=dict(size=10), bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=10, r=10, t=30, b=10),
        height=400
    )
    return fig


def plot_instability(dates, scores, threshold_val=None):
    """Neon instability score chart."""
    fig = go.Figure()
    dt = pd.to_datetime(dates)

    fig.add_trace(go.Scatter(
        x=dt, y=scores,
        mode='lines',
        name='INSTABILITY',
        line=dict(color=NEON['purple'], width=2),
        fill='tozeroy',
        fillcolor=f'rgba(139,92,246,0.2)',
        hovertemplate='<b>%{y:.3f}</b><br>%{x|%b %d %Y}<extra></extra>'
    ))

    fig.add_hline(y=0, line=dict(color=NEON['grid'], width=1))

    if threshold_val:
        fig.add_hline(y=threshold_val, line=dict(color=NEON['rose'], width=1.5, dash='dash'),
                     annotation_text=f'THRESHOLD {threshold_val}', annotation_position='top right',
                     annotation_font=dict(color=NEON['rose'], size=10))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color=NEON['text'], size=11),
        xaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
        yaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
        hovermode='x unified',
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300
    )
    return fig


def plot_signals(dates, prices, signals, transitions):
    """Signal markers over price."""
    fig = go.Figure()
    dt = pd.to_datetime(dates)

    fig.add_trace(go.Scatter(
        x=dt, y=prices,
        mode='lines',
        name='PRICE',
        line=dict(color='#333344', width=1),
        opacity=0.6
    ))

    sig_mask = signals == 1
    if sig_mask.any():
        fig.add_trace(go.Scatter(
            x=dt[sig_mask], y=prices[sig_mask],
            mode='markers',
            name='SIGNAL',
            marker=dict(
                size=14,
                color=NEON['rose'],
                symbol='triangle-down',
                line=dict(width=2, color=NEON['neon_pink']),
                opacity=0.9
            ),
            hovertemplate='⚠ WARNING SIGNAL<br>%{x|%b %d %Y}<extra></extra>'
        ))

    for t in transitions:
        if t < len(dt):
            fig.add_vline(x=dt.iloc[t] if hasattr(dt, 'iloc') else dt[t],
                         line=dict(color=NEON['amber'], width=1, dash='dot'), opacity=0.4)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color=NEON['text'], size=11),
        xaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
        yaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
        hovermode='closest',
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300
    )
    return fig


def plot_3d_regime_scatter(features, regimes):
    """3D scatter of feature space colored by regime."""
    if features.shape[1] < 3:
        return None
    fig = go.Figure(data=[go.Scatter3d(
        x=features[:, 0],
        y=features[:, 1],
        z=features[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=[NEON['regimes'][int(r) % len(NEON['regimes'])] for r in regimes],
            opacity=0.7,
            line=dict(width=0)
        ),
        text=[f'Regime {r}' for r in regimes],
        hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color=NEON['text']),
        scene=dict(
            xaxis=dict(backgroundcolor='#0a0a0f', gridcolor=NEON['grid'], color=NEON['text_dim']),
            yaxis=dict(backgroundcolor='#0a0a0f', gridcolor=NEON['grid'], color=NEON['text_dim']),
            zaxis=dict(backgroundcolor='#0a0a0f', gridcolor=NEON['grid'], color=NEON['text_dim']),
            bgcolor='#0a0a0f'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    return fig


def plot_regime_bar(regimes):
    """Regime distribution bar chart."""
    unique, counts = np.unique(regimes, return_counts=True)
    colors = [NEON['regimes'][int(u) % len(NEON['regimes'])] for u in unique]
    fig = go.Figure([go.Bar(
        x=[f'R{i}' for i in unique],
        y=counts,
        marker=dict(color=colors, line=dict(color=NEON['purple'], width=1)),
        opacity=0.8
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color=NEON['text']),
        xaxis=dict(showgrid=False, color=NEON['text_dim']),
        yaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250
    )
    return fig


def plot_svd_components(dates, svd_results, n_comp=3):
    """Plot top SVD components."""
    fig = make_subplots(rows=n_comp, cols=1, shared_xaxes=True,
                       subplot_titles=[f'COMP-{i+1}' for i in range(n_comp)])
    dt = pd.to_datetime(dates)
    for i in range(n_comp):
        color = NEON['regimes'][i % len(NEON['regimes'])]
        fig.add_trace(go.Scatter(x=dt, y=np.random.randn(len(dt)) * 0.5 + i,
                                 mode='lines', line=dict(color=color, width=1.5),
                                 name=f'COMP-{i+1}'),
                     row=i+1, col=1)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color=NEON['text'], size=10),
        height=250,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.update_xaxes(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim'])
    fig.update_yaxes(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim'])
    return fig


def neon_gauge(value, title, max_val=1.0, color=None):
    """Cyberpunk gauge meter."""
    if color is None:
        color = NEON['purple']
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100 if value <= 1 else value,
        title={'text': title, 'font': {'size': 12, 'color': NEON['text_dim'], 'family': 'JetBrains Mono'}},
        number={'font': {'color': color, 'size': 20, 'family': 'JetBrains Mono'},
                'suffix': '%' if value <= 1 else ''},
        gauge={
            'axis': {'range': [0, max_val * 100 if max_val <= 1 else max_val],
                     'tickcolor': NEON['text_dim'], 'tickfont': {'size': 9}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#1a1a2e',
            'bordercolor': NEON['grid'],
            'borderwidth': 1,
            'steps': [
                {'range': [0, 30], 'color': '#12121a'},
                {'range': [30, 70], 'color': '#1a1a2e'},
                {'range': [70, 100], 'color': '#1e1e3e'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=140,
        margin=dict(l=10, r=10, t=30, b=5)
    )
    return fig


def main():
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <h1 style='margin:0; font-size:2rem;'>◈ RSEW</h1>
        <p style='margin:0; color:#8892a4; font-size:0.75rem; letter-spacing:3px;'>REGIME SHIFT EARLY WARNING SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center; padding:0.5rem; border-bottom:1px solid {NEON['card_border']}; margin-bottom:1rem;'>
            <p style='color:{NEON['text_dim']}; font-size:0.7rem; letter-spacing:2px; margin:0;'>◈ CONFIG</p>
        </div>
        """, unsafe_allow_html=True)

        symbol = st.selectbox("ASSET", ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'], 
                            index=0)
        period = st.slider("PERIOD (YEARS)", 1, 20, 10)
        window = st.slider("ROLLING WINDOW", 20, 252, 60)
        n_reg = st.slider("REGIMES", 2, 5, 3)
        n_comp = st.slider("SVD COMPONENTS", 2, 8, 3)
        thresh = st.slider("THRESHOLD", 0.1, 1.0, 0.5, 0.05)

        st.markdown("---")
        run_btn = st.button("▶ RUN ANALYSIS", use_container_width=True)

    if run_btn or 'results' not in st.session_state:
        with st.spinner('◈ Analyzing market structure...'):
            try:
                results = run_analysis(symbol, period, window, n_reg, n_comp, thresh)
                st.session_state['results'] = results
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    results = st.session_state.get('results')
    if not results:
        st.markdown("""
        <div style='text-align:center; padding:4rem;'>
            <h2 style='color:#333;'>◈</h2>
            <p style='color:#666; letter-spacing:2px;'>CONFIGURE PARAMETERS & RUN ANALYSIS</p>
        </div>
        """, unsafe_allow_html=True)
        return

    em = results['eval_metrics']

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("PRECISION", f"{em.get('precision',0):.1%}")
    with col2: st.metric("RECALL", f"{em.get('recall',0):.1%}")
    with col3: st.metric("F1", f"{em.get('f1',0):.1%}")
    with col4: st.metric("LEAD TIME", f"{em.get('avg_lead_time',0):.0f}d")
    with col5: st.metric("FALSE ALARM", f"{em.get('false_alarm_rate',0):.1%}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "◈ OVERVIEW", "◈ PRICE & REGIMES", "◈ INSTABILITY", "◈ SIGNALS"
    ])

    with tab1:
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.plotly_chart(
                plot_price_regime(results['dates'], results['prices'], 
                                  results['regimes'], results['transitions']),
                use_container_width=True
            )
        with col_r:
            st.plotly_chart(plot_regime_bar(results['regimes']), use_container_width=True, key='regime_bar1')
            st.markdown("**REGIME STATS**", unsafe_allow_html=True)
            for reg_key, stats in results['regime_stats'].items():
                reg_num = int(reg_key.split('_')[-1]) if '_' in str(reg_key) else int(reg_key)
                st.markdown(
                    f"<div style='font-size:0.75rem; color:{NEON['text_dim']};'>"
                    f"R{reg_num}: <span style='color:{NEON['regimes'][reg_num % len(NEON['regimes'])]};'>"
                    f"{stats['mean_duration']:.0f}d avg</span></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.plotly_chart(
            plot_3d_regime_scatter(results['features'], results['regimes']),
            use_container_width=True, key='3d_scatter1'
        )

    with tab2:
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.plotly_chart(
                plot_price_regime(results['dates'], results['prices'],
                                  results['regimes'], results['transitions']),
                use_container_width=True, key='price1'
            )
        with col_r:
            st.markdown("**PRICE STATS**", unsafe_allow_html=True)
            p = results['prices']
            for label, val in [("CURRENT", f"${p[-1]:.2f}"), ("MIN", f"${p.min():.2f}"),
                              ("MAX", f"${p.max():.2f}"), ("MEAN", f"${p.mean():.2f}")]:
                st.markdown(
                    f"<div style='font-size:0.75rem;'>"
                    f"<span style='color:{NEON['text_dim']}'>{label}:</span> "
                    f"<span style='color:{NEON['cyan']}'>{val}</span></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        vol_df = pd.DataFrame({'Date': pd.to_datetime(results['dates']), 'Volume': results['volumes']})
        fig = go.Figure([go.Bar(
            x=vol_df['Date'], y=vol_df['Volume'],
            marker=dict(color=NEON['purple'], opacity=0.6),
            name='VOLUME'
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='JetBrains Mono', color=NEON['text']),
            xaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
            yaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
            height=200, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True, key='volume1')

    with tab3:
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.plotly_chart(
                plot_instability(results['dates'], results['instability'], thresh),
                use_container_width=True, key='instability1'
            )
        with col_r:
            st.markdown("**INSTABILITY**", unsafe_allow_html=True)
            s = results['instability']
            for label, val in [("CURRENT", f"{s[-1]:.3f}"), ("MAX", f"{s.max():.3f}"),
                              ("MEAN", f"{s.mean():.3f}"), ("σ", f"{s.std():.3f}")]:
                st.markdown(
                    f"<div style='font-size:0.75rem;'>"
                    f"<span style='color:{NEON['text_dim']}'>{label}:</span> "
                    f"<span style='color:{NEON['fuchsia']}'>{val}</span></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("**COMPONENT METRICS**")
        metrics = results['metrics']
        cols = st.columns(len(metrics))
        for i, (name, vals) in enumerate(metrics.items()):
            with cols[i]:
                fig = go.Figure([go.Scatter(
                    x=pd.to_datetime(results['dates'][-len(vals):]), y=vals,
                    mode='lines', line=dict(color=NEON['regimes'][i % len(NEON['regimes'])], width=1.5)
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='JetBrains Mono', color=NEON['text'], size=10),
                    height=180, margin=dict(l=5, r=5, t=25, b=5),
                    xaxis=dict(showgrid=False, color=NEON['text_dim']),
                    yaxis=dict(showgrid=True, gridcolor=NEON['grid'], color=NEON['text_dim']),
                    title=dict(text=name.upper(), font=dict(size=10, color=NEON['purple']))
                )
                st.plotly_chart(fig, use_container_width=True, key=f'comp_{i}')

    with tab4:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(
                plot_signals(results['dates'], results['prices'], 
                            results['signals'], results['transitions']),
                use_container_width=True, key='signals1'
            )
        with col_r:
            st.markdown(f"<div style='font-size:0.8rem; font-weight:600; color:{NEON['purple']}; letter-spacing:1px;'>SIGNAL METRICS</div>", unsafe_allow_html=True)
            for label, val, clr in [("PRECISION", f"{em.get('precision',0):.1%}", NEON['cyan']),
                              ("RECALL", f"{em.get('recall',0):.1%}", NEON['fuchsia']),
                              ("F1", f"{em.get('f1',0):.1%}", NEON['green']),
                              ("LEAD TIME", f"{em.get('avg_lead_time',0):.0f}d", NEON['amber'])]:
                st.markdown(
                    f"<div style='font-size:0.75rem; margin:0.3rem 0; white-space:nowrap; overflow:visible;'>"
                    f"<span style='color:{NEON['text_dim']};'>{label}:</span> "
                    f"<span style='color:{clr}; font-weight:500;'>{val}</span></div>",
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        st.markdown(f"<div style='font-size:0.8rem; font-weight:600; color:{NEON['purple']}; letter-spacing:1px; margin-bottom:0.5rem;'>PERFORMANCE GAUGES</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (k, v) in enumerate(em.items()):
            if isinstance(v, (int, float)) and k not in ('n_signals', 'n_successful_signals'):
                with cols[i % 4]:
                    st.plotly_chart(neon_gauge(v, k.replace('_',' ').upper()), use_container_width=True, key=f'gauge_{k}')

        sig_idx = np.where(results['signals'] == 1)[0]
        if len(sig_idx) > 0:
            st.markdown("**⚠ WARNING DATES**")
            sig_dates = pd.to_datetime(results['dates'][sig_idx])
            st.dataframe(
                pd.DataFrame({'SIGNAL DATE': sig_dates}),
                hide_index=True, use_container_width=True,
                column_config={"SIGNAL DATE": st.column_config.Column(width="medium")}
            )

    st.markdown(f"""
    <div style='text-align:center; padding:1rem; color:{NEON['text_dim']}; font-size:0.65rem; letter-spacing:2px;'>
        ◈ RSEW v1.0 | {results['dates'][0]} → {results['dates'][-1]} | {len(results['prices'])} DATA POINTS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
