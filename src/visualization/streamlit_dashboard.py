import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from twelvedata import TDClient
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from groq import Groq
from statsmodels.tsa.arima.model import ARIMA
import time

# -- Configuration and Initialization --
load_dotenv()
td_client = None
groq_client = None

POPULAR_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BABA', 'TSM',
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'BTC/USD', 'ETH/USD', 'XRP/USD', 'ADA/USD',
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD'
]

# -- Enhanced Custom Styling --
def apply_enhanced_styling():
    st.markdown("""
<style>
/* Fix for Sidebar Toggle Button */
[data-testid="stSidebarCollapseButton"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 2rem !important;
    z-index: 10000 !important;
    width: 2rem !important;
    height: 2rem !important;
    transform: none !important;
    transition: left 0.3s ease !important;
}

/* Adjust button position based on sidebar state */
.stApp:not([data-collapsed]) [data-testid="stSidebarCollapseButton"] {
    left: 16rem !important; /* Adjusted for expanded sidebar width */
}

.stApp[data-collapsed="true"] [data-testid="stSidebarCollapseButton"] {
    left: 0 !important; /* Flush with edge when collapsed */
}

/* Handle full-screen mode */
.stApp[style*="width: 100vw"] [data-testid="stSidebarCollapseButton"],
.stApp[style*="width:100vw"] [data-testid="stSidebarCollapseButton"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    left: 0 !important;
    top: 2rem !important;
    z-index: 10000 !important;
}

/* Style the button */
[data-testid="stSidebarCollapseButton"] button {
    background-color: rgba(102, 126, 234, 0.9) !important;
    color: white !important;
    border: none !important;
    border-radius: 0 50% 50% 0 !important;
    width: 2rem !important;
    height: 2rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}

/* Hover effect */
[data-testid="stSidebarCollapseButton"] button:hover {
    background-color: #667eea !important;
    transform: scale(1.1) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
}

/* Hide default Streamlit icon */
[data-testid="stSidebarCollapseButton"] button span[data-testid="stIconMaterial"] {
    display: none !important;
}

/* Add custom arrow for expanded state (left arrow to collapse) */
.stApp:not([data-collapsed]) [data-testid="stSidebarCollapseButton"] button::before {
    content: '←' !important;
    font-size: 1.2rem !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
}

/* Add custom arrow for collapsed state (right arrow to expand) */
.stApp[data-collapsed="true"] [data-testid="stSidebarCollapseButton"] button::before {
    content: '→' !important;
    font-size: 1.2rem !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
}

/* Ensure parent containers don't hide the button */
[data-testid="stSidebar"],
.css-1d391kg,
.css-1cypcdb,
.stSidebar {
    overflow: visible !important;
    position: relative !important;
}

/* Enhance button appearance when sidebar is collapsed */
.stApp[data-collapsed="true"] [data-testid="stSidebarCollapseButton"] button {
    background-color: #667eea !important;
    box-shadow: 0 0 0 2px white, 0 0 0 4px #667eea, 0 4px 15px rgba(0,0,0,0.3) !important;
}

/* Prevent main content from overlapping with toggle button */
.main .block-container {
    padding-left: 2.5rem !important;
}

/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Custom Font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header Styles */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.main-header h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
}

/* Enhanced Metric Cards */
.metric-card {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 0.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}

.metric-title {
    font-size: 0.9rem;
    font-weight: 500;
    color: #687280;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.metric-change {
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.25rem 0.5rem;
    border-radius: 20px;
    display: inline-block;
}

.positive { color: #108981; background: rgba(16,185,129,0.1); }
.negative { color: #EF4444; background: rgba(239,68,68,0.1); }
.neutral { color: #6B7280; background: rgba(107,114,128,0.1); }

/* Old metric container styles for compatibility */
.metric-container {
    background: #F8FAFC;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    text-align: center;
}

.metric-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.metric-label {
    font-size: 0.9rem;
    color: #687280;
    font-weight: 500;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 600;
    color: #1E3A8A;
    margin: 0.3rem 0;
}

.indicator-positive { color: #22C55E; }
.indicator-negative { color: #EF4444; }
.indicator-neutral { color: #687280; }

/* Status Badges */
.status-badge {
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: inline-block;
    margin: 0.25rem;
}

.status-connected { background: #108981; color: white; }
.status-disconnected { background: #EF4444; color: white; }
.status-loading { background: #F59E0B; color: white; }

/* Alert Boxes */
.alert {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid;
}

.alert-success {
    background: rgba(16,185,129,0.1);
    border-left-color: #108981;
    color: #065446;
}

.alert-warning {
    background: rgba(245,158,11,0.1);
    border-left-color: #F59E0B;
    color: #92400e;
}

.alert-error {
    background: rgba(239,68,68,0.1);
    border-left-color: #EF4444;
    color: #991b1b;
}

/* Chart Container */
.chart-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Hide default Streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #667eea;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #5a67d8;
}
</style>
<script>
// Ensure sidebar toggle button remains visible and functional
document.addEventListener('DOMContentLoaded', function() {
    const ensureToggleButtonVisible = () => {
        const toggleButton = document.querySelector('[data-testid="stSidebarCollapseButton"]');
        const app = document.querySelector('.stApp');
        if (toggleButton && app) {
            // Force visibility and positioning
            toggleButton.style.display = 'block';
            toggleButton.style.visibility = 'visible';
            toggleButton.style.opacity = '1';
            toggleButton.style.position = 'fixed';
            toggleButton.style.top = '2rem';
            toggleButton.style.zIndex = '10000';
            toggleButton.style.transform = 'none';
            // Dynamically adjust left position based on sidebar state
            toggleButton.style.left = app.dataset.collapsed === 'true' ? '0px' : '16rem';
            // Ensure parent doesn't hide it
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.overflow = 'visible';
            }
        } else {
            console.warn('Toggle button or .stApp not found');
        }
    };

    // Run initially
    ensureToggleButtonVisible();

    // Run on click events
    document.addEventListener('click', function(e) {
        if (e.target.closest('[data-testid="stSidebarCollapseButton"]')) {
            setTimeout(ensureToggleButtonVisible, 100); // Reduced delay for faster response
        }
    });

    // Observe DOM changes on stApp and stSidebar
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList' || mutation.type === 'attributes') {
                ensureToggleButtonVisible();
            }
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-collapsed', 'style', 'class']
    });

    // More frequent periodic check for Streamlit reruns and full-screen mode
    setInterval(ensureToggleButtonVisible, 500);
});
</script>
""", unsafe_allow_html=True)

# -- Technical Indicators Functions --
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    return atr

# -- AI and Data Interpretation Functions --
@st.cache_data(ttl=3600)
def get_stock_insights(query, model="llama3-8b-8192"):
    groq_client_instance = st.session_state.get('groq_client')
    if groq_client_instance is None:
        return "Error: Groq API client not initialized. Please enter your API key."
    try:
        chat_completion = groq_client_instance.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise financial analyst. Provide clear, insightful analysis based on the quantitative market data provided. Use markdown for formatting."
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error fetching AI insights from Groq: {e}")
        return "Could not generate AI insights due to an error."

def interpret_change(change_val):
    if change_val > 10: return "Strong bullish momentum"
    if change_val > 5: return "Positive performance"
    if change_val > 0: return "Slight positive movement"
    if change_val > -5: return "Slight weakness"
    return "Significant downward pressure"

def interpret_volatility(vol_val):
    return "High" if vol_val > 3 else "Moderate" if 1.5 < vol_val <= 3 else "Low"

def interpret_rsi(rsi_val):
    if rsi_val > 70: return "Overbought"
    elif rsi_val < 30: return "Oversold"
    else: return "Neutral"

# -- UI Display Components --
def display_metric(label, value, delta=None):
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-delta">{delta}</div>' if delta else ""}
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, change=None, change_type="neutral", icon="📈"):
    change_class = f'metric-change {change_type}'
    change_html = f'<div class="{change_class}">{change}</div>' if change else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

def create_technical_analysis_chart(data, ticker, indicators=None):
    if indicators is None:
        indicators = ['SMA20', 'SMA50', 'EMA20', 'BB', 'RSI', 'MACD', 'Stoch', 'ATR']
    
    rows = 4
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f'{ticker} - Price & Moving Averages',
            'RSI (Relative Strength Index)',
            'MACD',
            'Stochastic Oscillator'
        ],
        row_heights=[0.5, 0.2, 0.2, 0.15]
    )
    
    sma20 = calculate_sma(data['close'], 20)
    sma50 = calculate_sma(data['close'], 50)
    ema20 = calculate_ema(data['close'], 20)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data['close'])
    rsi = calculate_rsi(data['close'])
    macd, signal, histogram = calculate_macd(data['close'])
    stoch_k, stoch_d = calculate_stochastic(data['high'], data['low'], data['close'])
    atr = calculate_atr(data['high'], data['low'], data['close'])
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ), row=1, col=1)
    
    if 'SMA20' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=sma20, name='SMA 20', 
                                line=dict(color='orange')), row=1, col=1)
    if 'SMA50' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=sma50, name='SMA 50', 
                                line=dict(color='red')), row=1, col=1)
    if 'EMA20' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=ema20, name='EMA 20', 
                                line=dict(color='purple')), row=1, col=1)
    
    if 'BB' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=upper_bb, name='BB Upper', 
                                line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=lower_bb, name='BB Lower', 
                                line=dict(color='gray', dash='dash')), row=1, col=1)
    
    if 'RSI' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', 
                                line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    if 'MACD' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', 
                                line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=signal, name='Signal', 
                                line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', 
                             marker_color='gray'), row=3, col=1)
    
    if 'Stoch' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=stoch_k, name='%k', 
                                line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=stoch_d, name='%D', 
                                line=dict(color='red')), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.update_layout(
        title=f"{ticker} - Complete Technical Analysis",
        xaxis_title='Date',
        height=800,
        template='plotly_white',
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Price ($")", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stochastic", row=4, col=1)
    
    return fig

def create_prediction_chart(historical_data, future_dates, future_prices, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['close'], 
                             name='Historical', line=dict(color='#3882F6')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, 
                             name='Prediction', line=dict(color='#FF4444', dash='dash')))
    fig.update_layout(
        title=f"{ticker.upper()} Price Prediction",
        xaxis_title='Date', yaxis_title='Price ($)',
        template='plotly_white', height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -- Enhanced Dashboard Components --
def create_dashboard_header():
    st.markdown("""
    <div class="main-header">
        <h1>Real-Time Market Overview & Analytics</h1>
        <p>Stock Market Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

def fetch_market_data(td_client):
    if not td_client:
        return None
    
    symbols = {
        "S&P 500": "SPX",
        "NASDAQ": "IXIC",
        "DOW": "DJI",
        "VIX": "VIX",
        "Gold": "XAU/USD",
        "Silver": "XAG/USD",
        "Oil": "CL/USD",
        "Bitcoin": "BTC/USD",
        "Ethereum": "ETH/USD"
    }
    
    market_data = {}
    try:
        for name, symbol in symbols.items():
            quote = td_client.quote(symbol=symbol).as_json()
            if quote and 'close' in quote and 'change' in quote and 'percent_change' in quote:
                value = float(quote['close'])
                change = float(quote['percent_change'])
                
                # Determine type
                change_type = "positive" if change >= 0 else "negative"
                if name == "VIX":  # VIX has an inverse relationship
                    change_type = "negative" if change >= 0 else "positive"

                # Format values
                formatted_value = f"${value:,.2f}" if name not in ["S&P 500", "NASDAQ", "DOW", "VIX"] else f"{value:,.2f}"
                formatted_change = f"{change:+.2f}%"
                
                market_data[name] = {
                    "value": formatted_value,
                    "change": formatted_change,
                    "type": change_type
                }
            else:
                market_data[name] = {"value": "N/A", "change": "N/A", "type": "neutral"}
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None
        
    return market_data

def create_market_overview():
    st.markdown("### Market Overview")
    td_client_instance = st.session_state.get('td_client')
    
    if not td_client_instance:
        st.warning("API client not connected. Please connect in the sidebar to see live data.")
        # Display some placeholder data
        market_data = {
            "S&P 500": {"value": "Loading...", "change": "", "type": "neutral"},
            "NASDAQ": {"value": "Loading...", "change": "", "type": "neutral"},
            "DOW": {"value": "Loading...", "change": "", "type": "neutral"},
            "VIX": {"value": "Loading...", "change": "", "type": "neutral"},
            "Gold": {"value": "Loading...", "change": "", "type": "neutral"},
            "Oil": {"value": "Loading...", "change": "", "type": "neutral"}
        }
    else:
        market_data = fetch_market_data(td_client_instance)

    if not market_data:
        st.info("Market data is currently unavailable.")
        return

    cols = st.columns(3)
    for i, (name, data) in enumerate(market_data.items()):
        with cols[i % len(cols)]:
            create_metric_card(
                name,
                data["value"],
                data["change"],
                data["type"],
                "📈" if data["type"] == "positive" else "📉"
            )

def create_quick_stock_analysis():
    st.markdown("### Quick Stock Analysis")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_symbol = st.text_input("Enter Stock Symbol", "AAPL", placeholder="e.g., AAPL, GOOGL, TSLA")
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1D", "1M", "3M", "6M", "1Y"])
    
    with col3:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area"])
    
    if st.button("Quick Analysis", type="primary"):
        td_client_instance = st.session_state.get('td_client')
        if td_client_instance is None:
            st.error("Please connect to APIs first in the sidebar.")
            return
        
        with st.spinner("Analyzing stock data..."):
            try:
                ts = td_client_instance.time_series(symbol=stock_symbol, interval="1day", outputsize=90).as_pandas().sort_index(ascending=True)
                fig = go.Figure()
                
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=ts.index,
                        open=ts['open'], 
                        high=ts['high'], 
                        low=ts['low'], 
                        close=ts['close'], 
                        name=stock_symbol
                    ))
                elif chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=ts.index,
                        y=ts['close'], 
                        mode='lines', 
                        name=stock_symbol, 
                        line=dict(color='#667eea', width=3)
                    ))
                else: # Area
                    fig.add_trace(go.Scatter(
                        x=ts.index,
                        y=ts['close'], 
                        mode='lines', 
                        name=stock_symbol, 
                        fill='tonexty', 
                        line=dict(color='#667eea', width=2)
                    ))
                
                fig.update_layout(
                    title=f"{stock_symbol} - {timeframe} Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price ($")",
                    template="plotly_white",
                    height=400,
                    showlegend=True,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                current_price = ts['close'].iloc[-1]
                price_change = ts['close'].iloc[-1] - ts['close'].iloc[-2]
                price_change_pct = (price_change / ts['close'].iloc[-2]) * 100
                
                cols = st.columns(4)
                with cols[0]:
                    create_metric_card("Current Price", f"${current_price:.2f}", 
                                      f"{price_change_pct:.2f}%", 
                                      "positive" if price_change > 0 else "negative", 
                                      "📈" if price_change > 0 else "📉")
                with cols[1]:
                    create_metric_card("Day High", f"${ts['high'].iloc[-1]:.2f}", 
                                      "High", "positive", "⬆")
                with cols[2]:
                    create_metric_card("Day Low", f"${ts['low'].iloc[-1]:.2f}", 
                                      "Low", "negative", "⬇")
                with cols[3]:
                    volume = ts.get('volume', pd.Series([0])).iloc[-1] if 'volume' in ts.columns else 0
                    create_metric_card("Volume", f"{volume:,.0f}", 
                                      "Volume", "neutral", "📊")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

def create_news_alerts_section():
    st.markdown("### Market News & Alerts")
    news_items = [
        {
            "title": "📈 Tech Stocks Rally on AI Optimism",
            "time": "2 hours ago",
            "summary": "Major technology companies see significant gains as AI developments continue to drive investor confidence.",
            "impact": "positive"
        },
        {
            "title": "📊 Fed Hints at Rate Stability",
            "time": "4 hours ago",
            "summary": "Federal Reserve officials suggest interest rates may remain stable in upcoming meetings.",
            "impact": "neutral"
        },
        {
            "title": "📉 Energy Sector Faces Headwinds",
            "time": "6 hours ago",
            "summary": "Oil prices decline amid global supply concerns and demand uncertainty.",
            "impact": "negative"
        }
    ]
    
    for news in news_items:
        impact_class = "alert-success" if news['impact'] == 'positive' else "alert-warning" if news['impact'] == 'neutral' else "alert-error"
        st.markdown(f"""
        <div class="alert {impact_class}">
            <strong>{news['title']}</strong><br>
            <small>{news['time']}</small><br>
            {news['summary']}
        </div>
        """, unsafe_allow_html=True)

# -- Page/Feature Handlers --
def handle_dashboard_overview():
    create_dashboard_header()
    
    col1, col2 = st.columns([3, 1])
    with col2:
        td_client_instance = st.session_state.get('td_client')
        groq_client_instance = st.session_state.get('groq_client')
        
        if td_client_instance and groq_client_instance:
            st.markdown('<div class="status-badge status-connected">🔌 APIs Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-disconnected">🔌 APIs Disconnected</div>', unsafe_allow_html=True)
    
    create_market_overview()
    create_quick_stock_analysis()
    create_news_alerts_section()

def handle_ai_analysis(td_client_instance, groq_client_instance):
    st.header("🤖 AI-Powered Stock Analysis")
    ticker = st.text_input("Enter Stock Ticker for AI Analysis", "AAPL").upper()
    
    if st.button("Generate Analysis"):
        if not ticker:
            st.warning("Please enter a stock ticker.")
            return
        
        if td_client_instance is None:
            st.error("Twelve Data API Key is not provided. Please connect to APIs in the sidebar.")
            return
        
        if groq_client_instance is None:
            st.error("Groq API Key is not provided. Please connect to APIs in the sidebar.")
            return
        
        with st.spinner(f"Fetching data and generating analysis for {ticker}..."):
            try:
                ts = td_client_instance.time_series(symbol=ticker, interval="1day", outputsize=90).as_pandas().sort_index(ascending=True)
            except Exception as e:
                st.error(f"Failed to fetch time series data for {ticker} from Twelve Data: {e}")
                return
            
            latest_price = ts['close'].iloc[-1]
            change_30d = (latest_price / ts['close'].iloc[-30] - 1) * 100 if len(ts) >= 30 else 0
            volatility = ts['close'].pct_change().rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100 if len(ts) >= 30 else 0
            
            query = f"""
            Generate a concise stock analysis for {ticker} based on the following market data.
            - **Current Price:** ${latest_price:.2f}
            - **30-Day Change:** {change_30d:.2f}% ({interpret_change(change_30d)})
            - **Annualized Volatility:** {volatility:.2f}% ({interpret_volatility(volatility)})
            
            Provide a summary, a bullish case based on the data, a bearish case based on the data, and a final outlook.
            """
            
            st.subheader(f"Analysis for {ticker}")
            ai_insights = get_stock_insights(query)
            st.markdown(ai_insights)

def handle_prediction(td_client_instance, groq_client_instance):
    st.header("🔮 Price Prediction")
    ticker = st.text_input("Enter Ticker for Prediction", "TSLA").upper()
    days_to_predict = st.slider("Days to Predict", 7, 90, 30)
    
    if st.button("Predict Future Price"):
        if td_client_instance is None:
            st.error("Twelve Data API Key is not provided. Please connect to APIs in the sidebar.")
            return
        
        if groq_client_instance is None:
            st.error("Groq API Key is not provided. Please connect to APIs in the sidebar.")
            return
        
        with st.spinner(f"Running prediction model for {ticker}..."):
            try:
                hist_data = td_client_instance.time_series(symbol=ticker, interval="1day", outputsize=200).as_pandas().sort_index(ascending=True)
                model = ARIMA(hist_data['close'], order=(5,1,0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=days_to_predict)
                future_dates = pd.to_datetime(pd.date_range(start=hist_data.index[-1] + timedelta(days=1), periods=days_to_predict))
                future_prices = forecast.values
                
                st.plotly_chart(create_prediction_chart(hist_data, future_dates, future_prices, ticker), use_container_width=True)
                
                price_change_pct = ((future_prices[-1] / hist_data['close'].iloc[-1] - 1) * 100)
                prediction_query = (
                    f"Analyze the {days_to_predict}-day price prediction for {ticker}:"
                    f"\n- Current price: ${hist_data['close'].iloc[-1]:.2f}"
                    f"\n- Predicted price in {days_to_predict} days: ${future_prices[-1]:.2f} ({price_change_pct:+.2f}%)"
                    f"\n- Discuss potential factors, risks, and opportunities for this prediction based on time-series forecasting."
                )
                
                insights = get_stock_insights(prediction_query)
                st.subheader("AI Interpretation of Prediction")
                st.markdown(insights)
                
            except Exception as e:
                st.error(f"Could not generate prediction: {e}")

def handle_correlation(td_client_instance):
    st.header("📈 Asset Correlation Matrix")
    available_assets = ['SPY', 'QQQ', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BTC/USD', 'ETH/USD']
    selected_assets = st.multiselect(
        "Select at least 2 assets for correlation analysis",
        available_assets,
        default=['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    )
    
    if len(selected_assets) < 2:
        st.warning("Please select at least two assets.")
        return
    
    if td_client_instance is None:
        st.error("Twelve Data API Key is not provided. Please connect to APIs in the sidebar.")
        return
    
    with st.spinner("Fetching data and calculating correlations..."):
        try:
            all_data = {}
            for asset in selected_assets:
                ts = td_client_instance.time_series(symbol=asset, interval="1day", outputsize=90).as_pandas()
                all_data[asset] = ts['close']
            
            price_df = pd.DataFrame(all_data)
            corr_df = price_df.pct_change().corr()
            
            fig = px.imshow(corr_df, text_auto=".2f", aspect="auto",
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title="30-Day Price Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not calculate correlations: {e}")

def handle_portfolio_optimizer(td_client_instance):
    st.header("📊 Portfolio Optimizer")
    st.write("This tool is for demonstration purposes and uses simplified assumptions.")
    tickers_input = st.text_input("Enter stock symbols (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    if st.button("Optimize Portfolio!"):
        if td_client_instance is None:
            st.error("Twelve Data API Key is not provided. Please connect to APIs in the sidebar.")
            return
        
        with st.spinner("Fetching data and optimizing..."):
            try:
                returns_data = {}
                for ticker in tickers:
                    ts = td_client_instance.time_series(symbol=ticker, interval="1day", outputsize=252).as_pandas()
                    returns_data[ticker] = ts['close'].pct_change()
                
                returns_df = pd.DataFrame(returns_data).dropna()
                
                mu = returns_df.mean() * 252
                sigma = returns_df.cov() * 252
                
                weights = 1 / np.diag(sigma)
                weights /= weights.sum()
                
                portfolio_return = np.sum(mu * weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
                sharpe_ratio = portfolio_return / portfolio_vol
                
                st.subheader("Optimized Allocation (Max Sharpe Ratio)")
                fig = px.pie(values=weights, names=tickers, title="Recommended Asset Allocation", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
                
                c1, c2, c3 = st.columns(3)
                with c1: 
                    display_metric("Expected Return", f"{portfolio_return:.2%}", "Annual")
                with c2: 
                    display_metric("Portfolio Risk", f"{portfolio_vol:.2%}", "Volatility")
                with c3: 
                    display_metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
            except Exception as e:
                st.error(f"Could not optimize portfolio: {e}")

def handle_technical_analysis(td_client_instance):
    st.header("📉 Technical Analysis Dashboard")
    
    if td_client_instance is None:
        st.error("Twelve Data API Key is not provided. Please connect to APIs in the sidebar.")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Symbol Selection")
        selected_popular = st.selectbox(
            "Choose from popular symbols:",
            ["Select..."] + POPULAR_SYMBOLS,
            index=0
        )
        
        custom_symbol = st.text_input("Or enter custom symbol:", "").upper()
        
        if custom_symbol:
            ticker = custom_symbol
        elif selected_popular != "Select...":
            ticker = selected_popular
        else:
            ticker = "AAPL"
        
        period_options = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        selected_period = st.selectbox("Select time period:", list(period_options.keys()), index=1)
        outputsize = period_options[selected_period]
        
        st.subheader("Technical Indicators")
        indicators = []
        if st.checkbox("SMA 20", value=True): indicators.append("SMA20")
        if st.checkbox("SMA 50", value=True): indicators.append("SMA50")
        if st.checkbox("EMA 20", value=True): indicators.append("EMA20")
        if st.checkbox("Bollinger Bands", value=True): indicators.append("BB")
        if st.checkbox("RSI", value=True): indicators.append("RSI")
        if st.checkbox("MACD", value=True): indicators.append("MACD")
        if st.checkbox("Stochastic", value=True): indicators.append("Stoch")
        if st.checkbox("ATR", value=False): indicators.append("ATR")
    
    with col2:
        if st.button("Generate Technical Analysis", key="tech_analysis"):
            with st.spinner(f"Fetching data and generating technical analysis for {ticker}..."):
                try:
                    ts = td_client_instance.time_series(
                        symbol=ticker,
                        interval="1day",
                        outputsize=outputsize
                    ).as_pandas().sort_index(ascending=True)
                    
                    if ts.empty:
                        st.error(f"No data found for symbol {ticker}")
                        return
                    
                    fig = create_technical_analysis_chart(ts, ticker, indicators)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Key Technical Metrics")
                    current_price = ts['close'].iloc[-1]
                    rsi_current = calculate_rsi(ts['close']).iloc[-1]
                    sma20_current = calculate_sma(ts['close'], 20).iloc[-1]
                    sma50_current = calculate_sma(ts['close'], 50).iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        display_metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        rsi_color = "indicator-positive" if rsi_current < 30 else "indicator-negative" if rsi_current > 70 else "indicator-neutral"
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">RSI</div>
                            <div class="metric-value {rsi_color}">{rsi_current:.1f}</div>
                            <div class="metric-delta">{interpret_rsi(rsi_current)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        trend_color = "indicator-positive" if current_price > sma20_current else "indicator-negative"
                        delta_pct = ((current_price / sma20_current - 1) * 100)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">vs SMA 20</div>
                            <div class="metric-value {trend_color}">{delta_pct:+.1f}%</div>
                            <div class="metric-delta">Short-term trend</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        trend_color = "indicator-positive" if current_price > sma50_current else "indicator-negative"
                        delta_pct = ((current_price / sma50_current - 1) * 100)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">vs SMA 50</div>
                            <div class="metric-value {trend_color}">{delta_pct:+.1f}%</div>
                            <div class="metric-delta">Medium-term trend</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.subheader("Quick Quote")
                    try:
                        quote = td_client_instance.quote(symbol=ticker).as_json()
                        if quote and isinstance(quote, dict):
                            st.json(quote)
                        else:
                            st.info("Quote data not available in expected format")
                    except Exception as e:
                        st.warning(f"Could not fetch quote data: {e}")
                        
                except Exception as e:
                    st.error(f"Failed to generate technical analysis: {e}")

# -- Enhanced Sidebar --
def create_enhanced_sidebar():
    st.sidebar.markdown("#### Settings")
    
    with st.sidebar.expander("API Configuration", expanded=True):
        twelvedata_key = st.text_input("Twelve Data API Key", type="password", placeholder="Enter your API key")
        groq_key = st.text_input("Groq API Key", type="password", placeholder="Enter your API key")
        
        if st.button("Connect APIs"):
            if twelvedata_key and groq_key:
                try:
                    st.session_state['td_client'] = TDClient(apikey=twelvedata_key)
                    st.session_state['groq_client'] = Groq(api_key=groq_key)
                    st.sidebar.success("Connected successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error connecting to APIs: {e}")
            else:
                st.sidebar.error("Please enter both API keys")
    
    with st.sidebar.expander("Trading Preferences"):
        risk_tolerance = st.select_slider("Risk Tolerance", options=["Conservative", "Moderate", "Aggressive"])
        investment_horizon = st.selectbox("Investment Horizon", ["Short-term", "Medium-term", "Long-term"])
        preferred_sectors = st.multiselect("Preferred Sectors", ["Technology", "Healthcare", "Finance", "Energy", "Consumer"])
    
    with st.sidebar.expander("Alerts & Notifications"):
        price_alerts = st.checkbox("Price Movement Alerts")
        news_alerts = st.checkbox("Breaking News Alerts")
        portfolio_alerts = st.checkbox("Portfolio Performance Alerts")
    
    st.sidebar.markdown("#### Quick Actions")
    if st.sidebar.button("Refresh Data"):
        st.rerun()
    if st.sidebar.button("Export Portfolio"):
        st.sidebar.success("Portfolio exported!")
    if st.sidebar.button("Generate Report"):
        st.sidebar.success("Report generated!")

# -- Main App Function --
def main():
    st.set_page_config(
        page_title="Real-Time Market Overview & Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_enhanced_styling()
    create_enhanced_sidebar()
       
    td_client_instance = st.session_state.get('td_client')
    groq_client_instance = st.session_state.get('groq_client')
    
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio(
        "Choose a feature:",
        [
            "📊 Dashboard Overview",
            "🤖 AI Stock Analysis",
            "🔮 Price Prediction",
            "📈 Asset Correlation",
            "📊 Portfolio Optimizer",
            "📉 Technical Analysis"
        ]
    )
    
    if page == "📊 Dashboard Overview":
        handle_dashboard_overview()
    elif page == "🤖 AI Stock Analysis":
        handle_ai_analysis(td_client_instance, groq_client_instance)
    elif page == "🔮 Price Prediction":
        handle_prediction(td_client_instance, groq_client_instance)
    elif page == "📈 Asset Correlation":
        handle_correlation(td_client_instance)
    elif page == "📊 Portfolio Optimizer":
        handle_portfolio_optimizer(td_client_instance)
    elif page == "📉 Technical Analysis":
        handle_technical_analysis(td_client_instance)
    
    st.markdown("---")
    st.markdown(""" 
    <div style="text-align: center; color: #687280; font-size: 0.9rem;"> 
        <strong>Real-Time Market Overview & Analytics</strong> | Built with Python, Streamlit, Plotly & Groq |
        © 2025 Armando Saboia
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()