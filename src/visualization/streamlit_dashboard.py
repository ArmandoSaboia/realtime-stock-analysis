# src/visualization/streamlit_dashboard.py

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

# --- Configuration and Initialization ---

# Load environment variables from a .env file
load_dotenv()

# Global variables for API clients, initialized after user input
td_client = None
groq_client = None

# Lista de símbolos populares para demonstração
POPULAR_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BABA', 'TSM',
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'BTC/USD', 'ETH/USD', 'XRP/USD', 'ADA/USD',
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD'
]

# --- Custom Styling ---

def apply_custom_styling():
    """Apply custom CSS for the Streamlit app."""
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            font-weight: 600;
        }
        h1 {
            color: #1E3A8A;
            padding-bottom: 1rem;
            border-bottom: 1px solid #f0f0f0;
            margin-bottom: 2rem;
        }
        h2, h3 {
            color: #2563EB;
            margin-top: 1.5rem;
        }
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
            color: #6B7280;
            font-weight: 500;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1E3A8A;
            margin: 0.3rem 0;
        }
        .indicator-positive {
            color: #22C55E;
        }
        .indicator-negative {
            color: #EF4444;
        }
        .indicator-neutral {
            color: #6B7280;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Technical Indicators Functions ---

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    return atr

# --- AI and Data Interpretation Functions ---

@st.cache_data(ttl=3600)
def get_stock_insights(query, model="llama3-8b-8192"):
    """Get AI-powered insights from Groq."""
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
    """Interpret 30-day change value."""
    if change_val > 10: return "Strong bullish momentum"
    if change_val > 5: return "Positive performance"
    if change_val > 0: return "Slight positive movement"
    if change_val > -5: return "Slight weakness"
    return "Significant downward pressure"

def interpret_volatility(vol_val):
    """Interpret volatility value."""
    return "High" if vol_val > 3 else "Moderate" if 1.5 < vol_val <= 3 else "Low"

def interpret_rsi(rsi_val):
    """Interpret RSI value."""
    if rsi_val > 70: return "Overbought"
    elif rsi_val < 30: return "Oversold"
    else: return "Neutral"

# --- UI Display Components ---

def display_metric(label, value, delta=None):
    """Display a metric with custom styling."""
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f"<div class='metric-delta'>{delta}</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)

def create_technical_analysis_chart(data, ticker, indicators=None):
    """Create comprehensive technical analysis chart with all indicators."""
    if indicators is None:
        indicators = ['SMA20', 'SMA50', 'EMA20', 'BB', 'RSI', 'MACD', 'Stoch', 'ATR']
    
    # Create subplots
    rows = 4
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{ticker} - Price & Moving Averages',
            'RSI (Relative Strength Index)',
            'MACD',
            'Stochastic Oscillator'
        ),
        row_heights=[0.5, 0.2, 0.2, 0.15]
    )
    
    # Calculate indicators
    sma20 = calculate_sma(data['close'], 20)
    sma50 = calculate_sma(data['close'], 50)
    ema20 = calculate_ema(data['close'], 20)
    
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data['close'])
    rsi = calculate_rsi(data['close'])
    macd, signal, histogram = calculate_macd(data['close'])
    stoch_k, stoch_d = calculate_stochastic(data['high'], data['low'], data['close'])
    atr = calculate_atr(data['high'], data['low'], data['close'])
    
    # Main price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving averages
    if 'SMA20' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=sma20, name='SMA 20', line=dict(color='orange')), row=1, col=1)
    if 'SMA50' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=sma50, name='SMA 50', line=dict(color='red')), row=1, col=1)
    if 'EMA20' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=ema20, name='EMA 20', line=dict(color='purple')), row=1, col=1)
    
    # Bollinger Bands
    if 'BB' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=upper_bb, name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=lower_bb, name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI
    if 'RSI' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=signal, name='Signal', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color='gray'), row=3, col=1)
    
    # Stochastic
    if 'Stoch' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=stoch_k, name='%K', line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=stoch_d, name='%D', line=dict(color='red')), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - Complete Technical Analysis',
        xaxis_title='Date',
        height=800,
        template='plotly_white',
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stochastic", row=4, col=1)
    
    return fig

def create_prediction_chart(historical_data, future_dates, future_prices, ticker):
    """Create an interactive prediction chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['close'], name='Historical', line=dict(color='#3B82F6')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Prediction', line=dict(color='#EF4444', dash='dash')))
    fig.update_layout(
        title=f"{ticker.upper()} Price Prediction",
        xaxis_title='Date', yaxis_title='Price ($)',
        template='plotly_white', height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Page/Feature Handlers ---

def handle_ai_analysis(td_client_instance, groq_client_instance):
    """Handler for the AI Stock Analysis page."""
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
    """Handler for the Price Prediction page."""
    st.header("📈 Price Prediction")
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

                price_change_pct = (future_prices[-1] / hist_data['close'].iloc[-1] - 1) * 100
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
    """Handler for the Asset Correlation page."""
    st.header("📊 Asset Correlation Matrix")
    available_assets = ['SPY', 'QQQ', 'DIA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'BTC/USD', 'ETH/USD']
    selected_assets = st.multiselect(
        "Select at least 2 assets for correlation analysis",
        available_assets,
        default=['AAPL', 'MSFT', 'GOOG', 'TSLA']
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
    """Handler for the Portfolio Optimizer page."""
    st.header("⚖️ Portfolio Optimizer")
    st.write("This tool is for demonstration purposes and uses simplified assumptions.")

    tickers_input = st.text_input("Enter stock symbols (comma-separated)", "AAPL,MSFT,GOOG,AMZN,TSLA,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(',')]

    if st.button("Optimize Portfolio"):
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
                with c1: display_metric("Expected Return", f"{portfolio_return:.2%}", "Annual")
                with c2: display_metric("Portfolio Risk", f"{portfolio_vol:.2%}", "Volatility")
                with c3: display_metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            except Exception as e:
                st.error(f"Could not optimize portfolio: {e}")

def handle_technical_analysis(td_client_instance):
    """Handler for the Technical Analysis page with interactive charts."""
    st.header("📊 Technical Analysis Dashboard")
    
    if td_client_instance is None:
        st.error("Twelve Data API Key is not provided. Please connect to APIs in the sidebar.")
        return

    # Sidebar for symbol selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Symbol Selection")
        
        # Popular symbols dropdown
        selected_popular = st.selectbox(
            "Choose from popular symbols:",
            ["Select..."] + POPULAR_SYMBOLS,
            index=0
        )
        
        # Custom symbol input
        custom_symbol = st.text_input("Or enter custom symbol:", "").upper()
        
        # Final symbol selection
        if custom_symbol:
            ticker = custom_symbol
        elif selected_popular != "Select...":
            ticker = selected_popular
        else:
            ticker = "AAPL"
        
        # Time period selection
        period_options = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        selected_period = st.selectbox("Select time period:", list(period_options.keys()), index=1)
        outputsize = period_options[selected_period]
        
        # Technical indicators selection
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
        if st.button("📈 Generate Technical Analysis", key="tech_analysis"):
            with st.spinner(f"Fetching data and generating technical analysis for {ticker}..."):
                try:
                    # Fetch historical data
                    ts = td_client_instance.time_series(
                        symbol=ticker, 
                        interval="1day", 
                        outputsize=outputsize
                    ).as_pandas().sort_index(ascending=True)
                    
                    if ts.empty:
                        st.error(f"No data found for symbol {ticker}")
                        return
                    
                    # Create technical analysis chart
                    fig = create_technical_analysis_chart(ts, ticker, indicators)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display key metrics
                    st.subheader("📊 Key Technical Metrics")
                    
                    # Calculate current values
                    current_price = ts['close'].iloc[-1]
                    rsi_current = calculate_rsi(ts['close']).iloc[-1]
                    sma20_current = calculate_sma(ts['close'], 20).iloc[-1]
                    sma50_current = calculate_sma(ts['close'], 50).iloc[-1]
                    
                    # Display metrics in columns
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
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">vs SMA 20</div>
                            <div class="metric-value {trend_color}">{((current_price/sma20_current - 1) * 100):+.1f}%</div>
                            <div class="metric-delta">Short-term trend</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        trend_color = "indicator-positive" if current_price > sma50_current else "indicator-negative"
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">vs SMA 50</div>
                            <div class="metric-value {trend_color}">{((current_price/sma50_current - 1) * 100):+.1f}%</div>
                            <div class="metric-delta">Medium-term trend</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Quick quote information
                    st.subheader("💼 Quick Quote")
                    try:
                        quote = td_client_instance.quote(symbol=ticker).as_json()
                        if quote and isinstance(quote, dict):
                            st.json(quote)
                        else:
                            st.info("Quote data not available in expected format")
                    except Exception as e:
                        st.warning(f"Could not fetch quote data: {e}")
                    
                    # Price information
                    st.subheader("💰 Price Information")
                    try:
                        price_data = td_client_instance.price(symbol=ticker).as_json()
                        if price_data and isinstance(price_data, dict):
                            st.json(price_data)
                        else:
                            st.info("Price data not available in expected format")
                    except Exception as e:
                        st.warning(f"Could not fetch price data: {e}")
                        
                except Exception as e:
                    st.error(f"Failed to generate technical analysis: {e}")

# --- Main App ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Real Time Stock Market Analysis", layout="wide")
    apply_custom_styling()
    st.title("📈 Real-Time Stock Market Analysis")

    st.sidebar.header("🔑 API Configuration")
    twelvedata_api_key_input = st.sidebar.text_input("Enter your Twelve Data API Key", type="password")
    groq_api_key_input = st.sidebar.text_input("Enter your Groq API Key", type="password")

    if st.sidebar.button("Connect to APIs"):
        if twelvedata_api_key_input and groq_api_key_input:
            try:
                st.session_state['td_client'] = TDClient(apikey=twelvedata_api_key_input)
                st.session_state['groq_client'] = Groq(api_key=groq_api_key_input)
                st.sidebar.success("✅ APIs connected successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error connecting to APIs: {e}")
        else:
            st.sidebar.warning("⚠️ Please enter both API keys.")

    # Get client instances from session state
    td_client_instance = st.session_state.get('td_client')
    groq_client_instance = st.session_state.get('groq_client')

    # Display connection status
    if td_client_instance and groq_client_instance:
        st.sidebar.success("🟢 APIs Connected")
    else:
        st.sidebar.info("🔴 APIs Not Connected")
        st.info("👆 Please enter your API keys in the sidebar to start using the application.")

    # Navigation
    st.sidebar.header("🧭 Navigation")
    page = st.sidebar.radio(
        "Choose a feature:",
        [
            "📊 Technical Analysis",
            "🤖 AI Stock Analysis", 
            "📈 Price Prediction",
            "📊 Asset Correlation",
            "⚖️ Portfolio Optimizer"
        ]
    )

    # Page routing
    if page == "📊 Technical Analysis":
        handle_technical_analysis(td_client_instance)
    elif page == "🤖 AI Stock Analysis":
        handle_ai_analysis(td_client_instance, groq_client_instance)
    elif page == "📈 Price Prediction":
        handle_prediction(td_client_instance, groq_client_instance)
    elif page == "📊 Asset Correlation":
        handle_correlation(td_client_instance)
    elif page == "⚖️ Portfolio Optimizer":
        handle_portfolio_optimizer(td_client_instance)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Real Time Stock Market Analysis**")
    st.sidebar.markdown("Built with Streamlit, Twelve Data, and Groq AI")

if __name__ == "__main__":
    main()
