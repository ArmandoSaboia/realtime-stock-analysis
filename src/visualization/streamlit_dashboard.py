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
from groq import Groq
from statsmodels.tsa.arima.model import ARIMA

# --- Configuration and Initialization ---

# Load environment variables from a .env file
load_dotenv()

# Global variables for API clients, initialized after user input
td_client = None
groq_client = None


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
    </style>
    """, unsafe_allow_html=True)

# --- AI and Data Interpretation Functions ---

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_stock_insights(groq_client_instance, query, model="llama3-8b-8192"):
    """Get AI-powered insights from Groq."""
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
    st.header(" AI-Powered Stock Analysis")
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
            # 1. Fetch data from Twelve Data
            try:
                ts = td_client_instance.time_series(symbol=ticker, interval="1day", outputsize=90).as_pandas().sort_index(ascending=True)
            except Exception as e:
                st.error(f"Failed to fetch time series data for {ticker} from Twelve Data: {e}")
                return

            # 2. Prepare data points for analysis
            latest_price = ts['close'].iloc[-1]
            change_30d = (latest_price / ts['close'].iloc[-30] - 1) * 100 if len(ts) >= 30 else 0
            volatility = ts['close'].pct_change().rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100 if len(ts) >= 30 else 0

            # 3. Generate AI insights query based on market data
            query = f"""
            Generate a concise stock analysis for {ticker} based on the following market data.
            - **Current Price:** ${latest_price:.2f}
            - **30-Day Change:** {change_30d:.2f}% ({interpret_change(change_30d)})
            - **Annualized Volatility:** {volatility:.2f}% ({interpret_volatility(volatility)})

            Provide a summary, a bullish case based on the data, a bearish case based on the data, and a final outlook.
            """
            
            # 4. Display results
            st.subheader(f"Analysis for {ticker}")
            ai_insights = get_stock_insights(groq_client_instance, query)
            st.markdown(ai_insights)

def handle_prediction(td_client_instance, groq_client_instance):
    """Handler for the Price Prediction page."""
    st.header(" Price Prediction")
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
                # Fetch historical data
                hist_data = td_client_instance.time_series(symbol=ticker, interval="1day", outputsize=200).as_pandas().sort_index(ascending=True)
                
                # Simple ARIMA model for simulation
                model = ARIMA(hist_data['close'], order=(5,1,0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=days_to_predict)
                
                future_dates = pd.to_datetime(pd.date_range(start=hist_data.index[-1] + timedelta(days=1), periods=days_to_predict))
                future_prices = forecast.values

                # Display chart
                st.plotly_chart(create_prediction_chart(hist_data, future_dates, future_prices, ticker), use_container_width=True)

                # Display AI analysis of the prediction
                price_change_pct = (future_prices[-1] / hist_data['close'].iloc[-1] - 1) * 100
                prediction_query = (
                    f"Analyze the {days_to_predict}-day price prediction for {ticker}:"
                    f"\n- Current price: ${hist_data['close'].iloc[-1]:.2f}"
                    f"\n- Predicted price in {days_to_predict} days: ${future_prices[-1]:.2f} ({price_change_pct:+.2f}%)"
                    f"\n- Discuss potential factors, risks, and opportunities for this prediction based on time-series forecasting."
                )
                insights = get_stock_insights(groq_client_instance, prediction_query)
                st.subheader("AI Interpretation of Prediction")
                st.markdown(insights)

            except Exception as e:
                st.error(f"Could not generate prediction: {e}")

def handle_correlation(td_client_instance):
    """Handler for the Asset Correlation page."""
    st.header(" Asset Correlation Matrix")
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
                # Fetch 90 days of data for correlation
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
                
                # Simplified optimization for max Sharpe Ratio
                mu = returns_df.mean() * 252
                sigma = returns_df.cov() * 252
                
                # Inverse variance portfolio as a simple optimization strategy
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

def handle_market_data_browser(td_client_instance):
    """Handler for browsing market data like lists, fundamentals, etc."""
    st.header(" Market Data Browser")
    
    if td_client_instance is None:
        st.error("Twelve Data API Key is not provided. Please connect to APIs in the sidebar.")
        return

    # Corrected method name to .search() which is standard in the library
    st.subheader("Symbol Search")
    query = st.text_input("Enter search query (e.g., 'Apple', 'Micro')", "Apple")
    if st.button("Search Symbols"):
        try:
            results = td_client_instance.search(symbol=query).as_json()
            st.json(results)
        except Exception as e:
            st.error(f"An error occurred during symbol search: {e}")

    st.subheader("Instrument Lists")
    list_choice = st.selectbox("Select an instrument list", ["Stocks", "ETFs", "Indices", "Forex Pairs", "Cryptocurrencies"])
    if st.button("Get List"):
        with st.spinner("Fetching list..."):
            try:
                if list_choice == "Stocks": data = td_client_instance.get_stocks_list().as_pandas()
                elif list_choice == "ETFs": data = td_client_instance.get_etf_list().as_pandas()
                elif list_choice == "Indices": data = td_client_instance.get_indices_list().as_pandas()
                elif list_choice == "Forex Pairs": data = td_client_instance.get_forex_pairs_list().as_pandas()
                elif list_choice == "Cryptocurrencies": data = td_client_instance.get_cryptocurrencies_list().as_pandas()
                st.dataframe(data)
            except Exception as e:
                st.error(f"Failed to fetch list: {e}")

# --- Main App ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Real Time Stock Market Analysis", layout="wide")
    apply_custom_styling()
    st.title(" Real Time Stock Market Analysis")

    st.sidebar.header("API Configuration")
    twelvedata_api_key_input = st.sidebar.text_input("Enter your Twelve Data API Key", type="password")
    groq_api_key_input = st.sidebar.text_input("Enter your Groq API Key", type="password")

    if st.sidebar.button("Connect to APIs"):
        if twelvedata_api_key_input and groq_api_key_input:
            try:
                st.session_state['td_client'] = TDClient(apikey=twelvedata_api_key_input)
                st.session_state['groq_client'] = Groq(api_key=groq_api_key_input)
                st.sidebar.success("Successfully connected to APIs!")
            except Exception as e:
                st.sidebar.error(f"Error connecting to APIs: {e}")
                st.session_state['td_client'] = None
                st.session_state['groq_client'] = None
        else:
            st.sidebar.warning("Please enter both API keys.")
            st.session_state['td_client'] = None
            st.session_state['groq_client'] = None

    # Retrieve clients from session state, or None if not yet connected
    td_client_instance = st.session_state.get('td_client')
    groq_client_instance = st.session_state.get('groq_client')

    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a feature",
        [
            "AI Stock Analysis", 
            "Price Prediction", 
            "Portfolio Optimizer",
            "Asset Correlation", 
            "Market Data Browser",
        ]
    )

    if app_mode == "AI Stock Analysis":
        handle_ai_analysis(td_client_instance, groq_client_instance)
    elif app_mode == "Price Prediction":
        handle_prediction(td_client_instance, groq_client_instance)
    elif app_mode == "Portfolio Optimizer":
        handle_portfolio_optimizer(td_client_instance)
    elif app_mode == "Asset Correlation":
        handle_correlation(td_client_instance)
    elif app_mode == "Market Data Browser":
        handle_market_data_browser(td_client_instance)


if __name__ == "__main__":
    main()
