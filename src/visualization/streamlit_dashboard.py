import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.genai.langchain_insights import generate_insights
from src.genai.config import config
from src.data_ingestion.stock_apis import TwelveDataAPI
from sklearn.linear_model import LinearRegression
import os
import sys
from pathlib import Path
import logging
from functools import lru_cache

# Ensure the project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Configure environment
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize API clients
twelvedata_client = TwelveDataAPI()

# Global model instance
OPT_MODEL = None

@st.cache_resource
def initialize_model():
    """Initialize the OPT-125M model at startup with caching."""
    global OPT_MODEL
    if OPT_MODEL is None:
        try:
            logger.info("Initializing OPT-125M model")
            # Placeholder for actual model initialization
            # device = torch.device('cpu')
            # torch.set_num_threads(1)
            # torch.set_grad_enabled(False)
            # model_name = 'facebook/opt-125m'
            # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_name,
            #     device_map='cpu',
            #     torch_dtype=torch.float16,
            #     low_cpu_mem_usage=True
            # )
            
            # Dummy model for now
            def generate_text_dummy(prompt):
                return f"Generated text for: {prompt}"

            OPT_MODEL = generate_text_dummy
            logger.info("Model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OPT-125M model: {str(e)}")
            return False
    return True

def get_model():
    """Get the initialized model instance."""
    return OPT_MODEL if OPT_MODEL is not None else initialize_model()

# Time series options for Twelve Data API
TIME_SERIES_OPTIONS = {
    "Intraday": {
        "interval": "1min",
        "outputsize": 60
    },
    "Daily": {
        "interval": "1day",
        "outputsize": 90
    },
    "Weekly": {
        "interval": "1week",
        "outputsize": 52
    },
    "Monthly": {
        "interval": "1month",
        "outputsize": 12
    }
}

@st.cache_data(ttl=86400) # Cache for 24 hours
def get_global_symbols():
    """Get a comprehensive list of global symbols for the dropdown."""
    us_stocks = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "INTC", "CSCO", "ADBE"],
        "Finance": ["JPM", "BAC", "GS", "MS", "C", "WFC", "V", "MA", "AXP", "BLK"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABT", "TMO", "MDT", "ABBV", "CVS", "LLY"],
        "Consumer": ["PG", "KO", "PEP", "WMT", "DIS", "MCD", "SBUX", "NKE", "HD", "LOW"]
    }
    global_stocks = {
        "Europe": ["SAP.DE", "ASML.AS", "NESN.SW", "ROG.SW", "BAYN.DE"],
        "Asia": ["7203.T", "9984.T", "005930.KS", "0700.HK", "1398.HK"],
        "Other": ["SHOP.TO", "RY.TO", "CSU.TO", "BHP.AX", "CBA.AX", "WBC.AX"]
    }
    crypto = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD"] # Twelve Data uses / for crypto
    forex = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
    
    all_symbols = {
        "US Indices": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
        "Global Indices": ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^N225", "^HSI", "000001.SS", "^AXJO", "^GSPC", "^IXIC", "^DJI", "^RUT"],
    }
    for category, symbols in us_stocks.items():
        all_symbols[f"US {category}"] = symbols
    for region, symbols in global_stocks.items():
        all_symbols[f"{region} Stocks"] = symbols
    all_symbols["Cryptocurrencies"] = crypto
    all_symbols["Forex"] = forex
    return all_symbols

def display_insights(raw_response, ticker, price_data=None, news_data=None):
    """Display comprehensive stock insights with news integration."""
    response = (raw_response[7:] if isinstance(raw_response, str) and raw_response.startswith("Answer:") else str(raw_response)).strip()
    # Simplified data points extraction for demonstration
    data_points = {"ticker": ticker.upper(), "latest_price": price_data["close"].iloc[-1] if price_data is not None and not price_data.empty else "N/A"}
    
    with st.expander("Analysis Details", expanded=True):
        st.markdown(f'<div class="insights-container">{response}</div>', unsafe_allow_html=True)

def display_metric(label, value, delta=None, delta_color="normal"):
    """Display a metric with custom styling and improved error handling."""
    try:
        if isinstance(value, (int, float)):
            value_str = f"{value:.2f}"
        else:
            value_str = value
        delta_class = "metric-delta-positive" if delta_color == "positive" else \
                      "metric-delta-negative" if delta_color == "negative" else ""
        delta_html = "" if delta is None else f'<div class="metric-delta {delta_class}">{delta}</div>'
        html = f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value_str}</div>
            {delta_html}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying metric '{label}': {str(e)}")

def apply_custom_styling():
    """Apply custom CSS for light mode only."""
    styles = {
        ".css-12oz5g7": {
            "padding": "2rem 1rem",
            "borderRadius": "15px",
            "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.05)",
            "background": "white",
            "marginBottom": "1.5rem"
        },
        "h1, h2, h3": {
            "fontFamily": "'Segoe UI', sans-serif",
            "fontWeight": "600"
        },
        "h1": {
            "color": "#1E3A8A",
            "paddingBottom": "1rem",
            "borderBottom": "1px solid solid #f0f0f0",
            "marginBottom": "2rem"
        },
        "h2, h3": {
            "color": "#2563EB",
            "marginTop": "1.5rem"
        },
        ".metric-container": {
            "background": "#F8FAFC",
            "borderRadius": "10px",
            "padding": "1rem",
            "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.04)",
            "transition": "transform 0.2s ease, box-shadow 0.2s ease"
        },
        ".metric-container:hover": {
            "transform": "translateY(-2px)",
            "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)"
        },
        ".metric-label": {
            "fontSize": "0.9rem",
            "color": "#6B7280",
            "fontWeight": "500"
        },
        ".metric-value": {
            "fontSize": "1.8rem",
            "fontWeight": "600",
            "color": "#1E3A8A",
            "margin": "0.3rem 0"
        },
        ".metric-delta": {"fontSize": "0.9rem",
            "fontWeight": "500"
        },
        ".metric-delta-positive": {
            "color": "#10B981"
        },
        ".metric-delta-negative": {
            "color": "#EF4444"
        },
        ".stButton button": {
            "backgroundColor": "#2563EB",
            "color": "white",
            "borderRadius": "8px",
            "padding": "0.5rem 1rem",
            "fontWeight": "500",
            "border": "none",
            "transition": "background-color 0.2s ease"
        },
        ".stButton button:hover": {
            "backgroundColor": "#1E40AF"
        },
        ".insights-container": {
            "background": "#F0F9FF",
            "borderLeft": "4px solid #0EA5E9",
            "padding": "1rem",
            "borderRadius": "0 8px 8px 0",
            "margin": "1rem 0"
        },
        ".key-point": {
            "padding": "0.5rem 1rem",
            "margin": "0.5rem 0",
            "background": "#F0FDF4",
            "borderRadius": "8px",
            "borderLeft": "3px solid #10B981"
        }
    }
    css = "".join(
        f"{selector} {{\n" + "".join(f"{prop}: {value};\n" for prop, value in props.items()) + "}}\n"
        for selector, props in styles.items()
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def main():
    apply_custom_styling()
    st.sidebar.title("Real-Time Stock Market Analysis")
    st.sidebar.markdown("---")
    
    # Initialize model if not already done
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = initialize_model()

    modules = ["Stock Data", "Stock Prediction", "News Insights", "Technical Indicators", "Forex/Crypto Data", "Economic Indicators"]
    data_option = st.sidebar.radio("Select Module", modules, index=0)

    if data_option == "Stock Data":
        ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
        time_series_type = st.sidebar.selectbox(
            "Time Series Type",
            options=list(TIME_SERIES_OPTIONS.keys()),
            index=1 # Default to Daily
        )
        if st.sidebar.button("Get Stock Data"):
            st.subheader(f"Stock Data for {ticker.upper()} (via Twelve Data)")
            try:
                interval = TIME_SERIES_OPTIONS[time_series_type]["interval"]
                outputsize = TIME_SERIES_OPTIONS[time_series_type]["outputsize"]
                data = twelvedata_client.get_time_series(ticker, interval, outputsize=outputsize)
                if data is not None and not data.empty:
                    df = pd.DataFrame(data)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.set_index("datetime", inplace=True)
                    df["close"] = pd.to_numeric(df["close"])
                    st.line_chart(df["close"])
                    st.subheader("Key Metrics")
                    st.metric("Average Close Price", round(df["close"].mean(), 2))
                    st.metric("Max High Price", round(pd.to_numeric(df["high"]).max(), 2))
                    st.metric("Min Low Price", round(pd.to_numeric(df["low"]).min(), 2))
                else:
                    st.write("No data found for ticker:", ticker)

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching stock data: {e}")
            except ValueError as e:
                st.error(f"API Key Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    elif data_option == "Stock Prediction":
        ticker = st.sidebar.text_input("Enter stock ticker for prediction (e.g., AAPL)", value="AAPL")
        days = st.sidebar.number_input("Days ahead to predict", min_value=1, max_value=30, value=1)
        if st.sidebar.button("Predict Stock Price"):
            st.subheader(f"Stock Prediction for {ticker.upper()} (via Twelve Data)")
            try:
                data = twelvedata_client.get_time_series(ticker, "1day", outputsize=90)
                if data is not None and not data.empty:
                    df = pd.DataFrame(data)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.set_index("datetime", inplace=True)
                    df["close"] = pd.to_numeric(df["close"])
                    
                    if len(df["close"]) < 30:
                        st.error("Not enough data to make an accurate prediction (requires at least 30 data points).")
                    else:
                        # Twelve Data returns newest data first, so reverse for linear regression
                        X = np.arange(len(df["close"]))[::-1].reshape(-1, 1) 
                        y = df["close"].values[::-1]
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        future_index = np.array([[len(df["close"]) + days - 1]])
                        predicted_price = model.predict(future_index)[0]
                        
                        st.metric("Predicted Close Price", round(predicted_price, 2))
                        
                        fig, ax = plt.subplots()
                        ax.plot(df["close"].index, df["close"], label="Historical Close")
                        last_date = df["close"].index[-1]
                        future_date = last_date + pd.Timedelta(days=days)
                        ax.plot(future_date, predicted_price, "ro", label="Predicted Price")
                        ax.set_title(f"{ticker.upper()} Close Price Prediction")
                        ax.legend()
                        st.pyplot(fig)
                else:
                    st.write("No data found for ticker:", ticker)

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching stock data for prediction: {e}")
            except ValueError as e:
                st.error(f"API Key Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")

    elif data_option == "News Insights":
        user_query = st.sidebar.text_input("Enter your news query about stocks", value="Latest technology news")
        if st.sidebar.button("Get News Insights"):
            st.subheader("News Insights")
            with st.spinner("Generating insights..."):
                try:
                    insights = generate_insights(user_query)
                    st.write(insights)
                except Exception as e:
                    st.error(f"Error generating insights: {e}")
    
    elif data_option == "Technical Indicators":
        st.subheader(f"Technical Indicators (via Twelve Data)")
        ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
        indicator_choice = st.sidebar.selectbox(
            "Select Indicator",
            ("SMA", "EMA", "MACD", "RSI")
        )
        interval_ti = st.sidebar.selectbox("Interval", options=["1min", "5min", "15min", "30min", "1day", "1week", "1month"])
        time_period_ti = st.sidebar.number_input("Time Period (for SMA, EMA, RSI)", min_value=1, value=10)
        series_type_ti = st.sidebar.selectbox("Series Type", options=["close", "open", "high", "low"])

        if st.sidebar.button("Get Indicator Data"):
            try:
                data = client.get_technical_indicator(ticker, indicator_choice, interval_ti, time_period=time_period_ti, series_type=series_type_ti)
                if data is not None and not data.empty:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    # Plotting logic for indicators would go here, depending on their structure
                else:
                    st.write("No data found for indicator:", indicator_choice)
            except ValueError as e:
                st.error(f"API Key Error: {e}")
            except Exception as e:
                st.error(f"Error fetching technical indicator: {e}")

    elif data_option == "Forex/Crypto Data":
        st.subheader(f"Forex/Crypto Data (via Twelve Data)")
        currency_pair = st.sidebar.text_input("Enter currency pair (e.g., EUR/USD)", value="EUR/USD")
        if st.sidebar.button("Get Data"):
            try:
                data = client.get_forex_quote(currency_pair)
                if data:
                    st.json(data)
                else:
                    st.write("No data found for currency pair:", currency_pair)
            except ValueError as e:
                st.error(f"API Key Error: {e}")
            except Exception as e:
                st.error(f"Error fetching forex/crypto data: {e}")

    elif data_option == "Economic Indicators":
        st.subheader(f"Economic Indicators (via Twelve Data)")
        st.write("Economic indicators not directly supported via Twelve Data client in this demo. Please refer to their API docs.")

if __name__ == "__main__":
    main()
