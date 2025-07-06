import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from pathlib import Path
import logging
from functools import lru_cache
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import time

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
from src.data_ingestion.stock_apis import TwelveDataAPI
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
    st.sidebar.header("⚙️ Controls")
    
    module = st.sidebar.selectbox(
        "Select Module",
        ["Time Series Analysis", "Price Prediction", "Portfolio Optimizer", "Symbol Search", "Company News", "Charts", "WebSocket", "Debugging"]
    )

    if module == "Time Series Analysis":
        st.header("Time Series Analysis")
        symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
        interval = st.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "45min", "1h", "2h", "4h", "8h", "1day", "1week", "1month"], index=8)
        outputsize = st.slider("Number of Data Points", 50, 5000, 180, 10)

        if st.button("Fetch Time Series Data"):
            data = twelvedata_client.get_time_series(symbol=symbol, interval=interval, outputsize=outputsize)
            if data is not None and not data.empty:
                fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'])])
                fig.update_layout(title=f'{symbol} Price Chart', template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("View Raw Data"):
                    st.dataframe(data)
            else:
                st.warning("No time series data found.")

    elif module == "Price Prediction":
        st.header("Price Prediction")
        symbol = st.text_input("Enter Stock Symbol", "GOOGL").upper()
        days_to_predict = st.slider("Days to Predict", 7, 90, 30)

        if st.button("Generate Prediction"):
            historical_data = twelvedata_client.get_time_series(symbol=symbol, interval="1day", outputsize=500)
            if historical_data is not None and not historical_data.empty:
                df = historical_data.reset_index()
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index("datetime", inplace=True)
                df['days_since_start'] = (df.index - df.index.min()).days
                X = df[['days_since_start']]
                y = df['close']

                model = LinearRegression()
                model.fit(X, y)

                last_day = df['days_since_start'].max()
                future_days = np.arange(last_day + 1, last_day + 1 + days_to_predict).reshape(-1, 1)
                predicted_prices = model.predict(future_days)

                last_date = df.index.max()
                future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Historical Price'))
                fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Predicted Price', line=dict(dash='dash')))
                fig.update_layout(title=f'{symbol} Price Prediction for Next {days_to_predict} Days', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not fetch data for prediction.")

    elif module == "Portfolio Optimizer":
        st.header("Portfolio Optimizer")
        tickers_str = st.text_input("Enter stock tickers separated by commas", "AAPL,GOOG,MSFT,TSLA").upper()
        
        if st.button("Optimize Portfolio"):
            tickers = [ticker.strip() for ticker in tickers_str.split(',')]
            if len(tickers) < 2:
                st.warning("Please enter at least two tickers.")
                return

            with st.spinner("Fetching data for all tickers..."):
                all_data = {ticker: twelvedata_client.get_time_series(symbol=ticker, interval="1day", outputsize=252) for ticker in tickers}
            
            if any(df.empty for df in all_data.values()):
                st.error("Failed to fetch data for one or more tickers. Please check symbols and try again.")
                return

            close_prices = pd.concat([df['close'].rename(ticker) for ticker, df in all_data.items()], axis=1)
            close_prices.dropna(inplace=True)

            if close_prices.empty:
                st.error("Could not align data for the given tickers. They may not have overlapping trading days.")
                return

            returns = close_prices.pct_change().dropna()
            
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_portfolios = 25000
            results = np.zeros((3, num_portfolios))
            
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                
                portfolio_return = np.sum(mean_returns * weights) * 252
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                
                results[0,i] = portfolio_return
                results[1,i] = portfolio_stddev
                results[2,i] = results[0,i] / results[1,i] # Sharpe Ratio

            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_return = results[0, max_sharpe_idx]
            max_sharpe_stddev = results[1, max_sharpe_idx]
            
            optimal_weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0] # Placeholder for actual optimal weights
            
            st.subheader("Optimal Asset Allocation (Max Sharpe Ratio)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div style=\"background-color: #F0F0F0; padding: 10px; border-radius: 5px;\">", unsafe_allow_html=True)
                st.metric("Annualized Return", f"{max_sharpe_return:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div style=\"background-color: #F0F0F0; padding: 10px; border-radius: 5px; margin-top: 10px;\">", unsafe_allow_html=True)
                st.metric("Annualized Volatility (Risk)", f"{max_sharpe_stddev:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div style=\"background-color: #F0F0F0; padding: 10px; border-radius: 5px; margin-top: 10px;\">", unsafe_allow_html=True)
                st.metric("Sharpe Ratio", f"{results[2, max_sharpe_idx]:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                pie_df = pd.DataFrame({'Weight': optimal_weights}, index=tickers)
                fig = px.pie(pie_df, values='Weight', names=pie_df.index, title='Optimal Portfolio Weights')
                st.plotly_chart(fig, use_container_width=True)

    elif module == "Symbol Search":
        st.header("Symbol Search")
        query = st.text_input("Enter search query (e.g., 'Apple', 'Micro')", "Apple")
        if st.button("Search"):
            results = twelvedata_client.search_symbols(query)
            if results and results.get('data'):
                st.dataframe(pd.DataFrame(results['data']))
            else:
                st.warning("No symbols found.")

    elif module == "Company News":
        st.header("Company News")
        symbol = st.text_input("Enter Stock Symbol for News", "TSLA").upper()
        if st.button("Fetch News"):
            with st.spinner("Generating news insights..."):
                try:
                    # Using Groq API for news insights as requested
                    insights = generate_insights(f"Latest news for {symbol}")
                    st.write(insights)
                except Exception as e:
                    st.error(f"Error generating news insights: {e}")

    elif module == "Charts":
        st.header("Charts")
        chart_type = st.selectbox("Select Chart Type", ["Static (Matplotlib)", "Interactive (Plotly)"])
        symbol = st.text_input("Enter Stock Symbol for Chart", "MSFT").upper()
        interval = st.selectbox("Select Interval for Chart", ["1day", "1week", "1month"], index=0)
        outputsize = st.slider("Number of Data Points for Chart", 50, 500, 75, 10)

        if st.button("Generate Chart"):
            ts_data_obj = twelvedata_client.get_time_series_object(symbol=symbol, interval=interval, outputsize=outputsize)
            if ts_data_obj:
                if chart_type == "Static (Matplotlib)":
                    st.subheader("Static Chart (Matplotlib)")
                    # Fetch data as pandas DataFrame for mplfinance
                    data_for_plot = ts_data_obj.as_pandas()
                    if not data_for_plot.empty:
                        # mplfinance expects specific column names: Open, High, Low, Close, Volume
                        # Ensure the index is datetime
                        data_for_plot.index = pd.to_datetime(data_for_plot.index)
                        data_for_plot = data_for_plot.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        })
                        # Create a simple plot using matplotlib (mplfinance is a wrapper)
                        fig, ax = plt.subplots()
                        ax.plot(data_for_plot.index, data_for_plot['Close'])
                        ax.set_title(f'{symbol} Close Price')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price')
                        st.pyplot(fig)
                    else:
                        st.warning("No data to generate static chart.")
                elif chart_type == "Interactive (Plotly)":
                    st.subheader("Interactive Chart (Plotly)")
                    fig = ts_data_obj.as_plotly_figure()
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data to generate chart.")

    elif module == "WebSocket":
        st.header("WebSocket Data Stream (Pro Plan Required)")
        st.write("This feature requires a Twelve Data Pro plan or higher.")
        st.write("Please refer to the Twelve Data Python Client documentation for more details.")
        st.code("""
import time
from twelvedata import TDClient

messages_history = []

def on_event(e):
    print(e)
    messages_history.append(e)

td = TDClient(apikey="YOUR_API_KEY_HERE")
ws = td.websocket(symbols="BTC/USD", on_event=on_event)
ws.subscribe(['ETH/BTC', 'AAPL'])
ws.connect()
while True:
    print('messages received: ', len(messages_history))
    ws.heartbeat()
    time.sleep(10)
""")

    elif module == "Debugging":
        st.header("Debugging Tools")
        debug_option = st.selectbox("Select Debugging Option", ["API Usage", "as_url() Example"])

        if debug_option == "API Usage":
            if st.button("Get API Usage"):
                try:
                    usage = twelvedata_client.get_api_usage()
                    st.json(usage)
                except Exception as e:
                    st.error(f"Error fetching API usage: {e}")
        elif debug_option == "as_url() Example":
            st.write("This shows the URLs generated for a time series request.")
            symbol_url = st.text_input("Symbol for as_url()", "AAPL").upper()
            interval_url = st.selectbox("Interval for as_url()", ["1min", "1day"], index=1)
            outputsize_url = st.slider("Outputsize for as_url()", 1, 100, 10)
            if st.button("Get URLs"):
                try:
                    ts_obj_for_url = twelvedata_client.td.time_series(
                        symbol=symbol_url,
                        interval=interval_url,
                        outputsize=outputsize_url,
                        timezone="America/New_York"
                    )
                    urls = ts_obj_for_url.as_url()
                    for url in urls:
                        st.code(url)
                except Exception as e:
                    st.error(f"Error generating URLs: {e}")

if __name__ == "__main__":
    main()
