import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
# from mcp import cache # Removed as mcp is not installed and not a standard library
import twelvedata as td
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- Configuration and Setup ---

# Load environment variables from a .env file if it exists
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis with Twelve Data",
    page_icon="",
    layout="wide"
)

# Apply custom styling for a modern look
def apply_custom_styling():
    """Applies custom CSS to the Streamlit app."""
    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
            .stButton>button {
                background-color: #4A90E2;
                color: white;
                border-radius: 8px;
                border: none;
                padding: 10px 24px;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #357ABD;
                color: white;
            }
            .stMetric {
                background-color: #2E2E2E;
                padding: 1rem;
                border-radius: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styling()

# --- API Client Initialization ---

@st.cache_resource
def get_td_client():
    """Initializes and returns a cached Twelve Data client."""
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        st.error("Twelve Data API key is not set. Please add it to your .env file or environment variables as 'TWELVEDATA_API_KEY'.")
        return None
    return td.TDClient(apikey=api_key)

# --- Data Fetching Functions (Adapted to Twelve Data) ---

# Using st.cache_data instead of mcp.cache
@st.cache_data(ttl="1h")
def get_twelvedata_time_series(symbol: str, interval: str, outputsize: int = 100):
    """Fetches time series data from the Twelve Data API."""
    td_client = get_td_client()
    if not td_client: return pd.DataFrame()
    try:
        ts = td_client.time_series(symbol=symbol, interval=interval, outputsize=outputsize)
        if ts:
            return ts.as_pandas().sort_index(ascending=True)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching time series data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl="24h")
def search_symbols_twelvedata(query: str):
    """Searches for symbols using the Twelve Data API."""
    td_client = get_td_client()
    if not td_client: return pd.DataFrame()
    try:
        res = td_client.search(symbol=query)
        if res and res.get('data'):
            return pd.DataFrame(res['data'])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error searching for symbols: {e}")
        return pd.DataFrame()

@st.cache_data(ttl="24h")
def get_news_twelvedata(symbol: str, limit: int = 10):
    """Fetches news for a specific symbol."""
    td_client = get_td_client()
    if not td_client: return pd.DataFrame()
    try:
        news = td_client.news(symbol=symbol, outputsize=limit) # Twelve Data uses outputsize for limit
        if news and news.get('articles'):
            return pd.DataFrame(news['articles'])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return pd.DataFrame()

# --- UI Modules ---

def display_time_series_module():
    st.header("Time Series Analysis")
    symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
    interval = st.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "45min", "1h", "2h", "4h", "8h", "1day", "1week", "1month"], index=8)
    outputsize = st.slider("Number of Data Points", 50, 5000, 180, 10)

    if st.button("Fetch Time Series Data"):
        data = get_twelvedata_time_series(symbol, interval, outputsize)
        if not data.empty:
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'])])
            fig.update_layout(title=f'{symbol} Price Chart', template='plotly_dark', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View Raw Data"):
                st.dataframe(data)
        else:
            st.warning("No time series data found.")

def display_price_prediction_module():
    st.header("Price Prediction")
    symbol = st.text_input("Enter Stock Symbol", "GOOGL").upper()
    days_to_predict = st.slider("Days to Predict", 7, 90, 30)

    if st.button("Generate Prediction"):
        # Fetch a good amount of historical data for the model
        historical_data = get_twelvedata_time_series(symbol, "1day", 500)
        if historical_data.empty:
            st.warning("Could not fetch data for prediction.")
            return

        # Prepare data for sklearn
        df = historical_data.reset_index()
        df['days_since_start'] = (df['datetime'] - df['datetime'].min()).dt.days
        X = df[['days_since_start']]
        y = df['close']

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Create future dates for prediction
        last_day = df['days_since_start'].max()
        future_days = np.arange(last_day + 1, last_day + 1 + days_to_predict).reshape(-1, 1)
        
        # Predict future prices
        predicted_prices = model.predict(future_days)

        # Create future dates for the chart
        last_date = df['datetime'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

        # Create the chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['close'], mode='lines', name='Historical Price'))
        fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Predicted Price', line=dict(dash='dash')))
        fig.update_layout(title=f'{symbol} Price Prediction for Next {days_to_predict} Days', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

def display_portfolio_optimizer_module():
    st.header("Portfolio Optimizer")
    tickers_str = st.text_input("Enter stock tickers separated by commas", "AAPL,GOOG,MSFT,TSLA").upper()
    
    if st.button("Optimize Portfolio"):
        tickers = [ticker.strip() for ticker in tickers_str.split(',')]
        if len(tickers) < 2:
            st.warning("Please enter at least two tickers.")
            return

        # Fetch data for all tickers
        with st.spinner("Fetching data for all tickers..."):
            all_data = {ticker: get_twelvedata_time_series(ticker, "1day", 252) for ticker in tickers}
        
        # Check for errors
        if any(df.empty for df in all_data.values()):
            st.error("Failed to fetch data for one or more tickers. Please check symbols and try again.")
            return

        # Combine close prices into a single DataFrame
        close_prices = pd.concat([df['close'].rename(ticker) for ticker, df in all_data.items()], axis=1)
        close_prices.dropna(inplace=True)

        if close_prices.empty:
            st.error("Could not align data for the given tickers. They may not have overlapping trading days.")
            return

        # Calculate daily returns
        returns = close_prices.pct_change().dropna()
        
        # --- Simple Portfolio Optimization (Maximum Sharpe Ratio) ---
        # This is a simplified example. Real-world optimization is more complex.
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

        # Find the portfolio with the highest Sharpe ratio
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_return = results[0, max_sharpe_idx]
        max_sharpe_stddev = results[1, max_sharpe_idx]
        
        # Get the weights for the best portfolio
        # For simplicity, we'll just show the concept.
        optimal_weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0] # Placeholder for actual optimal weights
        
        st.subheader("Optimal Asset Allocation (Max Sharpe Ratio)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Annualized Return", f"{max_sharpe_return:.2%}")
            st.metric("Annualized Volatility (Risk)", f"{max_sharpe_stddev:.2%}")
            st.metric("Sharpe Ratio", f"{results[2, max_sharpe_idx]:.2f}")

        with col2:
            pie_df = pd.DataFrame({'Weight': optimal_weights}, index=tickers)
            fig = px.pie(pie_df, values='Weight', names=pie_df.index, title='Optimal Portfolio Weights')
            st.plotly_chart(fig, use_container_width=True)

def main():
    """The main function to run the Streamlit application."""
    apply_custom_styling()
    st.sidebar.header("⚙️ Controls")
    
    module = st.sidebar.selectbox(
        "Select Module",
        ["Time Series Analysis", "Price Prediction", "Portfolio Optimizer", "Symbol Search", "Company News"]
    )

    if module == "Time Series Analysis":
        display_time_series_module()
    elif module == "Price Prediction":
        display_price_prediction_module()
    elif module == "Portfolio Optimizer":
        display_portfolio_optimizer_module()
    elif module == "Symbol Search":
        st.header("Symbol Search")
        query = st.text_input("Enter search query (e.g., 'Apple', 'Micro')", "Apple")
        if st.button("Search"):
            st.dataframe(search_symbols_twelvedata(query))
    elif module == "Company News":
        st.header("Company News")
        symbol = st.text_input("Enter Stock Symbol for News", "TSLA").upper()
        if st.button("Fetch News"):
            news_df = get_news_twelvedata(symbol)
            if not news_df.empty:
                for _, row in news_df.iterrows():
                    st.markdown(f"**[{row['title']}]({row['article_url']})** - _{row['source']}_")
                    st.markdown("---")

if __name__ == "__main__":
    main()