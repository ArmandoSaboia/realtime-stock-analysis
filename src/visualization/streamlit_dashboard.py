import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from src.genai.langchain_insights import generate_insights
from src.genai.config import config

# Function to parse Finnhub JSON responses
def parse_finnhub_data(json_data):
    """
    Parse Finnhub JSON result into a DataFrame.
    Supports both Time Series and Technical Indicator data.
    """
    # Look for keys starting with "Time Series"
    for key in json_data.keys():
        if key.startswith("Time Series"):
            data_dict = json_data[key]
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            # For keys like "1. open", get last part as column name.
            df.rename(columns=lambda col: col.split(". ")[-1] if ". " in col else col, inplace=True)
            return df
    # Look for keys starting with "Technical Analysis" (e.g., for SMA, EMA, MACD, RSI)
    for key in json_data.keys():
        if key.startswith("Technical Analysis"):
            data_dict = json_data[key]
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            return df
    return pd.DataFrame()

# Generic function to fetch data from Finnhub
def get_finnhub_data(av_function, symbol=None, **kwargs):
    """
    Retrieve data for any Finnhub function.
    The function parameter allows you to specify the endpoint (e.g., TIME_SERIES_INTRADAY, SMA, MACD, etc.).
    Additional parameters for the API endpoint should be passed in via kwargs.
    """
    api_key = config.get("finnhub", {}).get("api_key")
    if not api_key or api_key == "REPLACE_WITH_ACTUAL_KEY":
        st.error("Finnhub API key is not set. Please update config/config.yaml.")
        return pd.DataFrame()
    
    url = "https://www.finnhub.io/query"
    params = {
        "function": av_function,
        "apikey": api_key,
    }
    if symbol:
        params["symbol"] = symbol
    params.update(kwargs)
    
    response = requests.get(url, params=params)
    json_data = response.json()
    
    df = parse_finnhub_data(json_data)
    if df.empty:
        st.error("No data found or error retrieving data. Response: " + str(json_data))
    return df

# Main function
def main():
    st.title("Interactive Stock Data, Prediction & News Dashboard")
    
    # Sidebar options
    data_option = st.sidebar.radio(
        "Select Option", 
        (
            "Stock Data", 
            "Stock Prediction", 
            "News Insights", 
            "Finnhub Functions", 
            "Forex/Crypto Data", 
            "Economic Indicators"
        )
    )
    
    if data_option == "Stock Data":
        ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
        if st.sidebar.button("Get Stock Data"):
            st.subheader(f"Stock Data for {ticker.upper()}")
            data = get_finnhub_data("TIME_SERIES_DAILY", symbol=ticker)
            if data.empty:
                st.write("No data found for ticker:", ticker)
            else:
                st.line_chart(data["close"])
                st.subheader("Key Metrics")
                st.metric("Average Close Price", round(data["close"].mean(), 2))
                st.metric("Max High Price", round(data["high"].max(), 2))
                st.metric("Min Low Price", round(data["low"].min(), 2))
    
    elif data_option == "Stock Prediction":
        ticker = st.sidebar.text_input("Enter stock ticker for prediction (e.g., AAPL)", value="AAPL")
        days = st.sidebar.number_input("Days ahead to predict", min_value=1, max_value=30, value=1)
        if st.sidebar.button("Predict Stock Price"):
            st.subheader(f"Stock Prediction for {ticker.upper()}")
            data = get_finnhub_data("TIME_SERIES_DAILY", symbol=ticker)
            if data.empty:
                st.write("No data found for ticker:", ticker)
            else:
                if len(data) < 30:
                    st.error("Not enough data to make an accurate prediction")
                else:
                    X = np.arange(len(data)).reshape(-1, 1)
                    y = data["close"].values
                    model = LinearRegression()
                    model.fit(X, y)
                    future_index = np.array([[len(data) + days - 1]])
                    predicted_price = model.predict(future_index)[0]
                    st.metric("Predicted Close Price", round(predicted_price, 2))
                    
                    fig, ax = plt.subplots()
                    ax.plot(data.index, data["close"], label="Historical Close")
                    last_date = data.index[-1]
                    future_date = last_date + pd.Timedelta(days=days)
                    ax.plot(future_date, predicted_price, "ro", label="Predicted Price")
                    ax.set_title(f"{ticker.upper()} Close Price Prediction")
                    ax.legend()
                    st.pyplot(fig)
    
    elif data_option == "News Insights":
        user_query = st.sidebar.text_input("Enter your news query about stocks", value="Latest technology news")
        if st.sidebar.button("Get News Insights"):
            st.subheader("News Insights")
            with st.spinner("Generating insights..."):
                insights = generate_insights(user_query)
                st.write(insights)
    
    elif data_option == "Finnhub Functions":
        av_function = st.sidebar.selectbox(
            "Select Finnhub Function",
            options=[
                "TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED",
                "TIME_SERIES_WEEKLY", "TIME_SERIES_MONTHLY", "SMA", "EMA", "MACD", "RSI"
            ]
        )
        symbol = st.sidebar.text_input("Enter stock ticker", value="AAPL")
        extra_params = {}
        if av_function == "TIME_SERIES_INTRADAY":
            interval = st.sidebar.selectbox("Select interval", options=["1min", "5min", "15min", "30min", "60min"])
            extra_params["interval"] = interval
        if av_function in ["SMA", "EMA", "RSI"]:
            time_period = st.sidebar.number_input("Time period", min_value=1, max_value=100, value=20)
            extra_params["time_period"] = time_period
            extra_params["series_type"] = st.sidebar.selectbox("Series Type", options=["close", "open", "high", "low"])
        if av_function == "MACD":
            interval = st.sidebar.selectbox("Select interval for MACD", options=["1min", "5min", "15min", "30min", "60min"])
            extra_params["interval"] = interval
            extra_params["series_type"] = st.sidebar.selectbox("Series Type for MACD", options=["close", "open", "high", "low"])
        if st.sidebar.button("Get Finnhub Data"):
            st.subheader(f"Finnhub Data: {av_function} for {symbol.upper()}")
            df = get_finnhub_data(av_function, symbol, **extra_params)
            if not df.empty:
                st.dataframe(df)
                st.line_chart(df)
    
    elif data_option == "Forex/Crypto Data":
        forex_crypto_option = st.sidebar.selectbox(
            "Select Forex/Crypto Option",
            options=["CURRENCY_EXCHANGE_RATE", "CRYPTO_RATING"]
        )
        if forex_crypto_option == "CURRENCY_EXCHANGE_RATE":
            from_currency = st.sidebar.text_input("From Currency (e.g., USD)", value="USD")
            to_currency = st.sidebar.text_input("To Currency (e.g., EUR)", value="EUR")
            if st.sidebar.button("Get Exchange Rate"):
                st.subheader(f"Exchange Rate: {from_currency}/{to_currency}")
                data = get_finnhub_data("CURRENCY_EXCHANGE_RATE", from_currency=from_currency, to_currency=to_currency)
                if not data.empty:
                    st.json(data)
        elif forex_crypto_option == "CRYPTO_RATING":
            crypto_symbol = st.sidebar.text_input("Enter Crypto Symbol (e.g., BTC)", value="BTC")
            if st.sidebar.button("Get Crypto Rating"):
                st.subheader(f"Crypto Rating for {crypto_symbol.upper()}")
                data = get_finnhub_data("CRYPTO_RATING", symbol=crypto_symbol)
                if not data.empty:
                    st.json(data)
    
    elif data_option == "Economic Indicators":
        economic_indicator = st.sidebar.selectbox(
            "Select Economic Indicator",
            options=["REAL_GDP", "INFLATION", "UNEMPLOYMENT_RATE"]
        )
        if st.sidebar.button("Get Economic Data"):
            st.subheader(f"Economic Data: {economic_indicator}")
            data = get_finnhub_data(economic_indicator)
            if not data.empty:
                st.dataframe(data)
                st.line_chart(data)

if __name__ == "__main__":
    main()
