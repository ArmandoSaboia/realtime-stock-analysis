import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.genai.langchain_insights import generate_insights
from src.genai.config import config
from src.data_ingestion.stock_apis import AlphaVantageAPI, TwelveDataAPI

# Initialize API clients
alphavantage_client = AlphaVantageAPI()
twelvedata_client = TwelveDataAPI()

# Main function
def main():
    st.title("Interactive Stock Data, Prediction & News Dashboard")
    
    # Sidebar options
    api_choice = st.sidebar.selectbox(
        "Select API for Stock Data",
        ("Alpha Vantage", "Twelve Data")
    )

    if api_choice == "Alpha Vantage":
        client = alphavantage_client
    elif api_choice == "Twelve Data":
        client = twelvedata_client

    data_option = st.sidebar.radio(
        "Select Option", 
        (
            "Stock Data", 
            "News Insights", 
            "Technical Indicators", 
            "Forex/Crypto Data", 
            "Economic Indicators"
        )
    )
    
    if data_option == "Stock Data":
        ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
        if st.sidebar.button("Get Stock Data"):
            st.subheader(f"Stock Data for {ticker.upper()} (via {api_choice})")
            try:
                if api_choice == "Alpha Vantage":
                    data = client.get_daily_stock_data(ticker)
                    if not data.empty:
                        st.line_chart(data["4. close"])
                        st.subheader("Key Metrics")
                        st.metric("Average Close Price", round(data["4. close"].mean(), 2))
                        st.metric("Max High Price", round(data["2. high"].max(), 2))
                        st.metric("Min Low Price", round(data["3. low"].min(), 2))
                    else:
                        st.write("No data found for ticker:", ticker)
                elif api_choice == "Twelve Data":
                    data = client.get_time_series(ticker, "1day", outputsize=90)
                    if data:
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
        st.subheader(f"Technical Indicators (via {api_choice})")
        ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
        indicator_choice = st.sidebar.selectbox(
            "Select Indicator",
            ("SMA", "EMA", "MACD", "RSI")
        )
        if st.sidebar.button("Get Indicator Data"):
            try:
                if api_choice == "Alpha Vantage":
                    data = client.get_technical_indicator(ticker, indicator_choice)
                    if not data.empty:
                        st.dataframe(data)
                        st.line_chart(data)
                    else:
                        st.write("No data found for indicator:", indicator_choice)
                elif api_choice == "Twelve Data":
                    st.write("Technical indicators not directly supported via Twelve Data client in this demo. Please refer to their API docs.")
            except ValueError as e:
                st.error(f"API Key Error: {e}")
            except Exception as e:
                st.error(f"Error fetching technical indicator: {e}")

    elif data_option == "Forex/Crypto Data":
        st.subheader(f"Forex/Crypto Data (via {api_choice})")
        currency_pair = st.sidebar.text_input("Enter currency pair (e.g., EUR/USD for Twelve Data, USD for Alpha Vantage)", value="EUR/USD")
        if st.sidebar.button("Get Data"):
            try:
                if api_choice == "Alpha Vantage":
                    from_currency, to_currency = currency_pair.split("/")
                    data = client.get_currency_exchange_rate(from_currency, to_currency)
                    if not data.empty:
                        st.json(data)
                    else:
                        st.write("No data found for currency pair:", currency_pair)
                elif api_choice == "Twelve Data":
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
        st.subheader(f"Economic Indicators (via {api_choice})")
        indicator_choice = st.sidebar.selectbox(
            "Select Indicator",
            ("REAL_GDP", "INFLATION", "UNEMPLOYMENT_RATE")
        )
        if st.sidebar.button("Get Economic Data"):
            try:
                if api_choice == "Alpha Vantage":
                    data = client.get_economic_indicator(indicator_choice)
                    if not data.empty:
                        st.dataframe(data)
                    else:
                        st.write("No data found for indicator:", indicator_choice)
                elif api_choice == "Twelve Data":
                    st.write("Economic indicators not directly supported via Twelve Data client in this demo. Please refer to their API docs.")
            except ValueError as e:
                st.error(f"API Key Error: {e}")
            except Exception as e:
                st.error(f"Error fetching economic data: {e}")

if __name__ == "__main__":
    main()