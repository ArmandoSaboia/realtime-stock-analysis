# src/visualization/streamlit_dashboard.py

import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from twelvedata import TDClient

# Load environment variables from a .env file
load_dotenv()

def handle_symbol_search(td_client):
    """
    Handles the UI and logic for the Symbol Search feature.
    """
    st.subheader("Symbol Search")
    st.write("Find stock symbols, forex pairs, crypto, ETFs, and more.")
    
    query = st.text_input("Enter search query (e.g., 'Apple', 'EUR/USD', 'BTC')", "AAPL")

    if st.button("Search"):
        if not query:
            st.warning("Please enter a search query.")
            return
        try:
            with st.spinner("Searching..."):
                results = td_client.search(symbol=query).as_json()
                if results and len(results) > 0:
                    df = pd.DataFrame(results)
                    st.success(f"Found {len(df)} results for '{query}'.")
                    st.dataframe(df)
                else:
                    st.info(f"No results found for '{query}'.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def handle_core_data(td_client):
    """
    Handles the UI and logic for core data endpoints like Price, Quote, etc.
    """
    st.subheader("Core Market Data")
    
    core_options = [
        "Real-time Price", "End-of-Day (EOD) Price", "Quote", 
        "Exchange Rate", "Currency Conversion"
    ]
    choice = st.selectbox("Select Data Type", core_options)

    symbol = st.text_input("Enter Symbol (e.g., 'AAPL', 'EUR/USD')", "AAPL")

    try:
        if choice == "Real-time Price":
            if st.button("Get Price"):
                with st.spinner("Fetching price..."):
                    price = td_client.price(symbol=symbol).as_json()
                    st.json(price)

        elif choice == "End-of-Day (EOD) Price":
            if st.button("Get EOD Price"):
                 with st.spinner("Fetching EOD..."):
                    eod = td_client.eod(symbol=symbol).as_json()
                    st.json(eod)

        elif choice == "Quote":
            if st.button("Get Quote"):
                with st.spinner("Fetching quote..."):
                    quote = td_client.quote(symbol=symbol).as_json()
                    st.json(quote)

        elif choice == "Exchange Rate":
            st.info("Use a symbol format like 'USD/JPY'.")
            if st.button("Get Exchange Rate"):
                with st.spinner("Fetching exchange rate..."):
                    rate = td_client.exchange_rate(symbol=symbol).as_json()
                    st.json(rate)

        elif choice == "Currency Conversion":
            st.info("Use a symbol format like 'USD/JPY'.")
            amount = st.number_input("Enter Amount to Convert", value=100.0)
            if st.button("Convert Currency"):
                with st.spinner("Performing conversion..."):
                    conversion = td_client.currency_conversion(
                        symbol=symbol, amount=amount
                    ).as_json()
                    st.json(conversion)
    except Exception as e:
        st.error(f"An error occurred: {e}")


def handle_fundamentals(td_client):
    """
    Handles the UI and logic for all fundamental data endpoints.
    """
    st.subheader("Fundamental Data")
    
    fund_options = [
        "Logo", "Profile", "Dividends", "Splits", "Earnings", 
        "Earnings Calendar", "IPO Calendar", "Statistics", 
        "Insider Transactions", "Income Statement", "Balance Sheet", 
        "Cash Flow", "Key Executives", "Institutional Holders", "Fund Holders"
    ]
    choice = st.selectbox("Select Fundamental Data Type", fund_options)

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", "AAPL")
    with col2:
        exchange = st.text_input("Exchange (optional)", "NASDAQ")
    with col3:
        country = st.text_input("Country (optional)", "USA")

    try:
        if st.button(f"Get {choice}"):
            with st.spinner(f"Fetching {choice}..."):
                if choice == "Logo":
                    data = td_client.get_logo(symbol=symbol, exchange=exchange, country=country).as_json()
                    if data.get('url'):
                        st.image(data['url'], caption=f"Logo for {symbol}")
                    else:
                        st.warning("Logo not found.")
                
                else:
                    method_name = f"get_{choice.lower().replace(' ', '_')}"
                    api_call = getattr(td_client, method_name)
                    data = api_call(symbol=symbol, exchange=exchange, country=country).as_json()
                    st.json(data)

    except Exception as e:
        st.error(f"An error occurred: {e}")

def handle_options_data(td_client):
    """
    Handles the UI and logic for options data endpoints.
    """
    st.subheader("Options Data")
    
    options_type = st.radio("Select Options Data", ["Options Expiration", "Options Chain"])
    
    symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    try:
        if options_type == "Options Expiration":
            if st.button("Get Expiration Dates"):
                with st.spinner("Fetching expiration dates..."):
                    expirations = td_client.get_options_expiration(symbol=symbol).as_json()
                    if expirations.get('dates'):
                        st.success(f"Found {len(expirations['dates'])} expiration dates.")
                        st.json(expirations['dates'])
                    else:
                        st.warning("No expiration dates found.")

        elif options_type == "Options Chain":
            st.write("First, fetch expiration dates to select one for the chain.")
            if 'exp_dates' not in st.session_state:
                st.session_state.exp_dates = []

            if st.button("Fetch Expiration Dates for Chain"):
                 with st.spinner("Fetching expiration dates..."):
                    expirations = td_client.get_options_expiration(symbol=symbol).as_json()
                    if expirations.get('dates'):
                        st.session_state.exp_dates = expirations['dates']
                        st.success(f"Found {len(st.session_state.exp_dates)} dates. Select one below.")
                    else:
                        st.session_state.exp_dates = []
                        st.warning("No dates found.")

            if st.session_state.exp_dates:
                expiration_date = st.selectbox("Select Expiration Date", st.session_state.exp_dates)
                side = st.radio("Select Option Side", ["put", "call"], horizontal=True)
                
                if st.button("Get Options Chain"):
                    with st.spinner(f"Fetching {side} options..."):
                        chain = td_client.get_options_chain(
                            symbol=symbol, 
                            expiration_date=expiration_date,
                            side=side
                        ).as_json()
                        
                        if chain.get(f'{side}s'):
                            df = pd.DataFrame(chain[f'{side}s'])
                            st.dataframe(df)
                        else:
                            st.warning(f"No {side}s found for the selected criteria.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

def handle_time_series(td_client):
    """
    Handles the UI and logic for Time Series and Technical Indicators.
    """
    st.subheader("Time Series & Technical Indicators")
    st.write("Fetch historical data with optional technical indicators and charts.")

    st.info("For batch requests, separate symbols with a comma (e.g., 'AAPL,MSFT,GOOG').")
    symbol_input = st.text_input("Symbol(s)", "AAPL")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        interval = st.selectbox("Interval", ["1min", "5min", "15min", "30min", "45min", "1h", "2h", "4h", "1day", "1week", "1month"])
    with col2:
        output_size = st.number_input("Output Size", min_value=1, max_value=5000, value=100)
    with col3:
        timezone = st.text_input("Timezone", "America/New_York")

    st.write("**Technical Indicators**")
    indicator_options = [
        "adx", "aroon", "bbands", "ema", "macd", "rsi", "stoch", "sma", "wma"
    ]
    selected_indicators = st.multiselect("Select indicators to apply", indicator_options)
    
    output_format = st.radio(
        "Select Output Format", 
        ("Pandas DataFrame", "Interactive Chart (Plotly)", "JSON"),
        horizontal=True
    )
    
    include_ohlc = st.checkbox("Include OHLCV data", True)

    if st.button("Get Time Series Data"):
        if not symbol_input:
            st.warning("Please enter at least one symbol.")
            return

        try:
            with st.spinner("Fetching time series data..."):
                ts = td_client.time_series(
                    symbol=symbol_input,
                    interval=interval,
                    outputsize=output_size,
                    timezone=timezone,
                )

                for indicator in selected_indicators:
                    ts = getattr(ts, f"with_{indicator}")()
                
                if not include_ohlc:
                    ts = ts.without_ohlc()

                if output_format == "JSON":
                    st.write("### JSON Output")
                    data = ts.as_json()
                    st.json(data)
                
                elif output_format == "Pandas DataFrame":
                    st.write("### Pandas DataFrame Output")
                    df = ts.as_pandas()
                    st.dataframe(df)

                elif output_format == "Interactive Chart (Plotly)":
                    st.write("### Interactive Chart")
                    if ',' in symbol_input:
                        st.warning("Plotly charts are best viewed with a single symbol. Only the first symbol may be plotted.")
                    
                    fig = ts.as_plotly_figure()
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

def handle_advanced_tools(td_client):
    """
    Handles UI for advanced features like custom endpoints, URL debugging, and API usage.
    """
    st.subheader("Advanced Tools")
    
    adv_choice = st.selectbox("Select an Advanced Tool", ["API Usage", "Custom Endpoint", "Debug Request URL"])

    if adv_choice == "API Usage":
        st.write("Check your current API credit consumption.")
        if st.button("Get API Usage"):
            try:
                with st.spinner("Fetching API usage..."):
                    usage = td_client.api_usage().as_json()
                    st.json(usage)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif adv_choice == "Custom Endpoint":
        st.write("Request an endpoint not explicitly available in the library.")
        endpoint_name = st.text_input("Endpoint Name (e.g., 'quote')", "quote")
        st.write("Enter parameters as a JSON object:")
        params_text = st.text_area("Parameters", '{"symbol": "AAPL", "interval": "1day"}')
        
        if st.button("Call Custom Endpoint"):
            try:
                params = json.loads(params_text)
                with st.spinner(f"Calling endpoint '{endpoint_name}'..."):
                    endpoint = td_client.custom_endpoint(name=endpoint_name, **params)
                    data = endpoint.as_json()
                    st.json(data)
            except json.JSONDecodeError:
                st.error("Invalid JSON in parameters. Please check the format.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif adv_choice == "Debug Request URL":
        st.write("Build a time series request and see the generated API URL for debugging.")
        symbol = st.text_input("Symbol", "AAPL")
        interval = st.selectbox("Interval", ["1min", "5min", "1day"])
        indicator = st.selectbox("Add a sample indicator to the URL", ["ema", "rsi", "None"])

        if st.button("Generate Debug URL"):
            with st.spinner("Generating URL..."):
                ts = td_client.time_series(symbol=symbol, interval=interval)
                if indicator != "None":
                    ts = getattr(ts, f"with_{indicator}")()
                
                urls = ts.as_url()
                st.write("### Generated URL(s)")
                st.code(urls, language="json")

def handle_mcp_guide():
    """
    Displays a guide for setting up and using the MCP Server and U-Tool.
    """
    st.subheader("MCP & U-Tool Guide")
    st.write("The MCP (Model-Context Protocol) Server allows you to connect tools like Claude Desktop and VS Code directly to the Twelve Data API.")

    st.markdown("---")
    st.markdown("### MCP Functions")
    st.write("The server exposes several core functions:")
    st.json({
        "time_series": "Fetch historical price data.",
        "price": "Get the latest price.",
        "stocks": "List available stocks.",
        "forex_pairs": "List available forex pairs.",
        "cryptocurrencies": "List available cryptocurrencies."
    })

    st.markdown("---")
    st.markdown("### U-Tool: Natural Language for Financial Data")
    st.write("The `u-tool` is an AI-powered router that lets you query the Twelve Data API using plain English. It uses GPT-4o to find the right endpoint and parameters for your request.")
    st.info('**Example questions:** "Show me Apple stock performance this week" or "Calculate RSI for Bitcoin"')
    st.write("**Setup for U-Tool:**")
    st.code("""
{
  "mcpServers": {
    "twelvedata": {
      "command": "uvx",
      "args": ["mcp-server-twelve-data@latest", "-k", "TWELVEDATA_API_KEY", "-u", "GROQ_API_KEY"]
    }
  }
}
    """, language="json")


    st.markdown("---")
    st.markdown("### Configuration Snippets")

    st.markdown("#### Claude Desktop Integration")
    st.code("""
{
  "mcpServers": {
    "twelvedata": {
      "command": "uvx",
      "args": ["mcp-server-twelve-data@latest", "-k", "TWELVEDATA_API_KEY"]
    }
  }
}
    """, language="json")

    st.markdown("#### VS Code Integration (Manual Setup)")
    st.code("""
{
  "mcp": {
    "servers": {
      "twelvedata": {
        "command": "uvx",
        "args": [
          "mcp-server-twelve-data@latest",
          "-t", "streamable-http",
          "-k", "TWELVEDATA_API_KEY"
        ]
      }
    }
  }
}
    """, language="json")

    st.markdown("#### Docker Usage")
    st.code("""
# Build the image
docker build -t mcp-server-twelve-data .

# Run the server
docker run --rm mcp-server-twelve-data -k TWELVEDATA_API_KEY
    """, language="bash")


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Twelve Data Dashboard", layout="wide")
    st.title(" Twelve Data API Dashboard")

    st.sidebar.header("Configuration")
    # Use the environment variable as the default, but allow user to override
    api_key_from_env = os.getenv("TWELVEDATA_API_KEY")
    api_key = st.sidebar.text_input(
        "Enter your Twelve Data API Key", 
        type="password", 
        value=api_key_from_env
    )

    if not api_key:
        st.warning("Please enter your API key in the sidebar to begin.")
        st.stop()

    try:
        td = TDClient(apikey=api_key)
    except Exception as e:
        st.error(f"Failed to initialize API client: {e}")
        st.stop()

    st.sidebar.header("Features")
    app_mode = st.sidebar.selectbox(
        "Choose a feature",
        [
            "Symbol Search", 
            "Time Series & Indicators", 
            "Core Market Data", 
            "Fundamental Data", 
            "Options Data",
            "Advanced Tools",
            "MCP & U-Tool Guide"
        ]
    )

    # Route to the correct handler based on user selection
    if app_mode == "Symbol Search":
        handle_symbol_search(td)
    elif app_mode == "Time Series & Indicators":
        handle_time_series(td)
    elif app_mode == "Core Market Data":
        handle_core_data(td)
    elif app_mode == "Fundamental Data":
        handle_fundamentals(td)
    elif app_mode == "Options Data":
        handle_options_data(td)
    elif app_mode == "Advanced Tools":
        handle_advanced_tools(td)
    elif app_mode == "MCP & U-Tool Guide":
        handle_mcp_guide()


if __name__ == "__main__":
    main()
