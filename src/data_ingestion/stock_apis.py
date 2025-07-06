import requests
from kafka import KafkaProducer
import json
import os
from twelvedata import TDClient

class TwelveDataAPI:
    def __init__(self):
        self.api_key = os.getenv("TWELVEDATA_API_KEY")
        if not self.api_key:
            raise ValueError("Twelve Data API key is not set. Please set TWELVEDATA_API_KEY environment variable.")
        self.td = TDClient(apikey=self.api_key)

    def get_time_series(self, symbol, interval, outputsize=30, **kwargs):
        ts = self.td.time_series(symbol=symbol, interval=interval, outputsize=outputsize, **kwargs)
        return ts.as_pandas()

    def get_time_series_object(self, symbol, interval, outputsize=30, **kwargs):
        # Returns the raw time_series object for chaining indicators or charting
        return self.td.time_series(symbol=symbol, interval=interval, outputsize=outputsize, **kwargs)

    def get_quote(self, symbol):
        quote = self.td.quote(symbol=symbol)
        return quote.as_json()

    def get_forex_quote(self, symbol):
        forex_quote = self.td.forex_quote(symbol=symbol)
        return forex_quote.as_json()

    def get_cryptocurrency_quote(self, symbol):
        crypto_quote = self.td.cryptocurrency_quote(symbol=symbol)
        return crypto_quote.as_json()

    def get_news(self, symbol, outputsize=10):
        news = self.td.news(symbol=symbol, outputsize=outputsize)
        return news.as_json()

    def get_technical_indicator(self, symbol, indicator_name, interval, **kwargs):
        ts = self.td.time_series(symbol=symbol, interval=interval)
        
        if indicator_name == "SMA":
            return ts.with_sma(**kwargs).as_pandas()
        elif indicator_name == "EMA":
            return ts.with_ema(**kwargs).as_pandas()
        elif indicator_name == "RSI":
            return ts.with_rsi(**kwargs).as_pandas()
        elif indicator_name == "MACD":
            return ts.with_macd(**kwargs).as_pandas()
        else:
            raise ValueError(f"Unsupported technical indicator: {indicator_name}")

    def search_symbols(self, query):
        return self.td.search(symbol=query).as_json()

    def get_api_usage(self):
        return self.td.api_usage().as_json()

    # Fundamental Data methods
    def get_logo(self, symbol, **kwargs):
        return self.td.get_logo(symbol=symbol, **kwargs).as_json()

    def get_profile(self, symbol, **kwargs):
        return self.td.get_profile(symbol=symbol, **kwargs).as_json()

    def get_dividends(self, symbol, **kwargs):
        return self.td.get_dividends(symbol=symbol, **kwargs).as_json()

    def get_splits(self, symbol, **kwargs):
        return self.td.get_splits(symbol=symbol, **kwargs).as_json()

    def get_earnings(self, symbol, **kwargs):
        return self.td.get_earnings(symbol=symbol, **kwargs).as_json()

    def get_earnings_calendar(self, symbol=None, **kwargs):
        return self.td.get_earnings_calendar(symbol=symbol, **kwargs).as_json()

    def get_ipo_calendar(self, **kwargs):
        return self.td.get_ipo_calendar(**kwargs).as_json()

    def get_statistics(self, symbol, **kwargs):
        return self.td.get_statistics(symbol=symbol, **kwargs).as_json()

    def get_insider_transactions(self, symbol, **kwargs):
        return self.td.get_insider_transactions(symbol=symbol, **kwargs).as_json()

    def get_income_statement(self, symbol, **kwargs):
        return self.td.get_income_statement(symbol=symbol, **kwargs).as_json()

    def get_balance_sheet(self, symbol, **kwargs):
        return self.td.get_balance_sheet(symbol=symbol, **kwargs).as_json()

    def get_cash_flow(self, symbol, **kwargs):
        return self.td.get_cash_flow(symbol=symbol, **kwargs).as_json()

    def get_options_expiration(self, symbol, **kwargs):
        return self.td.get_options_expiration(symbol=symbol, **kwargs).as_json()

    def get_options_chain(self, symbol, **kwargs):
        return self.td.get_options_chain(symbol=symbol, **kwargs).as_json()

    def get_key_executives(self, symbol, **kwargs):
        return self.td.get_key_executives(symbol=symbol, **kwargs).as_json()

    def get_institutional_holders(self, symbol, **kwargs):
        return self.td.get_institutional_holders(symbol=symbol, **kwargs).as_json()

    def get_fund_holders(self, symbol, **kwargs):
        return self.td.get_fund_holders(symbol=symbol, **kwargs).as_json()

class KafkaProducerWrapper:
    def __init__(self):
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_message(self, topic, message):
        self.producer.send(topic, value=message)
        self.producer.flush()