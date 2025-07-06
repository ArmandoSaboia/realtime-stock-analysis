import requests
from kafka import KafkaProducer
import json
import os
from twelvedata import TDClient

class TwelveDataAPI:
    def __init__(self):
        self.api_key = os.getenv("TWELVEDATA_API_KEY")
        if not self.api_key:
            raise ValueError("Twelve Data API key is not set. Please set TWELVE_DATA_API_KEY environment variable.")
        self.td = TDClient(apikey=self.api_key)

    def get_time_series(self, symbol, interval, outputsize=30, **kwargs):
        ts = self.td.time_series(symbol=symbol, interval=interval, outputsize=outputsize, **kwargs)
        return ts.as_pandas()

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
