import requests
from kafka import KafkaProducer
import json
import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.fundamentaldata import FundamentalData
from twelvedata import TDClient

class FinnhubAPI:
    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY")
        if self.api_key and self.api_key.startswith("token="):
            self.api_key = self.api_key.replace("token=", "")
        self.base_url = "https://finnhub.io/api/v1"

    def _make_request(self, endpoint, params=None):
        if not self.api_key:
            raise ValueError("Finnhub API key is not set. Please set FINNHUB_API_KEY environment variable.")
        
        full_url = f"{self.base_url}{endpoint}"
        all_params = {"token": self.api_key}
        if params:
            all_params.update(params)
        
        response = requests.get(full_url, params=all_params)
        response.raise_for_status()
        return response.json()

    def get_stock_candles(self, symbol, resolution, _from, to):
        endpoint = "/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": _from,
            "to": to
        }
        return self._make_request(endpoint, params)

    def get_quote(self, symbol):
        endpoint = "/quote"
        params = {"symbol": symbol}
        return self._make_request(endpoint, params)

    def get_company_news(self, symbol, _from, to):
        endpoint = "/news"
        params = {
            "symbol": symbol,
            "from": _from,
            "to": to
        }
        return self._make_request(endpoint, params)

    def get_forex_rates(self, base, quote):
        endpoint = "/forex/rates"
        params = {
            "base": base,
            "to": quote
        }
        return self._make_request(endpoint, params)

    def get_economic_calendar(self, _from, to):
        endpoint = "/calendar/economic"
        params = {
            "from": _from,
            "to": to
        }
        return self._make_request(endpoint, params)

class AlphaVantageAPI:
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is not set. Please set ALPHA_VANTAGE_API_KEY environment variable.")
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.fx = ForeignExchange(key=self.api_key, output_format='pandas')
        self.ti = TechIndicators(key=self.api_key, output_format='pandas')
        self.fd = FundamentalData(key=self.api_key, output_format='pandas')

    def get_daily_stock_data(self, symbol):
        data, meta_data = self.ts.get_daily(symbol=symbol, outputtype='full')
        return data

    def get_intraday_stock_data(self, symbol, interval='5min'):
        data, meta_data = self.ts.get_intraday(symbol=symbol, interval=interval, outputtype='full')
        return data

    def get_currency_exchange_rate(self, from_currency, to_currency):
        data, meta_data = self.fx.get_currency_exchange_rate(from_currency=from_currency, to_currency=to_currency)
        return data

    def get_technical_indicator(self, symbol, function, interval='daily', time_period=60, series_type='close'):
        if function == 'SMA':
            data, meta_data = self.ti.get_sma(symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
        elif function == 'EMA':
            data, meta_data = self.ti.get_ema(symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
        elif function == 'MACD':
            data, meta_data = self.ti.get_macd(symbol=symbol, interval=interval, series_type=series_type)
        elif function == 'RSI':
            data, meta_data = self.ti.get_rsi(symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
        else:
            raise ValueError("Unsupported technical indicator function for Alpha Vantage.")
        return data

    def get_economic_indicator(self, indicator):
        if indicator == 'REAL_GDP':
            data, meta_data = self.fd.get_real_gdp()
        elif indicator == 'INFLATION':
            data, meta_data = self.fd.get_inflation()
        elif indicator == 'UNEMPLOYMENT_RATE':
            data, meta_data = self.fd.get_unemployment_rate()
        else:
            raise ValueError("Unsupported economic indicator for Alpha Vantage.")
        return data

class TwelveDataAPI:
    def __init__(self):
        self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        if not self.api_key:
            raise ValueError("Twelve Data API key is not set. Please set TWELVE_DATA_API_KEY environment variable.")
        self.td = TDClient(apikey=self.api_key)

    def get_time_series(self, symbol, interval, outputsize=30, datatype='json'):
        ts = self.td.time_series(symbol=symbol, interval=interval, outputsize=outputsize, datatype=datatype)
        return ts.as_json()

    def get_quote(self, symbol):
        quote = self.td.quote(symbol=symbol)
        return quote.as_json()

    def get_forex_quote(self, symbol):
        forex_quote = self.td.forex_quote(symbol=symbol)
        return forex_quote.as_json()

    def get_cryptocurrency_quote(self, symbol):
        crypto_quote = self.td.cryptocurrency_quote(symbol=symbol)
        return crypto_quote.as_json()

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