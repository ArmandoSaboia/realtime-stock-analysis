import requests
from kafka import KafkaProducer
import json
from src.utils.helpers import load_secrets

class AlphaVantageAPI:
    def __init__(self):
        # Load API key from secrets.yaml
        secrets = load_secrets()
        self.api_key = secrets.get("alpha_vantage_api_key")
        self.base_url = "https://www.alphavantage.co/query"

    def get_stock_data(self, symbol, interval="5min"):
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params)
        return response.json()

class KafkaProducerWrapper:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_message(self, topic, message):
        self.producer.send(topic, value=message)
        self.producer.flush()