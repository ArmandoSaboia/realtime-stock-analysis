import requests
from kafka import KafkaProducer
import json
import os

class FinnhubAPI:
    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY")
        self.base_url = "https://www.finnhub.io/query"

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
    def __init__(self):
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_message(self, topic, message):
        self.producer.send(topic, value=message)
        self.producer.flush()
