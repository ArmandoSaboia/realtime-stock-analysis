import unittest
import requests_mock
from src.data_ingestion.finnhub_api import FinnhubAPI

class TestFinnhubAPI(unittest.TestCase):
    @requests_mock.Mocker()
    def test_get_stock_data(self, m):
        m.get('https://www.finnhub.io/query', json={'Time Series (5min)': {}})
        api = FinnhubAPI()
        data = api.get_stock_data("AAPL")
        self.assertIn("Time Series (5min)", data)

if __name__ == "__main__":
    unittest.main()
