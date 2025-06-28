import unittest
from src.data_ingestion.finnhub_api import FinnhubAPI

class TestFinnhubAPI(unittest.TestCase):
    def test_get_stock_data(self):
        api = FinnhubAPI(api_key="demo")
        data = api.get_stock_data("AAPL")
        self.assertIn("Time Series (5min)", data)

if __name__ == "__main__":
    unittest.main()
