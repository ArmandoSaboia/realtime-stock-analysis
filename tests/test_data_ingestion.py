import unittest
from src.data_ingestion.alphavantage_api import AlphaVantageAPI

class TestAlphaVantageAPI(unittest.TestCase):
    def test_get_stock_data(self):
        api = AlphaVantageAPI(api_key="demo")
        data = api.get_stock_data("AAPL")
        self.assertIn("Time Series (5min)", data)

if __name__ == "__main__":
    unittest.main()