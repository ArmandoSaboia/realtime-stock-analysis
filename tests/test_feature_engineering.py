import unittest
import pandas as pd
from src.feature_engineering.feature_pipeline import calculate_moving_average

class TestFeatureEngineering(unittest.TestCase):
    def test_calculate_moving_average(self):
        data = pd.DataFrame({"close": [10, 20, 30, 40, 50]})
        result = calculate_moving_average(data, window=3)
        self.assertEqual(result["moving_average"].iloc[-1], 40)

if __name__ == "__main__":
    unittest.main()