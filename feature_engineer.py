import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_functions = {
            'price_momentum': self.calculate_price_momentum,
            'volatility': self.calculate_volatility,
            'volume_features': self.calculate_volume_features,
            'trend_indicators': self.calculate_trend_indicators
        }

    def calculate_price_momentum(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate price momentum indicators"""
        momentum = {}

        # Price changes over different periods
        for period in [1, 5, 10, 20]:
            momentum[f'return_{period}d'] = data['close'].pct_change(period)

        # Rate of change
        momentum['rate_of_change'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5) * 100

        return momentum

    def calculate_volatility(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility indicators"""
        volatility = {}

        # Historical volatility
        for window in [5, 10, 20]:
            volatility[f'volatility_{window}d'] = data['close'].pct_change().rolling(window).std()

        return volatility

    def calculate_volume_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based features"""
        volume = {}

        # Volume momentum
        volume['volume_momentum'] = data['volume'].pct_change()

        # Volume moving averages
        for period in [5, 10, 20]:
            volume[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()

        return volume

    def calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate trend indicators"""
        trends = {}

        # Moving average crossovers
        ma_short = data['close'].rolling(10).mean()
        ma_long = data['close'].rolling(30).mean()
        trends['ma_crossover'] = (ma_short > ma_long).astype(int)

        return trends

    def engineer_features(self, data: pd.DataFrame, feature_sets: Optional[List[str]] = None) -> pd.DataFrame:
        """Engineer features for the dataset"""
        try:
            if feature_sets is None:
                feature_sets = list(self.feature_functions.keys())

            engineered_data = data.copy()

            for feature_set in feature_sets:
                if feature_set in self.feature_functions:
                    features = self.feature_functions[feature_set](engineered_data)
                    for name, feature in features.items():
                        engineered_data[name] = feature

            # Remove any infinite values
            engineered_data = engineered_data.replace([np.inf, -np.inf], np.nan)

            # Forward fill NaN values
            engineered_data = engineered_data.fillna(method='ffill')

            return engineered_data

        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return data