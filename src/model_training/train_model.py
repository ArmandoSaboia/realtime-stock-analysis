# src/model_training/train_model.py
import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress specific warnings from statsmodels
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning, module='statsmodels')


def train_model(time_series_data: pd.Series, order: tuple = (5, 1, 0), model_path: str = "arima_model.pkl"):
    """
    Train an ARIMA model for time series forecasting.

    :param time_series_data: A pandas Series containing the time series data (e.g., historical closing prices).
                             The Series should have a DatetimeIndex.
    :param order: The (p, d, q) order of the ARIMA model.
                  p: order of the AR (AutoRegressive) part.
                  d: order of the I (Integrated) part (number of non-seasonal differences).
                  q: order of the MA (Moving Average) part.
    :param model_path: Path to save the trained ARIMA model.
    """
    if not isinstance(time_series_data, pd.Series) or not isinstance(time_series_data.index, pd.DatetimeIndex):
        raise ValueError("time_series_data must be a pandas Series with a DatetimeIndex.")

    print(f"Training ARIMA model with order {order}...")
    try:
        model = ARIMA(time_series_data, order=order)
        model_fit = model.fit()
        joblib.dump(model_fit, model_path)
        print(f"ARIMA model successfully saved to {model_path}")
    except Exception as e:
        print(f"Error during ARIMA model training or saving: {e}")
        raise
        
