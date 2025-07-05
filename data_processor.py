import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamProcessor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_window = deque(maxlen=window_size)
        self.volume_window = deque(maxlen=window_size)
        self.last_processed = None

    def process_tick(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming tick data and compute metrics."""
        try:
            # Validate input
            if not all(k in tick_data for k in ["timestamp", "price", "volume"]):
                raise ValueError("Missing required tick data fields")
            timestamp = datetime.fromtimestamp(tick_data["timestamp"])
            price = float(tick_data["price"])
            volume = float(tick_data["volume"])
            # Update sliding windows
            self.price_window.append(price)
            self.volume_window.append(volume)
            # Calculate real-time metrics
            metrics = self.calculate_metrics()
            # Update last processed time
            self.last_processed = timestamp
            return {
                "status": "success",
                "timestamp": timestamp.isoformat(),
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            return {"status": "error", "error": str(e)}

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate real-time market metrics."""
        try:
            if len(self.price_window) < 2:
                return {"status": "error", "error": "Not enough data"}
            price_array = np.array(self.price_window, dtype=np.float64)
            volume_array = np.array(self.volume_window, dtype=np.float64)
            metrics = {
                "current_price": price_array[-1],
                "price_change": price_array[-1] - price_array[0],
                "price_change_pct": ((price_array[-1] - price_array[0]) / price_array[0]) * 100,
                "price_ma": np.mean(price_array),
                "price_std": np.std(price_array),
                "volume_ma": np.mean(volume_array),
                "volume_std": np.std(volume_array),
            }
            # Include technical indicators if enough data is available
            if len(price_array) >= 14:
                metrics.update(self.calculate_technical_indicators(price_array))
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {"status": "error", "error": str(e)}

    def calculate_technical_indicators(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate RSI and Bollinger Bands for technical analysis."""
        try:
            if len(prices) < 14:
                return {}
            # Relative Strength Index (RSI)
            deltas = np.diff(prices)
            gains = np.maximum(deltas, 0)
            losses = -np.minimum(deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
            rsi = 100 - (100 / (1 + rs))
            # Bollinger Bands (20-period SMA with ±2 standard deviations)
            if len(prices) >= 20:
                ma20 = np.mean(prices[-20:])
                std20 = np.std(prices[-20:])
                upper_band = ma20 + (2 * std20)
                lower_band = ma20 - (2 * std20)
            else:
                ma20, std20, upper_band, lower_band = np.nan, np.nan, np.nan, np.nan
            return {
                "rsi": rsi,
                "bollinger_middle": ma20,
                "bollinger_upper": upper_band,
                "bollinger_lower": lower_band,
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {"status": "error", "error": str(e)}

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current market state."""
        try:
            if not self.price_window:
                return {"status": "error", "error": "No data available"}
            metrics = self.calculate_metrics()
            last_update = self.last_processed.isoformat() if self.last_processed else None
            return {
                "status": "success",
                "last_update": last_update,
                "window_size": self.window_size,
                "data_points": len(self.price_window),
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {"status": "error", "error": str(e)}

    def clear_data(self):
        """Clear all stored data."""
        self.price_window.clear()
        self.volume_window.clear()
        self.last_processed = None
        logger.info("Stream processor data cleared")

