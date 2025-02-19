import unittest
import numpy as np
from src.model_training.train_model import train_model
from src.prediction.predict import predict

class TestModelTraining(unittest.TestCase):
    def test_train_and_predict(self):
        X_train = [[1], [2], [3], [4]]
        y_train = [0, 1, 0, 1]
        train_model(X_train, y_train, model_path="test_model.pkl")
        predictions = predict(X_train, model_path="test_model.pkl")
        self.assertIsInstance(predictions[0], np.int64)

if __name__ == "__main__":
    unittest.main()