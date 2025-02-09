import joblib

def predict(data, model_path="model.pkl"):
    """
    Make predictions using a trained model.
    :param data: Input data for prediction
    :param model_path: Path to the trained model
    :return: Predictions
    """
    model = joblib.load(model_path)
    return model.predict(data)