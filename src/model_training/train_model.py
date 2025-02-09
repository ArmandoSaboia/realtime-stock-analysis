from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train, model_path="model.pkl"):
    """
    Train a machine learning model.
    :param X_train: Training features
    :param y_train: Training labels
    :param model_path: Path to save the trained model
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")