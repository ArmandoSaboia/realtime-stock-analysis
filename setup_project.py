import os

# Define the project structure
PROJECT_STRUCTURE = {
    "src": {
        "data_ingestion": ["__init__.py", "alphavantage_api.py", "kafka_producer.py"],
        "feature_engineering": ["__init__.py", "feature_pipeline.py"],
        "model_training": ["__init__.py", "train_model.py"],
        "prediction": ["__init__.py", "predict.py"],
        "visualization": ["__init__.py", "streamlit_dashboard.py"],
        "genai": ["__init__.py", "langchain_insights.py"],
        "utils": ["__init__.py", "helpers.py"],
    },
    "tests": ["test_data_ingestion.py", "test_feature_engineering.py", "test_model_training.py"],
    "notebooks": ["exploratory_analysis.ipynb"],
    "config": ["config.yaml", "secrets.yaml"],
    "data": {
        "news_articles": [],
    },
    "assets": [],  # For banner image or other assets
}

# Files in the root directory
ROOT_FILES = [
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "README.md",
    "LICENSE",
]

def create_structure(base_path, structure):
    """Recursively create directories and files."""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, list):  # It's a directory with files
            os.makedirs(path, exist_ok=True)
            for file_name in content:
                file_path = os.path.join(path, file_name)
                open(file_path, "a").close()  # Create empty file
        elif isinstance(content, dict):  # It's a nested directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)

def main():
    # Base path for the project
    base_path = os.getcwd()  # Current working directory

    # Create project structure
    create_structure(base_path, PROJECT_STRUCTURE)

    # Create root files
    for file_name in ROOT_FILES:
        file_path = os.path.join(base_path, file_name)
        open(file_path, "a").close()  # Create empty file

    print("Project structure created successfully!")

if __name__ == "__main__":
    main()