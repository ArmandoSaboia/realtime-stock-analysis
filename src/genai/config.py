import os
import yaml
import re

def load_secrets():
    filepath = "/realtime-stock-analysis/config/secrets.yaml"
    try:
        with open(filepath, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Secrets file not found at {filepath}")
    except IsADirectoryError:
        raise IsADirectoryError(f"'{filepath}' is a directory, not a file.")

def substitute_env_variables(config):
    """
    Recursively replace ${VAR} placeholders in the config with
    their corresponding environment variable values.
    """
    pattern = re.compile(r"\$\{(\w+)\}")

    if isinstance(config, dict):
        return {k: substitute_env_variables(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_variables(i) for i in config]
    elif isinstance(config, str):
        # Look for patterns like ${VAR} in the string and replace them.
        matches = pattern.findall(config)
        for var in matches:
            value = os.environ.get(var, config)
            config = config.replace(f"${{{var}}}", value)
        return config
    else:
        return config

def load_config(filepath="config/config.yaml"):
    """
    Loads the configuration from a YAML file.
    
    Returns:
        dict: The configuration dictionary.
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            config_data = yaml.safe_load(file) or {}
        # Substitute environment variables in the config.yaml file.
        config_data = substitute_env_variables(config_data)
        return config_data
    else:
        raise FileNotFoundError(f"{filepath} not found. Please create it with the required keys.")

# Load secrets from a separate file (if any)
secrets = load_secrets()

# Load the base configuration
config = load_config()

# Override values for OpenAI and Finnhub.
# Precedence: Environment variable > secrets.yaml > config.yaml placeholder
config["openai"] = {
    "api_key": os.environ.get("OPENAI_API_KEY", secrets.get("openai_api_key", config.get("openai", {}).get("api_key")))
}

config["alpha_vantage"] = {
    "api_key": os.environ.get("FINNHUB_API_KEY", secrets.get("finnhub_api_key", config.get("finnhub", {}).get("api_key")))
} 
