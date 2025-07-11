import os
import yaml
import re

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

# Load the base configuration
config = load_config()

# Override values for API keys.
# Precedence: Environment variable > config.yaml placeholder
config["groq"] = {
    "api_key": os.environ.get("GROQ_API_KEY", config.get("groq", {}).get("api_key"))
}

config["twelve_data"] = {
    "api_key": os.environ.get("TWELVEDATA_API_KEY", config.get("twelve_data", {}).get("api_key"))
}
