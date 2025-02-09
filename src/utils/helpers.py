import yaml

def load_config(config_path="config/config.yaml"):
    """
    Load configuration from a YAML file.
    :param config_path: Path to the YAML configuration file
    :return: Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_secrets(secrets_path="config/secrets.yaml"):
    """
    Load secrets from a YAML file.
    :param secrets_path: Path to the YAML secrets file
    :return: Dictionary containing secrets
    """
    with open(secrets_path, "r") as f:
        secrets = yaml.safe_load(f)
    return secrets