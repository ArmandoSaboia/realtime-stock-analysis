#!/bin/bash

# Load secrets from secrets.yaml
SECRETS_FILE="/realtime-stock-analysis/config/secrets.yaml"  

if [ -f "$SECRETS_FILE" ]; then
    export OPENAI_API_KEY=$(python -c "import yaml; print(yaml.safe_load(open('$SECRETS_FILE'))['openai_api_key'])")
    export ALPHAVANTAGE_API_KEY=$(python -c "import yaml; print(yaml.safe_load(open('$SECRETS_FILE'))['alphavantage_api_key'])")
else
    echo "Secrets file not found!"
    exit 1
fi

# Execute the main command
exec "$@"