#!/bin/bash

# Ativa o ambiente virtual
source venv_prediction/bin/activate

# Configura variáveis de ambiente
export PYTHONPATH=/home/sasa2020/realtime-stock-analysis
export MODEL_PATH=/home/sasa2020/realtime-stock-analysis/models
export LOG_LEVEL=INFO

# Inicia o serviço
python src/model_serving/stock_prediction_service.py