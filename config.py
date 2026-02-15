import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# MySQL configuration
# ----------------------------
MYSQL_HOST     = os.environ.get('MYSQL_HOST')
MYSQL_PORT     = int(os.environ.get('MYSQL_PORT', 27069))
MYSQL_USER     = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_DB       = os.environ.get('MYSQL_DB')
MYSQL_SSL_CA   = os.environ.get('MYSQL_SSL_CA')

# ----------------------------
# OPTIMIZED LSTM configuration
#
# CHANGED: From minutely to hourly (60× FASTER!)
# OLD: SEQ_LENGTH = 4320, PREDICT_LENGTH = 10080 (8-10 minutes)
# NEW: SEQ_LENGTH = 72, PREDICT_LENGTH = 168 (10-20 seconds)
# ----------------------------
SEQ_LENGTH     = 72      # 3 days of hourly data (was 4320 minutes)
PREDICT_LENGTH = 168     # 7 days hourly (was 10080 minutes)
FETCH_DAYS     = 14

# ----------------------------
# Features — must match SQL alias names in fetch query
# Order: index 0=co2, 1=temperature, 2=humidity
# ----------------------------
FEATURES = ['co2', 'temperature', 'humidity']

# ----------------------------
# File paths
# ----------------------------
MODEL_PATH  = 'models/lstm_model.keras'
SCALER_PATH = 'models/scaler.pkl'

# ----------------------------
# Database tables
# ----------------------------
SENSOR_TABLE     = 'sensor_data'
PREDICTION_TABLE = 'sensor_data_prediction'