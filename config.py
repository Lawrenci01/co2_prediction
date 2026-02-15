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
# LSTM configuration (PRODUCTION)
#
# SEQ_LENGTH = 4320  → 3 days of per-minute data as input window
# PREDICT_LENGTH = 10080 → predict next 7 days (minute by minute)
# FETCH_DAYS = 14    → fetch last 14 days of sensor data for training
# ----------------------------
SEQ_LENGTH     = 4320    # 3 days look-back window
PREDICT_LENGTH = 10080   # 7 days prediction (60 min x 24 hr x 7)
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