import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key, default=None, required=False):
    """Get environment variable with better error messages"""
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(
            f" Missing required environment variable: {key}\n"
            f"   Please set it in Render Dashboard > Environment"
        )
    return value

# MySQL configuration
MYSQL_HOST     = get_env('MYSQL_HOST', required=True)
MYSQL_PORT     = int(get_env('MYSQL_PORT', '3306'))
MYSQL_USER     = get_env('MYSQL_USER', required=True)
MYSQL_PASSWORD = get_env('MYSQL_PASSWORD', required=True)
MYSQL_DB       = get_env('MYSQL_DB', required=True)
MYSQL_SSL_CA   = get_env('MYSQL_SSL_CA', '')  # Empty string if no SSL

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