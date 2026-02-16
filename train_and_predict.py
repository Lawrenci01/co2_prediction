import os
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_SSL_CA,
    SENSOR_TABLE, PREDICTION_TABLE, MODEL_PATH, SCALER_PATH,
    FEATURES, SEQ_LENGTH, PREDICT_LENGTH, FETCH_DAYS
)

# ----------------------------
# Configure logging (instead of print statements)
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# FIXED: Singleton database engine (prevents memory leak)
# ----------------------------
_engine = None

def get_engine():
    """
    Get or create a singleton SQLAlchemy engine.
    
    BEFORE: Created new engine every call (memory leak)
    AFTER: Creates once, reuses (efficient)
    """
    global _engine
    if _engine is None:
        url = (
            f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}"
            f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
            f"?ssl_ca={MYSQL_SSL_CA}"
        )
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600  # Recycle connections after 1 hour
        )
        logger.info("Database engine created successfully")
    return _engine


# ----------------------------
# IMPROVED: Fetch and resample IoT data to hourly
# ----------------------------
def fetch_sensor_data(days_back=FETCH_DAYS):
    """
    Fetch sensor data from MySQL and resample to hourly.
    
    NEW: Automatically resamples to hourly frequency
    This makes predictions 60× faster!
    """
    try:
        engine = get_engine()

        with engine.connect() as conn:
            total = conn.execute(text(f"SELECT COUNT(*) FROM {SENSOR_TABLE}")).scalar()
            latest = conn.execute(text(f"SELECT MAX(recorded_at) FROM {SENSOR_TABLE}")).scalar()
        
        logger.info(f"ℹ  Total rows in sensor_data: {total}, latest recorded_at: {latest}")

        # Try recent data first
        query = f"""
        SELECT
            recorded_at   AS timestamp,
            co2_density   AS co2,
            temperature_c AS temperature,
            humidity      AS humidity
        FROM {SENSOR_TABLE}
        WHERE recorded_at >= NOW() - INTERVAL {days_back} DAY
          AND co2_density   IS NOT NULL
          AND temperature_c IS NOT NULL
          AND humidity      IS NOT NULL
        ORDER BY recorded_at ASC
        """
        df = pd.read_sql_query(query, engine)

        # Fall back to all available data if not enough recent rows
        if len(df) <= SEQ_LENGTH:
            logger.warning(f"  Only {len(df)} recent rows (need >{SEQ_LENGTH}). Fetching all available data...")
            query_all = f"""
            SELECT
                recorded_at   AS timestamp,
                co2_density   AS co2,
                temperature_c AS temperature,
                humidity      AS humidity
            FROM {SENSOR_TABLE}
            WHERE co2_density   IS NOT NULL
              AND temperature_c IS NOT NULL
              AND humidity      IS NOT NULL
            ORDER BY recorded_at ASC
            """
            df = pd.read_sql_query(query_all, engine)

        if df.empty:
            logger.error(" No IoT data found in database.")
            return None

        df.set_index('timestamp', inplace=True)
        
        # NEW: Resample to hourly (this is the key optimization!)
        df = df.resample('H').mean()
        
        # Handle missing values
        df = df.interpolate(method='linear').ffill().bfill()

        logger.info(f" Fetched {len(df)} rows from sensor_data (resampled to hourly)")
        logger.info(f"   Date range: {df.index.min()} → {df.index.max()}")
        return df

    except Exception as e:
        logger.error(f" Error fetching IoT data: {e}", exc_info=True)
        return None


# ----------------------------
# Load fallback CSV when DB has no data yet
# ----------------------------
def load_fallback_csv():
    csv_path = "data/hourly_data.csv"
    if not os.path.exists(csv_path):
        logger.error(f" Fallback CSV not found at {csv_path}")
        return None

    
    logger.info(f" No DB data yet — loading fallback CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.columns = FEATURES
    
    logger.info(f" Loaded {len(df)} rows from CSV.")
    logger.info(f"   Date range: {df.index.min()} → {df.index.max()}")
    return df


# ----------------------------
# IMPROVED: Create sequences with train/validation split
# ----------------------------
def create_sequences(data_scaled, seq_length, train_split=0.8):
    """
    Create sequences for LSTM with train/validation split.
    
    NEW: Splits data into training and validation sets
    This lets us measure if the model is actually learning!
    """
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i : i + seq_length])
        y.append(data_scaled[i + seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and validation
    split_idx = int(len(X) * train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Created {len(X)} sequences")
    logger.info(f"  Training: {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val)} samples")
    
    return X_train, y_train, X_val, y_val


# ----------------------------
# Build LSTM model (unchanged - your architecture is good!)
# ----------------------------
def build_model(seq_length, num_features):
    model = Sequential([
        Input(shape=(seq_length, num_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_features)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logger.info(f"Model created with {model.count_params():,} parameters")
    return model


# ----------------------------
# Predict next 7 days autoregressively
# ----------------------------
def predict_next_week_avg(model, scaler, df):
    """
    Predict next week's average CO2, temperature, humidity.
    
    Now much faster with hourly data (168 predictions vs 10,080!)
    """
    data_scaled = scaler.transform(df[FEATURES].values)
    last_seq = data_scaled[-SEQ_LENGTH:].copy()
    predictions_scaled = []

    logger.info(f"🔮 Predicting next 7 days ({PREDICT_LENGTH} steps)...")
    for step in range(PREDICT_LENGTH):
        input_seq = last_seq.reshape(1, SEQ_LENGTH, len(FEATURES))
        next_pred = model.predict(input_seq, verbose=0)[0]
        predictions_scaled.append(next_pred)
        last_seq = np.vstack([last_seq[1:], next_pred.reshape(1, -1)])

        # Progress every 25%
        if PREDICT_LENGTH >= 4 and (step + 1) % (PREDICT_LENGTH // 4) == 0:
            logger.info(f"   {(step + 1) / PREDICT_LENGTH * 100:.0f}% complete...")

    predictions = scaler.inverse_transform(np.array(predictions_scaled))
    return (
        float(np.mean(predictions[:, 0])),  # co2
        float(np.mean(predictions[:, 1])),  # temperature
        float(np.mean(predictions[:, 2]))   # humidity
    )


# ----------------------------
# Store weekly prediction (unchanged)
# ----------------------------
def store_prediction(co2, temperature, humidity):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {PREDICTION_TABLE} (timestamp, co2, temperature, humidity)
                    VALUES (:ts, :co2, :temp, :hum)
                """),
                {"ts": datetime.now(), "co2": co2, "temp": temperature, "hum": humidity}
            )
            conn.commit()
        logger.info(f" Stored prediction: CO2={co2:.2f} ppm, Temp={temperature:.2f}°C, Humidity={humidity:.2f}%")
    except Exception as e:
        logger.error(f" Error storing prediction: {e}")


# ----------------------------
# IMPROVED: Main pipeline with better training
# ----------------------------
def run_pipeline():
    logger.info("="*55)
    logger.info(f"CO2 Prediction Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*55)

    # 1. Load data — real IoT if available, CSV fallback if not
    df = fetch_sensor_data()
    if df is None:
        df = load_fallback_csv()
    if df is None:
        logger.error(" No data available. Exiting.")
        return

    # 2. Validate enough rows for sequencing
    if len(df) <= SEQ_LENGTH:
        logger.error(f" Not enough data: need >{SEQ_LENGTH} rows, got {len(df)}.")
        logger.error(f"   Tip: Reduce SEQ_LENGTH in config.py or wait for more IoT data.")
        return

    # 3. Fit or load scaler
    os.makedirs('models', exist_ok=True)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        logger.info(" Loaded existing scaler.")
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[FEATURES].values)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(" Fitted and saved new scaler.")

    # 4. Scale and create sequences WITH validation split
    data_scaled = scaler.transform(df[FEATURES].values)
    X_train, y_train, X_val, y_val = create_sequences(data_scaled, SEQ_LENGTH)
    
    if len(X_train) == 0:
        logger.error(" Not enough data to create sequences.")
        return

    # 5. Train or fine-tune model with early stopping
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logger.info(" Loaded existing model — fine-tuning on latest data...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        logger.info(" Model fine-tuned and saved.")
    else:
        logger.info(" Training new model from scratch...")
        model = build_model(SEQ_LENGTH, len(FEATURES))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        logger.info(" New model trained and saved.")
    
    # Log training quality
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    logger.info(f"   Training loss: {final_train_loss:.4f}")
    logger.info(f"   Validation loss: {final_val_loss:.4f}")
    
    if final_val_loss > final_train_loss * 1.5:
        logger.warning("  Model may be overfitting (validation loss much higher than training)")

    # 6. Predict and store
    co2_avg, temp_avg, hum_avg = predict_next_week_avg(model, scaler, df)
    store_prediction(co2_avg, temp_avg, hum_avg)
    
    logger.info("\n Pipeline complete!")


if __name__ == "__main__":
    run_pipeline()