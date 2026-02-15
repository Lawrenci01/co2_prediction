import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_SSL_CA,
    SENSOR_TABLE, PREDICTION_TABLE, MODEL_PATH, SCALER_PATH,
    FEATURES, SEQ_LENGTH, PREDICT_LENGTH, FETCH_DAYS
)

# ----------------------------
# Shared SQLAlchemy engine
# ----------------------------
def get_engine():
    url = (
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        f"?ssl_ca={MYSQL_SSL_CA}"
    )
    return create_engine(url, pool_pre_ping=True)


# ----------------------------
# Fetch IoT data
# Tries last FETCH_DAYS first, then falls back to all available data.
# When real IoT is live, this will automatically pick up real readings.
# ----------------------------
def fetch_sensor_data(days_back=FETCH_DAYS):
    try:
        engine = get_engine()

        with engine.connect() as conn:
            total  = conn.execute(text(f"SELECT COUNT(*) FROM {SENSOR_TABLE}")).scalar()
            latest = conn.execute(text(f"SELECT MAX(recorded_at) FROM {SENSOR_TABLE}")).scalar()
        print(f"ℹ️  Total rows in sensor_data: {total}, latest recorded_at: {latest}")

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
            print(f"⚠️  Only {len(df)} recent rows (need >{SEQ_LENGTH}). Fetching all available data...")
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
            print("❌ No IoT data found in database.")
            return None

        df.set_index('timestamp', inplace=True)
        df = df.rolling(5).mean().ffill().bfill()

        # Report data source so we know if it's real or dummy
        print(f"✅ Fetched {len(df)} rows from sensor_data.")
        print(f"   Date range: {df.index.min()} → {df.index.max()}")
        return df

    except Exception as e:
        print("❌ Error fetching IoT data:", e)
        return None


# ----------------------------
# Load fallback CSV when DB has no data yet
# Once real IoT data arrives, this path will never be taken
# ----------------------------
def load_fallback_csv():
    csv_path = "data/hourly_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Fallback CSV not found at {csv_path}")
        return None
    print(f"📂 No DB data yet — loading fallback CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.columns = FEATURES
    print(f"✅ Loaded {len(df)} rows from CSV.")
    print(f"   Date range: {df.index.min()} → {df.index.max()}")
    return df


# ----------------------------
# Create sequences for LSTM
# ----------------------------
def create_sequences(data_scaled, seq_length):
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i : i + seq_length])
        y.append(data_scaled[i + seq_length])
    return np.array(X), np.array(y)


# ----------------------------
# Build LSTM model
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
    model.compile(optimizer='adam', loss='mse')
    return model


# ----------------------------
# Predict next 7 days autoregressively
# ----------------------------
def predict_next_week_avg(model, scaler, df):
    data_scaled = scaler.transform(df[FEATURES].values)
    last_seq = data_scaled[-SEQ_LENGTH:].copy()
    predictions_scaled = []

    print(f"🔮 Predicting next 7 days ({PREDICT_LENGTH} steps)...")
    for step in range(PREDICT_LENGTH):
        input_seq = last_seq.reshape(1, SEQ_LENGTH, len(FEATURES))
        next_pred = model.predict(input_seq, verbose=0)[0]
        predictions_scaled.append(next_pred)
        last_seq = np.vstack([last_seq[1:], next_pred.reshape(1, -1)])

        # Progress every 25% (guard against PREDICT_LENGTH < 4)
        if PREDICT_LENGTH >= 4 and (step + 1) % (PREDICT_LENGTH // 4) == 0:
            print(f"   {(step + 1) / PREDICT_LENGTH * 100:.0f}% complete...")

    predictions = scaler.inverse_transform(np.array(predictions_scaled))
    return (
        float(np.mean(predictions[:, 0])),  # co2
        float(np.mean(predictions[:, 1])),  # temperature
        float(np.mean(predictions[:, 2]))   # humidity
    )


# ----------------------------
# Store weekly prediction
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
        print(f"✅ Stored prediction: CO2={co2:.2f} ppm, Temp={temperature:.2f}°C, Humidity={humidity:.2f}%")
    except Exception as e:
        print("❌ Error storing prediction:", e)


# ----------------------------
# Main pipeline — runs once per call
# Called by scheduler.py every week automatically
# ----------------------------
def run_pipeline():
    print(f"\n{'='*55}")
    print(f"  CO2 Prediction Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")

    # 1. Load data — real IoT if available, CSV fallback if not
    df = fetch_sensor_data()
    if df is None:
        df = load_fallback_csv()
    if df is None:
        print("❌ No data available. Exiting.")
        return

    # 2. Validate enough rows for sequencing
    if len(df) <= SEQ_LENGTH:
        print(f"❌ Not enough data: need >{SEQ_LENGTH} rows, got {len(df)}.")
        print(f"   Tip: Reduce SEQ_LENGTH in config.py or wait for more IoT data.")
        return

    # 3. Fit or load scaler
    os.makedirs('models', exist_ok=True)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ Loaded existing scaler.")
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[FEATURES].values)
        joblib.dump(scaler, SCALER_PATH)
        print("✅ Fitted and saved new scaler.")

    # 4. Scale and create sequences
    data_scaled = scaler.transform(df[FEATURES].values)
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    if len(X) == 0:
        print("❌ Not enough data to create sequences.")
        return

    # 5. Train or fine-tune model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mse')
        print("✅ Loaded existing model — fine-tuning on latest data...")
        model.fit(X, y, epochs=3, batch_size=64, verbose=1)
        model.save(MODEL_PATH)
        print("✅ Model fine-tuned and saved.")
    else:
        print("🔄 Training new model from scratch...")
        model = build_model(SEQ_LENGTH, len(FEATURES))
        model.fit(X, y, epochs=10, batch_size=64, verbose=1)
        model.save(MODEL_PATH)
        print("✅ New model trained and saved.")

    # 6. Predict and store
    co2_avg, temp_avg, hum_avg = predict_next_week_avg(model, scaler, df)
    store_prediction(co2_avg, temp_avg, hum_avg)
    print(f"\n🎉 Pipeline complete!")


if __name__ == "__main__":
    run_pipeline()