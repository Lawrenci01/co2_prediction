import os
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from flask_cors import CORS
from train_and_predict import run_pipeline
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD,
    MYSQL_DB, MYSQL_SSL_CA, PREDICTION_TABLE, SENSOR_TABLE
)

app = Flask(__name__)
CORS(app)  # Allow frontend (React/Vue/etc) to call this API

# ----------------------------
# Database connection pool
# ----------------------------
DB_URL = (
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    f"?ssl_ca={MYSQL_SSL_CA}"
)

engine = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)

# ----------------------------
# CO2 status helper
# ----------------------------
def co2_status(co2):
    if co2 < 450:
        return {"label": "Good",      "color": "green",  "message": "Air quality is good"}
    elif co2 < 600:
        return {"label": "Moderate",  "color": "yellow", "message": "Moderate CO2 levels"}
    elif co2 < 1000:
        return {"label": "Poor",      "color": "orange", "message": "Poor air quality"}
    else:
        return {"label": "Hazardous", "color": "red",    "message": "High CO2 — ventilate now!"}

# ----------------------------
# GET /api/predictions/latest
# ----------------------------
@app.route('/api/predictions/latest', methods=['GET'])
def get_latest_prediction():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT id, timestamp, co2, temperature, humidity
                    FROM {PREDICTION_TABLE}
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
            )
            row = result.mappings().fetchone()

        if not row:
            return jsonify({'error': 'No prediction found yet'}), 404

        data = dict(row)
        data['timestamp'] = data['timestamp'].isoformat()
        data['status'] = co2_status(data['co2'])
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------
# GET /api/predictions
# ----------------------------
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT id, timestamp, co2, temperature, humidity
                    FROM {PREDICTION_TABLE}
                    ORDER BY timestamp DESC
                    LIMIT 10
                """)
            )
            rows = result.mappings().fetchall()

        data = []
        for row in rows:
            item = dict(row)
            item['timestamp'] = item['timestamp'].isoformat()
            item['status'] = co2_status(item['co2'])
            data.append(item)

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------
# POST /api/run-pipeline
# ----------------------------
@app.route('/api/run-pipeline', methods=['POST'])
def trigger_pipeline():
    try:
        run_pipeline()
        return jsonify({'message': 'Pipeline ran successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------
# Run Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
