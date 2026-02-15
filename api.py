import os
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from flask_cors import CORS
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
# Returns the most recent weekly prediction
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
# Returns the last 10 weekly predictions
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
# GET /api/sensor/latest
# Returns the latest raw live reading from the IoT sensor
# ----------------------------
@app.route('/api/sensor/latest', methods=['GET'])
def get_latest_sensor():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT data_id, sensor_id, co2_density, temperature_c,
                           humidity, heat_index_c, carbon_level, recorded_at
                    FROM {SENSOR_TABLE}
                    ORDER BY recorded_at DESC
                    LIMIT 1
                """)
            )
            row = result.mappings().fetchone()

        if not row:
            return jsonify({'error': 'No sensor data found'}), 404

        data = dict(row)
        data['recorded_at'] = data['recorded_at'].isoformat()
        data['humidity'] = str(data['humidity'])
        data['status'] = co2_status(float(data['co2_density'] or 0))
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------
# GET /api/sensor/history?hours=24
# Returns per-minute readings for the last N hours
# Default: 24 hours. Max: 72 hours.
# ----------------------------
@app.route('/api/sensor/history', methods=['GET'])
def get_sensor_history():
    try:
        hours = min(int(request.args.get('hours', 24)), 72)

        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT co2_density, temperature_c, humidity,
                           heat_index_c, carbon_level, recorded_at
                    FROM {SENSOR_TABLE}
                    WHERE recorded_at >= NOW() - INTERVAL :hours HOUR
                    ORDER BY recorded_at ASC
                """),
                {"hours": hours}
            )
            rows = result.mappings().fetchall()

        data = []
        for row in rows:
            item = dict(row)
            item['recorded_at'] = item['recorded_at'].isoformat()
            item['humidity'] = str(item['humidity'])
            data.append(item)

        return jsonify({
            "hours": hours,
            "count": len(data),
            "readings": data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)