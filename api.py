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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    pool_pre_ping=True,
    pool_recycle=3600  # Recycle connections after 1 hour
)

logger.info("Database engine initialized successfully")

# ----------------------------
# CO2 status helper
# ----------------------------
def co2_status(co2):
    """Return status information based on CO2 levels"""
    if co2 < 450:
        return {"label": "Good",      "color": "green",  "message": "Air quality is good"}
    elif co2 < 600:
        return {"label": "Moderate",  "color": "yellow", "message": "Moderate CO2 levels"}
    elif co2 < 1000:
        return {"label": "Poor",      "color": "orange", "message": "Poor air quality"}
    else:
        return {"label": "Hazardous", "color": "red",    "message": "High CO2 — ventilate now!"}

# ----------------------------
# Root endpoint
# ----------------------------
@app.route('/', methods=['GET'])
def root():
    """Root endpoint to verify API is running"""
    return jsonify({
        'message': 'CO2 Prediction API',
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'latest_prediction': '/api/predictions/latest',
            'all_predictions': '/api/predictions',
            'trigger_pipeline': '/api/run-pipeline (POST)'
        }
    }), 200

# ----------------------------
# Health check endpoint
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """Health check for monitoring and load balancers"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'service': 'operational'
        }), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': 'Database connection failed'
        }), 503

# ----------------------------
# GET /api/predictions/latest
# ----------------------------
@app.route('/api/predictions/latest', methods=['GET'])
def get_latest_prediction():
    """Get the most recent CO2 prediction"""
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
            logger.warning("No predictions found in database")
            return jsonify({'error': 'No prediction found yet'}), 404

        data = dict(row)
        data['timestamp'] = data['timestamp'].isoformat()
        data['status'] = co2_status(data['co2'])
        
        return jsonify(data), 200

    except Exception as e:
        logger.error(f"Error fetching latest prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# ----------------------------
# GET /api/predictions
# ----------------------------
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get recent CO2 predictions with optional limit"""
    try:
        # Get and validate limit parameter
        limit = request.args.get('limit', 10, type=int)
        if limit < 1 or limit > 100:
            return jsonify({'error': 'Limit must be between 1 and 100'}), 400
        
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT id, timestamp, co2, temperature, humidity
                    FROM {PREDICTION_TABLE}
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """),
                {'limit': limit}
            )
            rows = result.mappings().fetchall()

        data = []
        for row in rows:
            item = dict(row)
            item['timestamp'] = item['timestamp'].isoformat()
            item['status'] = co2_status(item['co2'])
            data.append(item)

        return jsonify({
            'predictions': data,
            'count': len(data),
            'limit': limit
        }), 200

    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# ----------------------------
# POST /api/run-pipeline
# ----------------------------
@app.route('/api/run-pipeline', methods=['POST'])
def trigger_pipeline():
    """Manually trigger the ML prediction pipeline"""
    try:
        logger.info("Manual pipeline trigger requested")
        run_pipeline()
        logger.info("Pipeline completed successfully")
        return jsonify({'message': 'Pipeline ran successfully'}), 200
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return jsonify({'error': 'Pipeline execution failed'}), 500

# ----------------------------
# Error handlers
# ----------------------------
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not supported for this endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# ----------------------------
# Run Flask app
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)