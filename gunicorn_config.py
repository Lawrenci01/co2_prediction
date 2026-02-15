import os

# Bind to the PORT that Render provides (CRITICAL!)
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Use 1 worker for TensorFlow (memory-intensive)
workers = 1

# Increase timeout for ML predictions
timeout = 120

# Use sync workers
worker_class = 'sync'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'