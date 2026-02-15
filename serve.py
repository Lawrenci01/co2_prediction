# Production WSGI server for Windows using Waitress
# Replaces Flask's built-in development server
# Run with: python serve.py

from waitress import serve
from api import app

HOST = '0.0.0.0'
PORT = 5000

if __name__ == "__main__":
    print(f"🚀 Production server started on http://{HOST}:{PORT}")
    print(f"   Press Ctrl+C to stop.")
    serve(app, host=HOST, port=PORT, threads=6)