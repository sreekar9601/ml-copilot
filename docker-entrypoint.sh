#!/bin/bash
set -e

# Debug environment variables
echo "=== Railway Deployment Debug ==="
echo "PORT: ${PORT:-not_set}"
echo "RAILWAY_PUBLIC_DOMAIN: ${RAILWAY_PUBLIC_DOMAIN:-not_set}"
echo "RAILWAY_STATIC_URL: ${RAILWAY_STATIC_URL:-not_set}"
echo "HOST: 0.0.0.0"
echo "PWD: $(pwd)"
echo "Python version: $(python --version)"
echo "Uvicorn version: $(uvicorn --version)"
echo "All env vars:"
env | grep -E "(PORT|RAILWAY)" || echo "No PORT or RAILWAY vars found"
echo "================================"

# Use PORT if set, otherwise default to 8000
PORT=${PORT:-8000}
echo "Using port: $PORT"

# Start the application
echo "Starting uvicorn on 0.0.0.0:$PORT"
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
