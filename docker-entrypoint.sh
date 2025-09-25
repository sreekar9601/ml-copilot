#!/bin/bash
set -e

# Debug environment variables
echo "=== Railway Deployment Debug ==="
echo "PORT: ${PORT:-not_set}"
echo "HOST: 0.0.0.0"
echo "PWD: $(pwd)"
echo "Python version: $(python --version)"
echo "Uvicorn version: $(uvicorn --version)"
echo "================================"

# Check if PORT is set
if [ -z "${PORT}" ]; then
    echo "WARNING: PORT environment variable not set, defaulting to 8000"
    export PORT=8000
fi

echo "Starting uvicorn on 0.0.0.0:${PORT}"

# Start the application
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"
