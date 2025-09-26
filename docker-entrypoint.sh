#!/bin/bash
set -e

# This script sets up the environment for the application to run.

echo "=== Railway Deployment Debug ==="

# Create the Google Credentials file from the environment variable
# This is the standard way to handle JSON credentials in containerized environments
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
  echo "Found GOOGLE_APPLICATION_CREDENTIALS_JSON, creating credentials file..."
  echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /app/gcp-credentials.json
  export GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json
  echo "✅ Google credentials file created at /app/gcp-credentials.json"
else
  echo "WARNING: GOOGLE_APPLICATION_CREDENTIALS_JSON not set."
fi

# Debug environment variables
echo "PORT: ${PORT:-not_set}"
echo "RAILWAY_PUBLIC_DOMAIN: ${RAILWAY_PUBLIC_DOMAIN:-not_set}"
echo "RAILWAY_STATIC_URL: ${RAILWAY_STATIC_URL:-not_set}"
echo "HOST: 0.0.0.0"
echo "PWD: $(pwd)"
echo "Python version: $(python --version)"
echo "Uvicorn version: $(uvicorn --version)"
echo "GOOGLE_CLOUD_PROJECT: ${GOOGLE_CLOUD_PROJECT:-not_set}"
echo "GOOGLE_CLOUD_LOCATION: ${GOOGLE_CLOUD_LOCATION:-not_set}"
echo "GOOGLE_GENAI_USE_VERTEXAI: ${GOOGLE_GENAI_USE_VERTEXAI:-not_set}"
echo "GOOGLE_APPLICATION_CREDENTIALS: ${GOOGLE_APPLICATION_CREDENTIALS:-not_set}"
echo "================================"

# Use PORT if set, otherwise default to 8000
PORT=${PORT:-8000}
echo "Using port: $PORT"

# Start the application
echo "Starting uvicorn on 0.0.0.0:$PORT"
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
