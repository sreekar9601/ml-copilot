#!/bin/bash
set -e

# Purge any Vertex AI environment variables that might interfere with Studio API
unset AIPLATFORM_ENDPOINT
unset VERTEXAI_ENDPOINT
unset GOOGLE_CLOUD_PROJECT
unset GOOGLE_APPLICATION_CREDENTIALS

# Debug environment variables
echo "=== Railway Deployment Debug ==="
echo "PORT: ${PORT:-not_set}"
echo "RAILWAY_PUBLIC_DOMAIN: ${RAILWAY_PUBLIC_DOMAIN:-not_set}"
echo "RAILWAY_STATIC_URL: ${RAILWAY_STATIC_URL:-not_set}"
echo "HOST: 0.0.0.0"
echo "PWD: $(pwd)"
echo "Python version: $(python --version)"
echo "Uvicorn version: $(uvicorn --version)"
echo "Google API Key prefix: ${GOOGLE_API_KEY:0:8}..."
echo "Vertex env vars (should be unset):"
echo "  AIPLATFORM_ENDPOINT: ${AIPLATFORM_ENDPOINT:-unset}"
echo "  VERTEXAI_ENDPOINT: ${VERTEXAI_ENDPOINT:-unset}"
echo "  GOOGLE_CLOUD_PROJECT: ${GOOGLE_CLOUD_PROJECT:-unset}"
echo "All env vars:"
env | grep -E "(PORT|RAILWAY|GOOGLE)" || echo "No relevant vars found"
echo "================================"

# Use PORT if set, otherwise default to 8000
PORT=${PORT:-8000}
echo "Using port: $PORT"

# Start the application
echo "Starting uvicorn on 0.0.0.0:$PORT"
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
