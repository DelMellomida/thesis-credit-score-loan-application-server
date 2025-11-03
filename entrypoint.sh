#!/usr/bin/env bash
set -e

# Ensure logs directory exists
mkdir -p logs

# Debug: Print environment variables (excluding sensitive values)
echo "Checking environment variables..."
echo "MONGODB_DB_NAME is set: ${MONGODB_DB_NAME:+yes}"
echo "MONGODB_URI is set: ${MONGODB_URI:+yes}"
echo "GEMINI_API_KEY is set: ${GEMINI_API_KEY:+yes}"
echo "JWT_SECRET_KEY is set: ${JWT_SECRET_KEY:+yes}"
echo "CLIENT_URL is set: ${CLIENT_URL:+yes}"

# Run the main Python script with debug logging
echo "Starting uvicorn server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug