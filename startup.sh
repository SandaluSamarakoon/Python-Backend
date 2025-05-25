#!/bin/bash

# Azure App Service startup script for FastAPI application
echo "Starting FastAPI application..."

# Install system dependencies if needed
apt-get update && apt-get install -y ffmpeg

# Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start the FastAPI application with gunicorn
gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000 --timeout 300