#!/bin/bash

# Exit on error
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t healthfound-api .

# Run the container
echo "Starting container..."
docker run -d \
  --name healthfound-api \
  -p 5000:5000 \
  --gpus all \
  --shm-size=8g \
  healthfound-api

echo "\nContainer is running!"
echo "API should be available at http://localhost:5000"
echo "To view logs: docker logs -f healthfound-api"
echo "To stop: docker stop healthfound-api"
