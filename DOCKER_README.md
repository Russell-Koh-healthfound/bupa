# HealthFound API - Docker Setup

This document explains how to build and run the HealthFound API using Docker.

## Prerequisites

- Docker installed on your system
- NVIDIA Container Toolkit (if using GPU)
- At least 8GB of free disk space for the Docker image

## Building the Docker Image

1. Navigate to the project directory:
   ```bash
   cd /path/to/Healthfound_v1
   ```

2. Build the Docker image:
   ```bash
   docker build -t healthfound-api .
   ```

   This might take a while as it needs to download and install all dependencies.

## Running the Container

### With GPU Support (Recommended for inference)

```bash
docker run -d \
  --name healthfound-api \
  -p 5000:5000 \
  --gpus all \
  --shm-size=8g \
  healthfound-api
```

### CPU-Only Mode

```bash
docker run -d \
  --name healthfound-api \
  -p 5000:5000 \
  --shm-size=8g \
  healthfound-api
```

## Using the Build Script

For convenience, you can use the provided build script:

1. Make the script executable:
   ```bash
   chmod +x build_and_run.sh
   ```

2. Run the script:
   ```bash
   ./build_and_run.sh
   ```

## Verifying the API

Once the container is running, you can test the API:

```bash
curl -X POST http://localhost:5000/health_assessment \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_here", "actual_age": 30}'
```

## Stopping the Container

To stop the container:

```bash
docker stop healthfound-api
```

To remove the container:

```bash
docker rm healthfound-api
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors, try:
1. Increasing Docker's memory allocation
2. Using a machine with more RAM
3. Reducing the model's batch size

### CUDA Errors

If you see CUDA-related errors:
1. Ensure you have NVIDIA drivers installed
2. Install the NVIDIA Container Toolkit
3. Make sure your GPU is compatible with the PyTorch version

## Notes

- The first run might take longer as it needs to download the models
- The API is configured to run on port 5000 by default
- Models are loaded from the `models/` directory
