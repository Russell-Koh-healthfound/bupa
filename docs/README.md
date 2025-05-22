# Medical Image Analysis API Documentation

Welcome to the Medical Image Analysis API documentation! This API is designed to analyze medical images for various eye conditions including Diabetic Retinopathy (DR), Age-related Macular Degeneration (AMD), and Glaucoma.

## Getting Started

### Base URL
```
http://localhost:5000/api/v1
```

### Prerequisites
- Python 3.8+
- Flask
- Flask-RESTX
- PyTorch
- Pillow
- Other dependencies listed in `requirements.txt`

### Installation
1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python flask_app.py
   ```

## API Endpoints

### Health Assessment
- **POST** `/health_assessment` - Analyzes a medical image and returns health assessment results.

## Viewing Documentation

Open the `docs/index.html` file in your web browser to view the interactive API documentation.

## Example Usage

### Using cURL
```bash
curl -X POST "http://localhost:5000/api/v1/health_assessment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/image.jpg" \
  -F "actual_age=45"
```

### Using Python
```python
import requests

url = "http://localhost:5000/api/v1/health_assessment"
files = {'image': open('path/to/image.jpg', 'rb')}
data = {'actual_age': 45}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Response Format

Successful responses will be returned in JSON format with the following structure:

```json
{
  "DR": {
    "score": 2,
    "confidence": 0.92,
    "probabilities": [0.01, 0.07, 0.82, 0.09, 0.01]
  },
  "AMD": {
    "score": 1,
    "confidence": 0.85,
    "probabilities": [0.1, 0.8, 0.1]
  },
  "Glaucoma": {
    "score": 0,
    "confidence": 0.97,
    "probabilities": [0.97, 0.03]
  },
  "Biological age": 45,
  "report_image": "base64_encoded_image_string"
}
```

## Error Handling

Errors are returned with the appropriate HTTP status code and a JSON object containing an error message.

### Error Response Format
```json
{
  "error": "Error type",
  "message": "Human-readable error message"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
