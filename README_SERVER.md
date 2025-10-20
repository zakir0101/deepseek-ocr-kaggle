# DeepSeek OCR Flask Server

This Flask server provides a web interface for the DeepSeek-OCR model with vLLM backend.

## Features

- **Efficient Model Loading**: Model is loaded once during server startup
- **Image OCR Endpoint**: Process images and return markdown with bounding boxes
- **Base64 Image Support**: Returns images as base64 for immediate display
- **Remote Image URLs**: Extracted images are served via web URLs
- **Health Check**: Server status monitoring endpoint
- **CORS Enabled**: Cross-origin requests supported for frontend integration

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA 11.8+
- vLLM 0.8.5+
- DeepSeek-OCR model weights

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r flask_requirements.txt
   ```

2. **Ensure vLLM and model dependencies are installed:**
   ```bash
   # From the main DeepSeek-OCR requirements
   pip install -r requirements.txt

   # Install vLLM (specific to your CUDA version)
   pip install vllm==0.8.5
   ```

3. **Configure the model path in `config.py`:**
   ```python
   MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # or local path to model
   ```

### Running the Server

1. **Start the Flask server:**
   ```bash
   python flask_server.py
   ```

2. **The server will start on:**
   ```
   http://localhost:5000
   ```

3. **Check server health:**
   ```bash
   curl http://localhost:5000/health
   ```

## API Endpoints

### POST /api/ocr/image

Process an image and return OCR results.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `image`: Image file (JPEG, PNG, etc.)
  - `prompt` (optional): Custom prompt for OCR

**Response:**
```json
{
  "success": true,
  "markdown": "Processed markdown with web URLs",
  "original_markdown": "Raw OCR output",
  "image_with_boxes": "data:image/jpeg;base64,...",
  "extracted_images": [
    "http://localhost:5000/images/0.jpg",
    "http://localhost:5000/images/1.jpg"
  ]
}
```

### GET /images/{filename}

Serve extracted images.

### GET /health

Server health check.

## Integration with React Client

The React client is configured to proxy API requests to the Flask server. Make sure both servers are running:

- Flask server: `http://localhost:5000`
- React client: `http://localhost:3000`

## Deployment to Kaggle + ngrok

For deployment to Kaggle with ngrok:

1. **Install ngrok:**
   ```bash
   pip install pyngrok
   ```

2. **Get ngrok auth token** from [ngrok dashboard](https://dashboard.ngrok.com)

3. **Create deployment script:**
   ```python
   from pyngrok import ngrok

   # Start ngrok tunnel
   public_url = ngrok.connect(5000)
   print(f"Public URL: {public_url}")

   # Update Flask server to use this URL for image links
   ```

4. **Update the Flask server** to use the ngrok URL for image links in the response.

## Configuration

Key configuration options in `config.py`:

- `MODEL_PATH`: Path to DeepSeek-OCR model
- `PROMPT`: Default OCR prompt
- `CROP_MODE`: Enable/disable dynamic cropping
- `BASE_SIZE`, `IMAGE_SIZE`: Image processing sizes

## Troubleshooting

1. **Model loading fails:**
   - Check CUDA installation
   - Verify model path is correct
   - Ensure sufficient GPU memory

2. **CORS errors:**
   - Flask-CORS is already configured
   - Check frontend proxy settings

3. **Image processing errors:**
   - Verify image format support
   - Check file upload limits

## File Structure

```
DeepSeek-OCR-vllm/
├── flask_server.py          # Main Flask server
├── flask_requirements.txt   # Flask dependencies
├── deepseek_ocr.py          # vLLM model implementation
├── config.py               # Configuration
├── process/                # Image processing modules
├── web-client/             # React frontend
└── uploads/                # Uploaded files (created)
└── outputs/                # Processed results (created)
    └── images/             # Extracted images (created)
```