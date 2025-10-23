# System Architecture

This document describes the architecture and technical implementation of the DeepSeek OCR system.

## 🏗️ System Overview

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│  Flask Server    │◄──►│  vLLM Engine    │
│   (React)       │    │   (Python)       │    │  (DeepSeek-OCR) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Browser  │    │   File System    │    │   GPU Memory    │
│                 │    │  (uploads/outputs)│    │   (~17GB)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Component Details

### 1. Web Client (Frontend)
- **Technology**: React + Vite
- **Port**: 8080 (development)
- **Features**:
  - Image upload with drag & drop
  - Three-tab result display
  - Bounding box visualization
  - Responsive design

### 2. Flask Server (Backend)
- **Technology**: Python Flask
- **Port**: 5000
- **Features**:
  - REST API endpoints
  - Async model inference
  - File upload handling
  - CORS support

### 3. vLLM Engine (Model Serving)
- **Framework**: vLLM 0.8.5
- **Model**: DeepSeek-OCR
- **Features**:
  - Async generation
  - GPU memory optimization
  - Batch processing
  - Logits processing

## 🔧 Technical Implementation

### Server Architecture

#### Main Server (`vast_server.py`)
```python
# Core components
- Flask app with CORS
- AsyncLLMEngine for model inference
- Image processing pipeline
- File management system
- Health monitoring
```

#### Key Functions
1. **Model Initialization**
   ```python
   def initialize_model():
       # Sets up vLLM engine with DeepSeek-OCR
       # Configures GPU memory utilization
       # Registers model with vLLM registry
   ```

2. **OCR Processing**
   ```python
   async def process_image_async(image_path):
       # Loads and preprocesses image
       # Generates OCR using vLLM
       # Extracts bounding boxes
       # Processes output format
   ```

3. **Result Processing**
   ```python
   def process_ocr_output(raw_text):
       # Removes <|ref|> and <|det|> tags
       # Cleans up markdown formatting
       # Extracts bounding box coordinates
   ```

### Frontend Architecture

#### Main Component (`App.jsx`)
```javascript
// Core states
- selectedFile: Uploaded image
- previewUrl: Image preview
- result: OCR results
- activeTab: Current view (rendered/source/raw)

// Key features
- Drag & drop file upload
- Three-tab result display
- Bounding box image display
- Error handling
```

## 📊 Data Flow

### OCR Request Flow
1. **Image Upload** → Frontend sends image to `/ocr/image`
2. **Image Processing** → Server preprocesses image
3. **Model Inference** → vLLM generates OCR output
4. **Result Processing** → Extract boxes and clean markdown
5. **Response** → Return raw, processed, and boxes data

### Response Structure
```json
{
  "success": true,
  "raw_result": "<|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>",
  "markdown": "Clean markdown text",
  "boxes_image": "base64_encoded_image",
  "image_name": "upload_filename.jpg"
}
```

## 🗄️ File System Structure

```
deepseek-ocr-vastai/
├── vast_server.py          # Main server
├── deploy.sh              # Deployment script
├── requirements.txt       # Dependencies
├── web-client/           # Frontend
│   ├── src/
│   │   └── App.jsx       # Main React component
│   └── package.json
├── models/               # Model files
│   └── deepseek-ocr/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files
├── uploads/              # User uploads
├── outputs/              # OCR results
└── docs/                 # Documentation
```

## 🔌 API Endpoints

### Health Check
- **Endpoint**: `GET /health`
- **Purpose**: Server and model status
- **Response**: `{status, model_loaded, timestamp}`

### OCR Processing
- **Endpoint**: `POST /ocr/image`
- **Purpose**: Process image and extract text
- **Input**: Multipart form with image file
- **Response**: OCR results in multiple formats

### Image Serving
- **Endpoint**: `GET /image/<filename>`
- **Purpose**: Serve uploaded images
- **Response**: Image file

## ⚡ Performance Characteristics

### Model Loading
- **Time**: ~35 seconds
- **GPU Memory**: ~6.2GB model weights
- **Total Memory**: ~17GB peak usage

### OCR Processing
- **Time**: 10-60 seconds (image dependent)
- **Memory**: Additional ~0.8GB during inference
- **Throughput**: Single request at a time (GPU memory limited)

### Frontend Performance
- **Initial Load**: Fast (React SPA)
- **Image Upload**: Client-side processing
- **Result Display**: Instant (pre-processed)

## 🔒 Security Considerations

### Input Validation
- File type checking (images only)
- Size limits for uploads
- Path traversal prevention

### Output Sanitization
- Markdown rendering safety
- Base64 encoding for images
- Error message filtering

### Network Security
- CORS configuration
- Port exposure management
- SSH key authentication

## 📈 Scalability

### Current Limitations
- Single GPU instance
- Sequential request processing
- Memory-bound performance

### Potential Improvements
- Multiple GPU support
- Request queuing system
- Model quantization
- Caching mechanisms

## 🛠️ Development Tools

### Debugging
- Server logs with detailed output
- Health endpoint monitoring
- Frontend browser console

### Testing
- Health check validation
- OCR functionality testing
- Frontend UI testing
- Deployment script testing

---

**Last Updated**: 2025-10-23
**Architecture Status**: ✅ Production Ready