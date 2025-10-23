# DeepSeek OCR - Vast.ai Deployment

A production-ready OCR server using DeepSeek's multimodal model for document text extraction and conversion to markdown format. Optimized for deployment on Vast.ai GPU instances.

## 🚀 Quick Start

### Prerequisites
- Vast.ai account with GPU credits
- SSH access configured
- Git repository access

### One-Command Deployment
```bash
./deploy.sh -m "Your deployment message"
```

### Manual Deployment
1. **Commit and push changes**:
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin master
   ```

2. **SSH to server**:
   ```bash
   ssh -p 40032 zakir@223.166.245.194 -L 8080:localhost:8080 -L 5000:localhost:5000
   ```

3. **Deploy on server**:
   ```bash
   pkill -9 python3
   cd /home/zakir/deepseek-ocr-kaggle
   git fetch origin && git reset --hard origin/master
   python3 vast_server.py
   ```

## 📋 Features

- **Document OCR**: Extract text from images and convert to markdown
- **Bounding Box Visualization**: Display detected text regions with bounding boxes
- **Multiple Output Formats**: Raw OCR output, processed markdown, and rendered markdown
- **Web Interface**: User-friendly frontend for image upload and result display
- **Auto-deployment**: One-command deployment to Vast.ai instances

## 🏗️ Architecture

### Server Components
- **Flask Backend**: REST API server on port 5000
- **vLLM Engine**: Async model inference with DeepSeek-OCR
- **Frontend**: React web interface on port 8080
- **File Management**: Upload and output directories

### Model Information
- **Model**: DeepSeek-OCR (multimodal)
- **Framework**: vLLM 0.8.5
- **GPU**: NVIDIA RTX 3090 (compatible with sm_86+)
- **Memory**: ~17GB GPU memory utilization

## 🔧 Configuration

### Server Endpoints
- `GET /health` - Health check and model status
- `POST /ocr/image` - OCR processing endpoint
- `GET /image/<filename>` - Serve uploaded images

### Environment Variables
```bash
VLLM_USE_V1=0  # Legacy vLLM API compatibility
MODEL_PATH=/home/zakir/deepseek-ocr-kaggle/models/deepseek-ocr
UPLOAD_FOLDER=/home/zakir/deepseek-ocr-kaggle/uploads
OUTPUT_FOLDER=/home/zakir/deepseek-ocr-kaggle/outputs
```

## 📁 Project Structure

```
deepseek-ocr-vastai/
├── vast_server.py          # Main Flask server
├── deploy.sh              # Auto-deployment script
├── requirements.txt       # Python dependencies
├── web-client/           # React frontend
│   ├── src/
│   │   └── App.jsx       # Main React component
│   └── package.json
├── models/               # Model files
│   └── deepseek-ocr/
├── uploads/              # Uploaded images
├── outputs/              # OCR results
└── docs/                 # Documentation
```

## 🛠️ Development

### Local Development
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start server**:
   ```bash
   python3 vast_server.py
   ```

3. **Access endpoints**:
   - Server: http://localhost:5000
   - Frontend: http://localhost:8080

### Frontend Development
```bash
cd web-client
npm install
npm run dev
```

## 🔍 Testing

### Health Check
```bash
curl http://localhost:5000/health
```

### OCR Test
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/ocr/image
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check GPU compatibility (RTX 3090/4090/A100 recommended)
   - Verify model files exist in `models/deepseek-ocr/`
   - Ensure vLLM 0.8.5 is installed

2. **Server Won't Start**
   - Check NumPy version (1.26.4 required)
   - Verify all dependencies in requirements.txt
   - Check port 5000 availability

3. **OCR Timeout**
   - Large images may timeout - resize images before upload
   - Check GPU memory usage
   - Verify model is properly loaded

4. **Frontend Issues**
   - Check if React dev server is running
   - Verify API endpoints are accessible
   - Check browser console for errors

### Server Logs
Check server logs for detailed error information:
```bash
# On the server
tail -f /home/zakir/deepseek-ocr-kaggle/server.log
```

## 📊 Performance

- **Model Loading**: ~35 seconds
- **OCR Processing**: 10-60 seconds depending on image complexity
- **GPU Memory**: ~17GB peak usage
- **Concurrent Requests**: Limited by GPU memory

## 🔒 Security

- **CORS**: Enabled for cross-origin requests
- **File Upload**: Restricted to image files only
- **Input Validation**: All inputs validated server-side
- **No Authentication**: Public API - use with caution

## 📈 Monitoring

### Health Metrics
- Server status via `/health` endpoint
- Model loading status
- Request processing times
- Error rates

### GPU Monitoring
```bash
nvidia-smi  # Check GPU utilization
htop        # System resource monitoring
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Use `./deploy.sh -m "Your commit message"` for deployment
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [DeepSeek AI](https://github.com/deepseek-ai/DeepSeek-OCR) for the OCR model
- [Vast.ai](https://vast.ai) for GPU hosting
- [vLLM](https://github.com/vllm-project/vllm) for model serving

---

**Last Updated**: 2025-10-23
**Server Status**: ✅ Running on Vast.ai
**Model Status**: ✅ Loaded and functional