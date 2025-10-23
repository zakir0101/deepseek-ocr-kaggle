# Deployment Guide

This document provides detailed deployment instructions for the DeepSeek OCR project.

## 🚀 Quick Deployment

### Auto-Deployment (Recommended)
```bash
./deploy.sh -m "Your deployment message"
```

### Manual Deployment
```bash
# 1. Commit and push changes
git add .
git commit -m "Your changes"
git push origin master

# 2. SSH to server
ssh -p 40032 zakir@223.166.245.194 -L 8080:localhost:8080 -L 5000:localhost:5000

# 3. Deploy on server
pkill -9 python3
cd /home/zakir/deepseek-ocr-kaggle
git fetch origin && git reset --hard origin/master
python3 vast_server.py
```

## 📋 Server Configuration

### Current Server Details
- **Host**: 223.166.245.194:40032
- **User**: zakir
- **Project Path**: `/home/zakir/deepseek-ocr-kaggle`
- **Ports**: 5000 (server), 8080 (frontend)

### Environment
- **Python**: 3.10
- **GPU**: NVIDIA RTX 3090
- **Model**: DeepSeek-OCR
- **Framework**: vLLM 0.8.5

## 🔧 Deployment Script Details

### deploy.sh Features
- Automatic git commit and push
- SSH connection to server
- Process management (stop/start)
- Code synchronization
- Server startup

### Script Usage
```bash
# With custom commit message
./deploy.sh -m "feat: add new feature"

# Without commit message (prompts)
./deploy.sh
```

## 🛠️ Server Management

### Starting Server
```bash
cd /home/zakir/deepseek-ocr-kaggle
python3 vast_server.py
```

### Stopping Server
```bash
pkill -9 python3
```

### Checking Server Status
```bash
curl http://localhost:5000/health
```

## 📊 Health Monitoring

### Expected Health Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-23T..."
}
```

### Server Logs
Check server output for:
- Model loading status
- Import errors
- OCR processing times
- Error messages

## 🔒 Security Considerations

### SSH Access
- Use strong SSH keys
- Limit port exposure
- Monitor connection attempts

### Server Security
- No authentication (public API)
- Input validation on all endpoints
- File type restrictions for uploads
- CORS enabled for cross-origin requests

## 📈 Performance

### Expected Performance
- **Model Loading**: ~35 seconds
- **OCR Processing**: 10-60 seconds
- **GPU Memory**: ~17GB peak
- **Concurrent Requests**: Limited by GPU memory

### Monitoring Commands
```bash
# GPU utilization
nvidia-smi

# System resources
htop

# Server logs
tail -f server.log
```

## 🚨 Troubleshooting

### Common Issues

1. **Server Won't Start**
   - Check NumPy version (1.26.4 required)
   - Verify vLLM 0.8.5 is installed
   - Check port 5000 availability

2. **Model Loading Failed**
   - Verify GPU compatibility
   - Check model files exist
   - Review server logs for specific errors

3. **OCR Timeout**
   - Reduce image size
   - Check GPU memory usage
   - Verify model is properly loaded

### Recovery Procedures

1. **Server Crash**
   ```bash
   pkill -9 python3
   python3 vast_server.py
   ```

2. **Code Issues**
   ```bash
   git fetch origin && git reset --hard origin/master
   ```

3. **Dependency Problems**
   ```bash
   pip install -r requirements.txt
   ```

## 🔄 Update Procedures

### Regular Updates
1. Make code changes locally
2. Test thoroughly
3. Use `./deploy.sh -m "Your message"`
4. Verify server restarts successfully

### Emergency Updates
1. SSH to server directly
2. Pull latest changes
3. Restart server
4. Verify functionality

## 📝 Change Log

### Recent Deployments
- **2025-10-23**: Auto-deployment script with SSH automation
- **2025-10-23**: Frontend improvements for result display
- **2025-10-23**: Bounding box visualization fixes

---

**Last Updated**: 2025-10-23
**Status**: ✅ Production Ready