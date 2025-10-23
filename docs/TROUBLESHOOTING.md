# Troubleshooting Guide

This document provides solutions for common issues encountered with the DeepSeek OCR system.

## 🚨 Critical Issues

### Server Won't Start

**Symptoms:**
- Python process exits immediately
- Import errors in console
- Port 5000 already in use

**Solutions:**
1. **Check Dependencies**
   ```bash
   pip install -r requirements.txt
   python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
   python -c "from vllm import AsyncLLMEngine; print('vLLM OK')"
   ```

2. **Check Port Availability**
   ```bash
   netstat -tulpn | grep :5000
   # If port is in use:
   pkill -9 python3
   ```

3. **Verify GPU Compatibility**
   ```bash
   nvidia-smi
   # Check for RTX 3090/4090/A100
   ```

### Model Loading Failed

**Symptoms:**
- `model_loaded: false` in health response
- CUDA out of memory errors
- Model architecture not recognized

**Solutions:**
1. **Check Model Files**
   ```bash
   ls -la models/deepseek-ocr/
   # Should contain: config.json, model.safetensors, tokenizer files
   ```

2. **Verify vLLM Version**
   ```bash
   python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
   # Must be 0.8.5
   ```

3. **Check GPU Memory**
   ```bash
   nvidia-smi
   # Ensure at least 16GB free
   ```

### OCR Processing Timeout

**Symptoms:**
- 120 second timeout errors
- Frontend hangs during processing
- No results returned

**Solutions:**
1. **Reduce Image Size**
   - Resize images to < 2000px on longest side
   - Use JPEG format instead of PNG
   - Compress images before upload

2. **Check Server Logs**
   ```bash
   # Look for generation errors
   tail -f server_output.log
   ```

3. **Test with Small Image**
   - Use a simple test image first
   - Verify basic functionality
   - Then try larger images

## 🔧 Frontend Issues

### Boxes Image Not Displaying

**Symptoms:**
- Bounding boxes section empty
- Base64 image errors in console
- "Image with Bounding Boxes" section missing

**Solutions:**
1. **Check API Response**
   ```javascript
   // Verify response contains boxes_image
   console.log(result.boxes_image);
   ```

2. **Verify Base64 Encoding**
   ```javascript
   // Frontend should use:
   src={`data:image/jpeg;base64,${result.boxes_image}`}
   ```

3. **Check Server Box Generation**
   - Verify `create_boxes_image()` function works
   - Check bounding box extraction
   - Verify image saving to outputs/

### Tab Switching Problems

**Symptoms:**
- Active tab not highlighting
- Content not switching between tabs
- Raw/source/rendered views mixed up

**Solutions:**
1. **Check Active Tab State**
   ```javascript
   // Verify activeTab state management
   console.log('Active tab:', activeTab);
   ```

2. **Verify Conditional Rendering**
   ```javascript
   // Should have proper conditional logic
   {activeTab === 'rendered' && <RenderedContent />}
   {activeTab === 'source' && <SourceContent />}
   {activeTab === 'raw' && <RawContent />}
   ```

3. **Check CSS Classes**
   - Verify `.tab.active` styling
   - Check tab button event handlers

## 🛠️ Deployment Issues

### Deployment Script Fails

**Symptoms:**
- Git push errors
- SSH connection failures
- Server not restarting

**Solutions:**
1. **Manual Deployment**
   ```bash
   # Commit and push manually
   git add .
   git commit -m "Manual fix"
   git push origin master

   # SSH manually
   ssh -p 40032 zakir@223.166.245.194

   # Deploy manually
   pkill -9 python3
   cd /home/zakir/deepseek-ocr-kaggle
   git fetch origin && git reset --hard origin/master
   python3 vast_server.py
   ```

2. **Check SSH Configuration**
   - Verify SSH keys are set up
   - Check firewall settings
   - Test SSH connection separately

3. **Verify Git Access**
   - Check repository permissions
   - Verify remote URL
   - Test git push manually

### Server Not Updating

**Symptoms:**
- Old code still running after deployment
- Git pull not fetching new changes
- Server restarting with old version

**Solutions:**
1. **Force Git Reset**
   ```bash
   git fetch origin
   git reset --hard origin/master
   ```

2. **Verify Process Kill**
   ```bash
   # Ensure all Python processes are stopped
   pkill -9 python3
   ps aux | grep python3
   ```

3. **Check File Permissions**
   ```bash
   ls -la /home/zakir/deepseek-ocr-kaggle/
   # Ensure user has write permissions
   ```

## 📊 Performance Issues

### Slow OCR Processing

**Symptoms:**
- Processing takes > 60 seconds
- GPU utilization low
- System memory high

**Solutions:**
1. **Optimize Image Size**
   - Reduce image dimensions
   - Use appropriate compression
   - Consider image cropping

2. **Monitor GPU Usage**
   ```bash
   watch nvidia-smi
   # Check for memory bottlenecks
   ```

3. **Check System Resources**
   ```bash
   htop
   # Monitor CPU and memory usage
   ```

### Memory Issues

**Symptoms:**
- CUDA out of memory errors
- System swapping
- Process killed by OOM killer

**Solutions:**
1. **Reduce GPU Memory Usage**
   ```python
   # In vast_server.py, reduce:
   gpu_memory_utilization=0.75  # Try 0.65 if issues
   ```

2. **Limit Concurrent Requests**
   - Implement request queue
   - Reject simultaneous requests
   - Add request timeout

3. **Optimize Model Settings**
   - Reduce max_tokens if possible
   - Adjust block_size parameters
   - Consider model quantization

## 🔍 Diagnostic Commands

### Server Health Check
```bash
# Basic health
curl http://localhost:5000/health

# Detailed server info
ps aux | grep python3
netstat -tulpn | grep :5000
```

### Model Status
```bash
# Check model loading
python3 -c "
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
print('vLLM imports OK')
"

# Check GPU
nvidia-smi
```

### Frontend Diagnostics
```javascript
// Browser console
console.log('API Response:', result);
console.log('Active Tab:', activeTab);
console.log('Boxes Image:', result.boxes_image?.length);
```

## 🚨 Emergency Recovery

### Complete System Reset
```bash
# Stop everything
pkill -9 python3
pkill -9 node

# Clean and reset
cd /home/zakir/deepseek-ocr-kaggle
git fetch origin
git reset --hard origin/master

# Fresh start
python3 vast_server.py &
cd web-client && npm run dev &
```

### Model Reinstallation
```bash
# If model files corrupted
rm -rf models/deepseek-ocr/
# Re-download model (manual process)
```

### Dependency Reinstallation
```bash
# Fresh dependencies
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

## 📝 Common Error Messages

### "ModuleNotFoundError: No module named 'numpy'"
**Solution**: `pip install numpy==1.26.4`

### "CUDA out of memory"
**Solution**: Reduce image size or adjust memory utilization

### "Request timeout after 120 seconds"
**Solution**: Use smaller images or optimize processing

### "Connection refused"
**Solution**: Check if server is running on port 5000

---

**Last Updated**: 2025-10-23
**Troubleshooting Status**: ✅ Comprehensive