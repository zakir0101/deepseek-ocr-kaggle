# DeepSeek OCR - Claude Assistant Documentation

This document provides essential information for Claude assistants to effectively debug, test, and deploy the DeepSeek OCR project.

## 🚨 CRITICAL REMINDERS

- **NEVER run code locally** - Always use deployment scripts for testing
- **ALWAYS use `deploy.sh`** for any code changes
- **NEVER read `/process/image_process.py`** - Contains malformed strings that cause infinite loops
- **ALWAYS sleep 30+ seconds** when reading Bash output from long-running processes

## 🚀 Quick Deployment Commands

### Standard Deployment
```bash
./deploy.sh -m "Your descriptive commit message"
```

### Manual Deployment (if script fails)
```bash
# Commit and push
git add .
git commit -m "Your changes"
git push origin master

# SSH to server
ssh -p 40032 zakir@223.166.245.194 -L 8080:localhost:8080 -L 5000:localhost:5000

# Deploy on server
pkill -9 python3
cd /home/zakir/deepseek-ocr-kaggle
git fetch origin && git reset --hard origin/master
python3 vast_server.py
```

## 🔧 Debugging & Testing Guide

### Server Health Check
```bash
curl http://localhost:5000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-23T..."
}
```

### Model Loading Issues
**Symptoms:**
- `model_loaded: false` in health response
- Import errors in server logs
- GPU memory allocation failures

**Debug Steps:**
1. Check GPU compatibility (RTX 3090/4090/A100 required)
2. Verify vLLM version is 0.8.5
3. Check NumPy version is 1.26.4
4. Verify model files exist in `models/deepseek-ocr/`

### OCR Processing Issues
**Symptoms:**
- Timeout errors (120s default)
- Empty results returned
- Frontend hangs on upload

**Debug Steps:**
1. Check server logs for generation errors
2. Verify image preprocessing is working
3. Check bounding box extraction
4. Test with smaller images

### Frontend Issues
**Symptoms:**
- Boxes image not displaying
- Raw vs processed results confusion
- Tab switching problems

**Debug Steps:**
1. Check browser console for errors
2. Verify API response structure matches frontend expectations
3. Test base64 image encoding
4. Check React component state management

## 📁 Critical File Information

### Server Files
- **`vast_server.py`**: Main Flask server with async vLLM integration
- **`deploy.sh`**: Auto-deployment script with SSH automation
- **`requirements.txt`**: Python dependencies (vLLM 0.8.5, NumPy 1.26.4)

### Frontend Files
- **`web-client/src/App.jsx`**: Main React component
- **`web-client/src/config.js`**: API endpoint configuration

### Model Files
- **`models/deepseek-ocr/`**: DeepSeek OCR model weights and config
- **`deepseek_ocr.py`**: Model implementation compatible with vLLM

### ⚠️ DANGEROUS FILES (DO NOT READ)
- **`/process/image_process.py`**: Contains malformed strings causing infinite loops

## 🔍 Common Error Patterns

### Import Errors
```python
# Fixed imports in vast_server.py
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
```

### Async Generation Issues
```python
# Correct async generation pattern
async def generate_ocr():
    async for request_output in engine.generate(
        request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            # Process output...
```

### Bounding Box Extraction
```python
# Box extraction from OCR output
def extract_boxes_from_ocr(raw_text):
    det_pattern = r'<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>'
    matches = re.findall(det_pattern, raw_text)
    # Convert coordinates...
```

## 🛠️ Testing Procedures

### Server Startup Test
1. Start server: `python3 vast_server.py`
2. Check health endpoint: `curl http://localhost:5000/health`
3. Verify model loads successfully

### OCR Functionality Test
1. Upload test image via frontend
2. Check all three result tabs work:
   - Rendered Markdown
   - Source Markdown
   - Raw OCR Output
3. Verify bounding boxes image displays

### Deployment Test
1. Run `./deploy.sh -m "Test deployment"`
2. Verify commit and push succeeds
3. Check server auto-restarts with new code

## 📊 Performance Monitoring

### Expected Server Output
```
✓ Using numpy version: 1.26.4
✓ vLLM modules imported successfully
✓ DeepSeek OCR modules imported successfully
✓ Model initialization complete!
Server running on: http://localhost:5000
```

### Memory Usage
- **GPU Memory**: ~17GB peak
- **Model Loading**: ~35 seconds
- **OCR Processing**: 10-60 seconds

### Timeout Settings
- **Generation Timeout**: 120 seconds
- **Request Timeout**: 60 seconds
- **Health Check**: 10 seconds

## 🔄 Code Change Guidelines

### When Modifying Server Code
1. Test async generation patterns thoroughly
2. Verify bounding box extraction still works
3. Check frontend API response compatibility
4. Use `deploy.sh` for deployment

### When Modifying Frontend Code
1. Test all three result tabs
2. Verify boxes image display
3. Check responsive design
4. Test with various image sizes

### Git Workflow
1. Always use descriptive commit messages
2. Test deployment script after changes
3. Verify server restarts successfully
4. Check health endpoint after deployment

## 🚨 Emergency Procedures

### Server Crash Recovery
1. SSH to server: `ssh -p 40032 zakir@223.166.245.194`
2. Kill old processes: `pkill -9 python3`
3. Restart server: `python3 vast_server.py`

### Model Loading Failure
1. Check GPU compatibility
2. Verify model files exist
3. Reinstall dependencies if needed
4. Check server logs for specific errors

### Frontend Issues
1. Check browser console for errors
2. Verify API endpoints are accessible
3. Test with different browsers
4. Check network connectivity

## 📝 Documentation Updates

When making significant changes:
1. Update this CLAUDE.md file
2. Update README.md for human users
3. Update relevant docs/*.md files
4. Test deployment script still works

---

**Last Updated**: 2025-10-23
**Current Status**: ✅ Fully functional
**Known Issues**: None
**Deployment Method**: `./deploy.sh`
