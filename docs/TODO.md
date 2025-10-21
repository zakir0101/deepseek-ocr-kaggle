# DeepSeek-OCR Deployment TODO

## Current Status
- ✅ All fixes committed to Git repository (commit: 031f020)
- ✅ Old DeepSeek directory removed from server
- ✅ Fresh repository cloned on server
- ⏳ Setup script needs to be run
- ⏳ Server startup needs to be tested

## Server Environment Details
- **Server**: Vast.ai instance (ssh9.vast.ai:15594)
- **Working Directory**: `/opt/workspace-internal/DeepSeek-OCR-vllm`
- **Model Path**: `/opt/workspace-internal/deepseek-ocr-model`
- **Python**: 3.10 (server environment)

## Issues Fixed So Far

### 1. vLLM Import Issues
- Fixed `ModelRegistry` import: `from vllm.model_executor.models.registry import ModelRegistry`
- Fixed `SamplingMetadata` import: `from vllm.v1.sample.metadata import SamplingMetadata`
- Fixed processor import: `from process.image_process import DeepseekOCRProcessor`

### 2. Multimodal Processor Compatibility
- Updated `_cached_apply_hf_processor` method signature for vLLM 0.11.0
- Updated `_call_hf_processor` method signature to accept `tok_kwargs` parameter

### 3. Dependency Issues
- **NumPy**: Downgraded to 1.26.4 for compatibility with compiled modules
- **Missing dependencies**: Added `matplotlib`, `easydict`, `addict`
- **Flash Attention**: Made optional with fallback implementation

### 4. Triton Compilation
- Installed gcc and g++ compilers on server
- Fixed Triton kernel compilation issues

## Files Modified

### `deepseek_ocr.py`
- Fixed SamplingMetadata import
- Fixed multimodal processor method signatures
- Added proper error handling for missing dependencies

### `vast_server.py`
- Fixed ModelRegistry import
- Fixed processor import
- Added graceful handling for optional flash-attn
- Improved error reporting

### `vast_setup.py`
- Updated to install latest vLLM
- Added missing dependencies: `matplotlib`, `easydict`, `addict`
- Forces NumPy 1.26.4 for compatibility
- Added ngrok setup

### `deepencoder/sam_vary_sdpa.py` & `deepencoder/clip_sdpa.py`
- Made flash_attn imports optional
- Added fallback to standard attention

## Next Steps

### 1. Run Setup Script
```bash
cd /opt/workspace-internal/DeepSeek-OCR-vllm
python vast_setup.py
```

**Expected Setup Steps:**
- Create directories: uploads, outputs
- Install dependencies (numpy 1.26.4, vllm, flask, etc.)
- Download model if not exists
- Setup ngrok for public access

### 2. Test Server Startup
```bash
cd /opt/workspace-internal/DeepSeek-OCR-vllm
python vast_server.py
```

**Expected Server Output:**
- ✓ Using numpy version: 1.26.4
- ✓ vLLM modules imported successfully
- ✓ DeepSeek OCR modules imported successfully
- ✓ Model initialization complete!
- ✓ Ngrok public URL displayed
- Server running on http://localhost:5000

### 3. Verify Health Endpoint
```bash
curl http://localhost:5000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-21T..."
}
```

## Known Issues & Solutions

### Issue: NumPy Version Mismatch
**Problem**: Server shows NumPy 2.2.6 instead of 1.26.4
**Solution**: Force reinstall numpy 1.26.4
```bash
pip install --force-reinstall numpy==1.26.4
```

### Issue: Missing Dependencies
**Problem**: Import errors for matplotlib, easydict, addict
**Solution**: Install missing dependencies
```bash
pip install matplotlib easydict addict
```

### Issue: Flash Attention Not Available
**Problem**: flash_attn module not found
**Solution**: Continue without flash_attn (already handled in code)

### Issue: Model Architecture Not Recognized
**Problem**: DeepseekOCRForCausalLM not in vLLM supported architectures
**Solution**: Ensure model is properly registered in vLLM registry

### Issue: Triton Compilation
**Problem**: Failed to find C compiler
**Solution**: gcc and g++ already installed on server

## Testing Commands

### Test Import
```bash
python test_import.py
```

### Test Health Endpoint
```bash
curl http://localhost:5000/health
```

### Test OCR Endpoint
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/ocr/image
```

## Git Commands for Updates

### Pull Latest Changes
```bash
git pull origin master
```

### Check Status
```bash
git status
git log --oneline -5
```

### Reset if Needed
```bash
git reset --hard origin/master
```

## Environment Variables
- `CUDA_HOME=/usr/local/cuda`
- `PYTHONPATH=/opt/workspace-internal/DeepSeek-OCR-vllm`

## Model Information
- **Model Repository**: deepseek-ai/DeepSeek-OCR
- **Local Path**: /opt/workspace-internal/deepseek-ocr-model
- **Architecture**: DeepseekOCRForCausalLM
- **Required Files**: config.json, modeling files, tokenizer

## Troubleshooting

### Server Won't Start
1. Check NumPy version: `python -c "import numpy; print(numpy.__version__)"`
2. Check vLLM imports: `python -c "from vllm import AsyncLLMEngine; print('vLLM OK')"`
3. Check OCR imports: `python test_import.py`
4. Check model path exists: `ls -la /opt/workspace-internal/deepseek-ocr-model/`

### Model Loading Issues
1. Verify model files: `find /opt/workspace-internal/deepseek-ocr-model -name "*.json" -o -name "*.bin" | head -10`
2. Check model registration in vLLM
3. Verify processor compatibility

### Dependencies Issues
1. Reinstall dependencies: `pip install -r requirements.txt`
2. Force NumPy: `pip install --force-reinstall numpy==1.26.4`
3. Install missing packages as they appear

## Success Criteria
- [ ] Server starts without errors
- [ ] Model loads successfully
- [ ] Health endpoint returns "healthy" and "model_loaded": true
- [ ] OCR endpoint accepts images and returns markdown
- [ ] Ngrok tunnel provides public access URL

---
*Last Updated: 2025-10-21*
*Commit: 031f020 - Fix DeepSeek-OCR deployment issues*