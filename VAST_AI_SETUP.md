# Vast.ai DeepSeek OCR Setup Guide

## Current Status
- **Instance**: RTX 5080 on vast.ai (Instance ID: 27095595)
- **SSH**: `ssh -p 15594 root@ssh9.vast.ai`
- **Working Directory**: `/opt/workspace-internal`
- **Model**: Downloaded to `/opt/workspace-internal/deepseek-ocr-model`
- **Setup**: Completed (dependencies installed, model downloaded)

## Current Issue
The `vast_server.py` is failing due to import errors:
- `ModelRegistry` not found in `vllm.model_executor.model_loader`
- `deepseek_ocr` module not available

## Files on Instance
```
/opt/workspace-internal/
├── deepseek-ocr-model/     # Model files
├── uploads/                # Upload directory
├── outputs/                # Output directory
├── DeepSeek-OCR-vllm/      # Repository files
├── vast_server.py          # Main server script
└── vast_setup.py           # Setup script
```

## Next Steps
1. Fix import issues in `vast_server.py`:
   - Check correct vLLM ModelRegistry import path
   - Add DeepSeek-OCR repository to Python path

2. Test server startup:
   ```bash
   cd /opt/workspace-internal
   python vast_server.py
   ```

3. Once running, access via ngrok URL (printed on startup)

## Connection Commands
```bash
# SSH to instance
ssh -p 15594 root@ssh9.vast.ai

# Check server status
curl http://localhost:5000/health

# Deploy updated files
scp -P 15594 vast_server.py root@ssh9.vast.ai:/opt/workspace-internal/
```

## Notes
- Model already downloaded and dependencies installed
- Focus on fixing import issues in server script
- Use proper vLLM ModelRegistry import path
- Add DeepSeek-OCR repository to sys.path for imports