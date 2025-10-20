# 🧠 DeepSeek OCR - Kaggle Model Setup Guide

This guide explains how to set up the DeepSeek-OCR model on Kaggle, including model selection and storage options.

## 📋 Model Options

### **Option 1: HuggingFace Hub (Recommended for Testing)**
- **Pros**: No storage needed, automatic download
- **Cons**: Requires internet, slower first run
- **Model**: `deepseek-ai/DeepSeek-OCR`

### **Option 2: Kaggle Dataset (Recommended for Production)**
- **Pros**: Faster loading, no internet required
- **Cons**: Requires dataset creation
- **Model**: Upload model to Kaggle dataset

### **Option 3: Manual Download**
- **Pros**: Full control
- **Cons**: Complex setup
- **Model**: Download from HuggingFace manually

## 🚀 Quick Start - HuggingFace Hub

**Use this for quick testing:**

```python
# Set before running kaggle_setup.py
import os
os.environ['MODEL_PATH'] = 'deepseek-ai/DeepSeek-OCR'

# Then run setup
!python kaggle_setup.py
```

**That's it!** The model will download automatically when the server starts.

## 📦 Advanced Setup - Kaggle Dataset

### Step 1: Create Model Dataset

1. **Go to [Kaggle Datasets](https://www.kaggle.com/datasets)**
2. **Click "New Dataset"**
3. **Upload model files:**

**Required files from HuggingFace:**
```
config.json
model.safetensors (or pytorch_model.bin)
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.json
merges.txt
```

### Step 2: Download Model to Dataset

**Method A: Using huggingface_hub (in Kaggle)**
```python
from huggingface_hub import snapshot_download

# Download model
model_path = snapshot_download(
    repo_id="deepseek-ai/DeepSeek-OCR",
    local_dir="/kaggle/working/deepseek-model",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {model_path}")
```

**Method B: Using git (in terminal)**
```bash
# Install git-lfs first
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

### Step 3: Upload to Kaggle Dataset

1. **Compress the model folder:**
   ```bash
   tar -czf deepseek-ocr-model.tar.gz DeepSeek-OCR/
   ```

2. **Upload to Kaggle dataset**
3. **Attach dataset to your notebook**

### Step 4: Configure in Notebook

```python
# Set before running kaggle_setup.py
import os

# Set model path to your dataset
os.environ['MODEL_PATH'] = '/kaggle/input/deepseek-ocr-model'

# Or if you have multiple versions:
os.environ['MODEL_PATH'] = '/kaggle/input/your-dataset-name/DeepSeek-OCR'

# Then run setup
!python kaggle_setup.py
```

## 🔧 Model Configuration

### Available Models

| Model | Size | VRAM Required | Best For |
|-------|------|---------------|----------|
| `deepseek-ai/DeepSeek-OCR` | ~7B | 16GB+ | General OCR |
| Quantized versions | ~4GB | 8GB | Kaggle T4 |

### Memory Optimization

For Kaggle T4 (16GB VRAM):

```python
# In kaggle_server.py - adjust these settings:
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    gpu_memory_utilization=0.7,  # Lower for stability
    max_model_len=4096,          # Reduce context length
    tensor_parallel_size=1,      # Single GPU
)
```

## 🛠️ Complete Kaggle Notebook Example

```python
# Cell 1: Setup everything (run ONCE)
!python kaggle_setup.py
```

```python
# Cell 2: Start server (run whenever)
!python kaggle_server.py
```

**That's it!** The setup script handles everything automatically:
- ✅ Repository cloning
- ✅ Dependency installation
- ✅ Ngrok configuration
- ✅ Model setup
- ✅ Environment configuration

## ⚠️ Common Model Issues

### 1. **Out of Memory (OOM)**
**Symptoms**: CUDA out of memory error
**Solutions**:
- Reduce `gpu_memory_utilization` to 0.6
- Use smaller images
- Reduce `MAX_CONCURRENCY` in config

### 2. **Model Download Failed**
**Symptoms**: Connection timeout
**Solutions**:
- Use Kaggle dataset method
- Enable internet in notebook settings
- Try different HuggingFace mirror

### 3. **vLLM Compatibility**
**Symptoms**: Model loading errors
**Solutions**:
- Ensure vLLM version matches model requirements
- Check CUDA compatibility
- Use `trust_remote_code=True`

## 📊 Performance Tips

### For Kaggle T4 GPU:
- **Batch Size**: 1-2 concurrent requests
- **Image Size**: Keep under 1024x1024
- **Memory**: Monitor VRAM usage
- **Timeout**: Set reasonable timeouts (30-60s)

### Model Loading Times:
- **HuggingFace**: 2-5 minutes first time
- **Kaggle Dataset**: 30-60 seconds
- **Subsequent runs**: 10-30 seconds

## 🔄 Model Updates

To update the model:

1. **HuggingFace**: Automatic on server restart
2. **Kaggle Dataset**: Create new dataset version
3. **Manual**: Re-download and re-upload

## 🎯 Best Practices

1. **Use Kaggle Datasets** for production
2. **Monitor VRAM usage** in notebook
3. **Set reasonable timeouts**
4. **Keep model files compressed**
5. **Test with sample images** first

## 📞 Troubleshooting

### Model Not Found
```python
# Check if model path is correct
import os
print(f"Model path: {os.environ.get('MODEL_PATH')}")

# Test model loading
from transformers import AutoModel
try:
    model = AutoModel.from_pretrained(os.environ['MODEL_PATH'], trust_remote_code=True)
    print("✓ Model loads successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
```

### Memory Issues
```python
# Check GPU memory
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Available: {torch.cuda.memory_allocated() / 1e9:.1f} GB used")
```

---

**💡 Pro Tip**: Start with HuggingFace Hub for testing, then move to Kaggle datasets for better performance!