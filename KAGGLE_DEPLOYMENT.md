# 🚀 DeepSeek OCR - Kaggle Deployment Guide

**SIMPLE 2-SCRIPT WORKFLOW - NO MANUAL DEPENDENCY INSTALLATION**

## 📋 Prerequisites

- Kaggle account with GPU access (T4 x2 recommended)
- Ngrok account (free: https://ngrok.com)
- Ngrok auth token from dashboard

## 🎯 **SIMPLE 2-STEP PROCESS**

### **Step 1: Setup (Run ONCE per session)**
```python
# Cell 1: Run this ONCE per Kaggle session
!python kaggle_setup.py
```

**What this does automatically:**
- ✅ Clones DeepSeek-OCR repository if not present
- ✅ Installs ALL dependencies (skips already installed)
- ✅ Sets up Kaggle environment and directories
- ✅ Configures mandatory ngrok setup
- ✅ Detects model path (HuggingFace or local)

### **Step 2: Runtime (Run whenever needed)**
```python
# Cell 2: Run this to start the server
!python kaggle_server.py
```

**What this does:**
- ✅ Loads DeepSeek-OCR model
- ✅ Starts Flask server on port 5000
- ✅ Shows ngrok URL for public access
- ✅ Ready for OCR requests

---

## 🔧 **Complete Kaggle Notebook Example**

```python
# Cell 1: Setup (run ONCE)
!python kaggle_setup.py
```

```python
# Cell 2: Runtime (run whenever)
!python kaggle_server.py
```

**That's it!** No manual dependency installation needed.

---

## ⚙️ **Optional Configuration**

### **Custom Model Path** (before running setup)
```python
import os
os.environ['MODEL_PATH'] = '/kaggle/input/your-model-dataset'  # Local dataset
# OR
os.environ['MODEL_PATH'] = 'deepseek-ai/DeepSeek-OCR'  # HuggingFace (default)
```

### **Custom OCR Prompt**
```python
import os
os.environ['PROMPT'] = '<image>\nFree OCR.'  # Simple OCR
# OR
os.environ['PROMPT'] = '<image>\n<|grounding|>Convert the document to markdown.'  # Default
```

---

## 🌐 **Using the React Client**

### **1. Get Your Ngrok URL**
After starting the server, you'll see:
```
✓ Ngrok tunnel started: https://abc123.ngrok.io
```

### **2. Configure React Client**
In your React client, set the server URL:
```javascript
// In config.js or environment variables
const SERVER_URL = 'https://abc123.ngrok.io';
```

### **3. Start Client**
```bash
cd web-client
npm install
npm run dev
```

---

## 🛠️ **What's Installed Automatically**

The setup script installs everything:

**Core OCR Dependencies:**
- transformers==4.46.3
- tokenizers==0.20.3
- PyMuPDF, img2pdf
- einops, easydict, addict
- Pillow==10.0.1, numpy==1.24.3

**Server Dependencies:**
- flask==2.3.3, flask-cors==4.0.0
- requests==2.31.0, pyngrok==7.0.0

**Inference Engine:**
- vllm==0.8.5
- flash-attn==2.7.3 (optional optimization)

---

## ⚠️ **Important Notes**

### **Ngrok is MANDATORY**
- Setup will fail if ngrok authentication fails
- Get your free token: https://dashboard.ngrok.com/get-started/your-authtoken
- Token only needs to be set once per session

### **Setup vs Runtime**
- **Setup**: Heavy operations (5-10 min) - run ONCE
- **Runtime**: Light operations (30-60 sec) - run MULTIPLE times

### **Model Loading**
- First run: Downloads model (2-5 min)
- Subsequent runs: Loads from cache (30-60 sec)
- Use Kaggle datasets for faster loading

---

## 🔍 **Troubleshooting**

### **Setup Fails**
- Check ngrok auth token
- Ensure GPU is enabled in notebook settings
- Verify internet connection

### **Server Won't Start**
- Run setup script first
- Check if in correct directory
- Verify model path

### **Model Loading Issues**
- Use HuggingFace hub for testing
- Use Kaggle datasets for production
- Check GPU memory availability

---

## 📊 **Performance Tips**

### **For Kaggle T4 GPU:**
- Keep images under 1024x1024
- Set reasonable timeouts (30-60s)
- Monitor VRAM usage

### **Memory Optimization:**
- Reduce `gpu_memory_utilization` if OOM errors
- Lower `MAX_CONCURRENCY` for stability
- Use smaller batch sizes

---

## 🔄 **Workflow Summary**

### **First Time in Session:**
```python
# Cell 1
!python kaggle_setup.py

# Cell 2
!python kaggle_server.py
```

### **After Session Restart:**
```python
# Cell 1 (setup already done)
!python kaggle_server.py
```

---

**💡 Pro Tip**: Keep the setup cell and runtime cell separate. This way you can restart the server quickly without waiting for setup!

**🚨 Remember**: Kaggle sessions expire after 9 hours. Save your ngrok URL and configurations externally!