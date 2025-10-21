#!/bin/bash
# DeepSeek-OCR Dependencies Installation Script
# This script installs all required dependencies for DeepSeek-OCR deployment

echo "=========================================="
echo "DeepSeek-OCR Dependencies Installation"
echo "=========================================="

# Set environment variables for compatibility
export VLLM_USE_V1=0
echo "✓ Set VLLM_USE_V1=0 for legacy API compatibility"

# Force correct NumPy version (required by DeepSeek-OCR)
echo "Installing NumPy 1.26.4 (required version)..."
pip install --force-reinstall numpy==1.26.4
if [ $? -eq 0 ]; then
    echo "✓ NumPy 1.26.4 installed successfully"
else
    echo "✗ NumPy installation failed"
    exit 1
fi

# Install vLLM 0.8.5 (official supported version)
echo "Installing vLLM 0.8.5 (official supported version)..."
pip install --timeout 600 vllm==0.8.5
if [ $? -eq 0 ]; then
    echo "✓ vLLM 0.8.5 installed successfully"
else
    echo "✗ vLLM installation failed"
    exit 1
fi

# Install required packages from official DeepSeek-OCR requirements
echo "Installing required packages from official requirements..."
pip install transformers==4.46.3 tokenizers==0.20.3
pip install PyMuPDF img2pdf einops easydict addict Pillow

# Install server dependencies
echo "Installing server dependencies..."
pip install flask flask-cors pyngrok

# Install optional packages (may fail on some systems)
echo "Installing optional packages..."
pip install matplotlib || echo "⚠ matplotlib installation failed (optional)"

# Try flash-attn (optional, may fail without CUDA_HOME)
echo "Attempting to install flash-attn (optional)..."
pip install flash-attn --no-build-isolation || echo "⚠ Flash attention optional, continuing without it..."

# Verify installations
echo "Verifying installations..."
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import vllm; print(f'✓ vLLM: {vllm.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import flask; print(f'✓ Flask: {flask.__version__}')"

echo "=========================================="
echo "Dependencies installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure you have a compatible GPU (RTX 3090/4090/A100)"
echo "2. Run: python vast_server.py"
echo "3. Test: curl http://localhost:5000/health"
echo ""
echo "Compatible GPU architectures: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90"
echo "Incompatible: sm_120 (RTX 5080)"
echo "=========================================="