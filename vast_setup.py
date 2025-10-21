#!/usr/bin/env python3
"""
DeepSeek OCR Setup Script for Vast.ai
Simplified setup without Kaggle-specific dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Vast.ai workspace directory
WORKSPACE = Path("/opt/workspace-internal")
MODEL_DIR = WORKSPACE / "deepseek-ocr-model"
UPLOAD_DIR = WORKSPACE / "uploads"
OUTPUT_DIR = WORKSPACE / "outputs"

MODEL_REPO = "deepseek-ai/DeepSeek-OCR"


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")

    # Force numpy 1.26.4 for compatibility
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Install core dependencies
    dependencies = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "vllm>=0.3.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "pillow>=10.0.0",
        "pyngrok>=7.0.0",
        "requests>=2.31.0"
    ]

    for dep in dependencies:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", dep
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✓ Dependencies installed successfully")


def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")

    for directory in [MODEL_DIR, UPLOAD_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")


def download_model():
    """Download DeepSeek OCR model if not exists"""
    print("Checking model...")

    # Check if model already exists
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"✓ Model already exists at {MODEL_DIR}")
        return

    print("Downloading DeepSeek OCR model...")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✓ Model downloaded to {MODEL_DIR}")

    except Exception as e:
        print(f"✗ Model download failed: {e}")
        print("Please download manually or check internet connection")
        sys.exit(1)


def setup_ngrok():
    """Setup ngrok for public access"""
    print("Setting up ngrok...")

    # Ngrok token - replace with your token
    NGROK_TOKEN = "2Xggvjlzi2yhVSoKzaxbqGdw3hq_hu2s9JyNg54nyvSaEhai"

    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(NGROK_TOKEN)
        print("✓ Ngrok authentication configured")
    except Exception as e:
        print(f"✗ Ngrok setup failed: {e}")
        print("Please check ngrok token and internet connection")


def main():
    """Main setup function"""
    print("=" * 50)
    print("DeepSeek OCR - Vast.ai Setup")
    print("=" * 50)

    # Setup directories first
    setup_directories()

    # Install dependencies
    install_dependencies()

    # Download model
    download_model()

    # Setup ngrok
    setup_ngrok()

    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print(f"Model path: {MODEL_DIR}")
    print(f"Uploads: {UPLOAD_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: python vast_server.py")
    print("2. Access via ngrok URL or PORTAL_CONFIG")


if __name__ == "__main__":
    main()