#!/usr/bin/env python3
"""
DeepSeek OCR Auto-Deployment Script
Automatically sets up the complete environment including model download
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

# Configuration
BASE_DIR = Path("/home/zakir/deepseek-ocr-kaggle")
MODEL_DIR = BASE_DIR / "models" / "deepseek-ocr"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Model configuration
MODEL_REPO = "deepseek-ai/DeepSeek-OCR"
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json"
]

def run_command(cmd, description=""):
    """Run a shell command and handle output"""
    print(f"\n📦 {description}")
    print(f"   Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def create_directories():
    """Create all required directories"""
    print("\n📁 Creating directories...")

    directories = [MODEL_DIR, UPLOAD_DIR, OUTPUT_DIR]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}: {e}")
            return False

    return True

def install_dependencies():
    """Install all required dependencies"""
    print("\n🔧 Installing dependencies...")

    # Core dependencies
    dependencies = [
        "pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3",
        "pip install einops addict easydict",
        "pip install flask flask-cors Pillow",
        "pip install vllm==0.8.5",
        "pip install PyMuPDF img2pdf matplotlib"
    ]

    for dep in dependencies:
        if not run_command(dep, f"Installing: {dep.split()[2]}"):
            return False

    return True

def download_model():
    """Download the DeepSeek OCR model"""
    print(f"\n🤖 Downloading DeepSeek OCR model...")
    print(f"   Model repo: {MODEL_REPO}")
    print(f"   Target dir: {MODEL_DIR}")

    # Check if model already exists
    model_exists = all((MODEL_DIR / file).exists() for file in MODEL_FILES)
    if model_exists:
        print("   ✅ Model already exists, skipping download")
        return True

    # Download using huggingface_hub
    download_cmd = f"python3 -c \""
    download_cmd += "from huggingface_hub import snapshot_download; "
    download_cmd += f"snapshot_download(repo_id='{MODEL_REPO}', local_dir='{MODEL_DIR}', local_dir_use_symlinks=False)\""

    if run_command(download_cmd, "Downloading model files"):
        # Verify download
        downloaded_files = list(MODEL_DIR.iterdir())
        if downloaded_files:
            print(f"   ✅ Model downloaded successfully")
            print(f"   Files: {[f.name for f in downloaded_files[:5]]}...")
            return True
        else:
            print("   ❌ Model download failed - no files found")
            return False
    else:
        print("   ❌ Model download failed")
        return False

def update_server_config():
    """Update the server configuration to use the correct model path"""
    print("\n⚙️  Updating server configuration...")

    server_file = BASE_DIR / "vast_server.py"

    if not server_file.exists():
        print("   ❌ Server file not found")
        return False

    try:
        with open(server_file, 'r') as f:
            content = f.read()

        # Update model path
        old_path = 'MODEL_PATH = WORKSPACE / "deepseek-ocr-model"'
        new_path = f'MODEL_PATH = Path("{MODEL_DIR}")'

        if old_path in content:
            content = content.replace(old_path, new_path)
            print(f"   ✅ Updated model path: {MODEL_DIR}")
        else:
            print("   ℹ️  Model path already updated or not found")

        # Write updated content
        with open(server_file, 'w') as f:
            f.write(content)

        print("   ✅ Server configuration updated")
        return True

    except Exception as e:
        print(f"   ❌ Failed to update server config: {e}")
        return False

def test_imports():
    """Test if all imports work correctly"""
    print("\n🧪 Testing imports...")

    test_commands = [
        "python3 -c 'import vllm; print(\"✅ vLLM imported\")'",
        "python3 -c 'from flask import Flask; print(\"✅ Flask imported\")'",
        "python3 -c 'import torch; print(f\"✅ PyTorch {torch.__version__}\")'",
    ]

    all_success = True
    for cmd in test_commands:
        if not run_command(cmd, "Testing imports"):
            all_success = False

    return all_success

def main():
    """Main deployment function"""
    print("🚀 DeepSeek OCR Auto-Deployment")
    print("=" * 50)

    # Parse arguments
    parser = argparse.ArgumentParser(description='DeepSeek OCR Auto-Deployment')
    parser.add_argument('--skip-download', action='store_true', help='Skip model download')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    args = parser.parse_args()

    # Step 1: Create directories
    if not create_directories():
        print("\n❌ Failed to create directories")
        sys.exit(1)

    # Step 2: Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            sys.exit(1)

    # Step 3: Download model
    if not args.skip_download:
        if not download_model():
            print("\n❌ Failed to download model")
            sys.exit(1)

    # Step 4: Update server config
    if not update_server_config():
        print("\n❌ Failed to update server configuration")
        sys.exit(1)

    # Step 5: Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ Deployment completed successfully!")
    print(f"📁 Model directory: {MODEL_DIR}")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("\n🚀 To start the server:")
    print(f"   python3 {BASE_DIR / 'vast_server.py'}")
    print("\n🌐 Test the server:")
    print("   curl http://localhost:5000/health")
    print("=" * 50)

if __name__ == "__main__":
    main()