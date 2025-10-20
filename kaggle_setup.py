#!/usr/bin/env python3
"""
Kaggle Environment Setup Script for DeepSeek OCR
This script handles the specific constraints of Kaggle notebooks
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def setup_kaggle_environment():
    """Configure the environment for Kaggle - clone repo if needed"""
    print("Setting up Kaggle environment...")

    # Kaggle-specific paths
    kaggle_working = Path('/kaggle/working')
    kaggle_input = Path('/kaggle/input')

    # Clone repository if not already present
    repo_dir = kaggle_working / 'DeepSeek-OCR'
    if not repo_dir.exists():
        print("Cloning DeepSeek-OCR repository...")
        subprocess.run(['git', 'clone', '-q', 'https://github.com/deepseek-ai/DeepSeek-OCR.git'],
                      cwd=str(kaggle_working), check=True)
        print("✓ Repository cloned")
    else:
        print("✓ Repository already exists")

    # Change to vllm directory
    vllm_dir = repo_dir / 'DeepSeek-OCR-vllm'
    if vllm_dir.exists():
        os.chdir(str(vllm_dir))
        print(f"✓ Changed to directory: {vllm_dir}")
    else:
        print(f"✗ vLLM directory not found: {vllm_dir}")
        return False

    # Create necessary directories
    (kaggle_working / 'deepseek-ocr').mkdir(exist_ok=True)
    (kaggle_working / 'deepseek-ocr' / 'uploads').mkdir(exist_ok=True)
    (kaggle_working / 'deepseek-ocr' / 'outputs').mkdir(exist_ok=True)
    (kaggle_working / 'deepseek-ocr' / 'outputs' / 'images').mkdir(exist_ok=True)

    # Set environment variables for Kaggle
    os.environ['KAGGLE_WORKING_DIR'] = str(kaggle_working / 'deepseek-ocr')
    os.environ['UPLOAD_FOLDER'] = str(kaggle_working / 'deepseek-ocr' / 'uploads')
    os.environ['OUTPUT_FOLDER'] = str(kaggle_working / 'deepseek-ocr' / 'outputs')
    os.environ['IMAGES_FOLDER'] = str(kaggle_working / 'deepseek-ocr' / 'outputs' / 'images')

    print(f"Working directory: {os.environ['KAGGLE_WORKING_DIR']}")

    return True

def install_kaggle_dependencies():
    """Install dependencies in Kaggle environment - skip already installed"""
    print("Installing Kaggle-specific dependencies...")

    # All required dependencies for DeepSeek OCR
    all_deps = [
        # Core OCR dependencies
        'transformers==4.46.3',
        'tokenizers==0.20.3',
        'PyMuPDF',
        'img2pdf',
        'einops',
        'easydict',
        'addict',
        'Pillow==10.0.1',
        'numpy==1.24.3',
        # Flask server dependencies
        'flask==2.3.3',
        'flask-cors==4.0.0',
        'requests==2.31.0',
        'pyngrok==7.0.0',
        # vLLM inference
        'vllm==0.8.5',
        # Optional optimizations
        'flash-attn==2.7.3'
    ]

    # Check which dependencies are already installed
    print("Checking installed dependencies...")
    installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split('\n')
    installed_packages = [pkg.split('==')[0].lower() for pkg in installed_packages if pkg]

    # Install missing dependencies
    for dep in all_deps:
        dep_name = dep.split('==')[0].lower()

        if dep_name in installed_packages:
            print(f"✓ {dep} (already installed)")
            continue

        try:
            if dep_name == 'flash-attn':
                # Special handling for flash-attn
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', dep, '--no-build-isolation'])
            else:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', dep])
            print(f"✓ {dep}")
        except subprocess.CalledProcessError:
            if dep_name == 'flash-attn':
                print(f"⚠ {dep} installation failed (optional)")
            elif dep_name == 'vllm':
                print(f"✗ {dep} installation failed - will use demo mode")
            else:
                print(f"✗ {dep}")

    return True

def download_model_if_needed():
    """Download model if not present in Kaggle dataset"""
    print("Checking for model files...")

    # Check if model is available in Kaggle input
    kaggle_input = Path('/kaggle/input')

    # Look for common model directories
    model_dirs = list(kaggle_input.glob('*deepseek*'))

    if model_dirs:
        print(f"Found model directories: {[d.name for d in model_dirs]}")
        # Set model path to the first found directory
        model_path = str(model_dirs[0])
        os.environ['MODEL_PATH'] = model_path
        print(f"Model path set to: {model_path}")
    else:
        print("No model found in Kaggle input. Will use HuggingFace hub.")
        os.environ['MODEL_PATH'] = 'deepseek-ai/DeepSeek-OCR'

    return True

def setup_ngrok():
    """Setup ngrok for public access - MANDATORY with exception handling"""
    print("Setting up ngrok (MANDATORY)...")

    try:
        from pyngrok import ngrok

        # Check if ngrok is authenticated
        try:
            # Try to get existing tunnels
            tunnels = ngrok.get_tunnels()
            print("✓ ngrok is already configured")
            return True
        except Exception as e:
            print(f"✗ ngrok authentication failed: {e}")
            print("\nNGROK SETUP IS MANDATORY FOR PUBLIC ACCESS")
            print("Please add your ngrok authtoken to the notebook:")
            print("from pyngrok import ngrok")
            print("ngrok.set_auth_token('YOUR_AUTH_TOKEN')")
            print("\nGet your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
            raise Exception("NGROK SETUP FAILED - Authentication required") from e

    except ImportError as e:
        print("✗ pyngrok not installed - installing now...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyngrok==7.0.0'])
            print("✓ pyngrok installed")
            # Retry setup
            return setup_ngrok()
        except subprocess.CalledProcessError:
            print("✗ Failed to install pyngrok")
            raise Exception("NGROK SETUP FAILED - Could not install pyngrok") from e

def main():
    """Main setup function for Kaggle - robust with mandatory ngrok"""
    print("DeepSeek OCR - Kaggle Environment Setup")
    print("=" * 50)

    # Run setup steps
    steps = [
        ("Environment Setup", setup_kaggle_environment),
        ("Dependencies", install_kaggle_dependencies),
        ("Model Check", download_model_if_needed),
        ("Ngrok Setup (MANDATORY)", setup_ngrok)
    ]

    success = True
    for step_name, step_func in steps:
        print(f"\n[{step_name}]")
        try:
            if step_func():
                print(f"✓ {step_name} completed successfully")
            else:
                print(f"✗ {step_name} failed")
                if "MANDATORY" in step_name:
                    success = False
                    break
        except Exception as e:
            print(f"✗ {step_name} failed with error: {e}")
            if "MANDATORY" in step_name:
                print("\n❌ SETUP FAILED: Mandatory ngrok setup failed!")
                print("Please fix the ngrok authentication issue and run setup again.")
                sys.exit(1)

    if success:
        print("\n" + "=" * 50)
        print("🎉 Setup Complete!")
        print("\n📝 Next Steps:")
        print("1. Start the server: python kaggle_server.py")
        print("2. Use the React client with the ngrok URL")
        print("3. Server will be accessible at the ngrok URL")
        print("\n⚠ Note: Run this setup script ONCE per Kaggle session")
    else:
        print("\n" + "=" * 50)
        print("⚠ Setup completed with warnings")
        print("Some optional components failed but core setup is complete")

    return success

if __name__ == "__main__":
    main()