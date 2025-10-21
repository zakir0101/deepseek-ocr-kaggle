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
    """Configure the environment for Kaggle - assumes deepseek-ocr-kaggle is cloned"""
    print("Setting up Kaggle environment...")

    # Kaggle-specific paths
    kaggle_working = Path("/kaggle/working")
    kaggle_input = Path("/kaggle/input")

    # Check if we're in the deepseek-ocr-kaggle directory
    current_dir = Path.cwd()
    if current_dir.name != "deepseek-ocr-kaggle":
        print("⚠ Warning: Not in deepseek-ocr-kaggle directory")
        print("Please ensure you're in the correct directory:")
        print(
            "!git clone https://github.com/zakir0101/deepseek-ocr-kaggle.git"
        )
        print("%cd deepseek-ocr-kaggle")

    # Create necessary directories
    (kaggle_working / "deepseek-ocr").mkdir(exist_ok=True)
    (kaggle_working / "deepseek-ocr" / "uploads").mkdir(exist_ok=True)
    (kaggle_working / "deepseek-ocr" / "outputs").mkdir(exist_ok=True)
    (kaggle_working / "deepseek-ocr" / "outputs" / "images").mkdir(
        exist_ok=True
    )

    # Set environment variables for Kaggle
    os.environ["KAGGLE_WORKING_DIR"] = str(kaggle_working / "deepseek-ocr")
    os.environ["UPLOAD_FOLDER"] = str(
        kaggle_working / "deepseek-ocr" / "uploads"
    )
    os.environ["OUTPUT_FOLDER"] = str(
        kaggle_working / "deepseek-ocr" / "outputs"
    )
    os.environ["IMAGES_FOLDER"] = str(
        kaggle_working / "deepseek-ocr" / "outputs" / "images"
    )

    print(f"Working directory: {os.environ['KAGGLE_WORKING_DIR']}")
    print("✓ Environment configured")

    return True


def install_kaggle_dependencies():
    """Install dependencies in Kaggle environment - skip already installed"""
    print("Installing Kaggle-specific dependencies...")

    # FORCE NUMPY 1.26.4 FIRST - this is critical
    print("Forcing numpy 1.26.4 installation...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "numpy==1.26.4"
        ])
        print("✓ numpy 1.26.4 installed")
    except subprocess.CalledProcessError:
        print("⚠ numpy installation failed - trying without force")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4"
        ])

    # All required dependencies for DeepSeek OCR
    # Using versions compatible with Kaggle environment
    all_deps = [
        # Core OCR dependencies
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "PyMuPDF",
        "img2pdf",
        "einops",
        "easydict",
        "addict",
        "Pillow==10.0.1",
        # Flask server dependencies
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "requests==2.31.0",
        "pyngrok==7.0.0",
        # vLLM inference
        "vllm==0.8.5",
        # Optional optimizations
        "flash-attn==2.7.3",
        # Additional dependencies for model download
        "huggingface_hub",
    ]

    # Check which dependencies are already installed
    print("Checking installed dependencies...")
    installed_packages = (
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        .decode()
        .split("\n")
    )
    installed_packages = [
        pkg.split("==")[0].lower() for pkg in installed_packages if pkg
    ]

    # Install missing dependencies
    for dep in all_deps:
        dep_name = dep.split("==")[0].lower()

        if dep_name in installed_packages:
            print(f"✓ {dep} (already installed)")
            continue

        try:
            if dep_name == "flash-attn":
                # Special handling for flash-attn
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        dep,
                        "--no-build-isolation",
                    ]
                )
            elif dep_name == "numpy":
                # Force install numpy 1.26.4 to override conflicts
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", dep]
                )
            else:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", dep]
                )
            print(f"✓ {dep}")
        except subprocess.CalledProcessError:
            if dep_name == "flash-attn":
                print(f"⚠ {dep} installation failed (optional)")
            elif dep_name == "vllm":
                print(f"✗ {dep} installation failed - will use demo mode")
            else:
                print(f"✗ {dep}")

    return True


def download_model_if_needed():
    """Download model to /kaggle/working for persistence"""
    print("Checking for model files...")

    # Model will be stored in /kaggle/working for persistence
    kaggle_working = Path("/kaggle/working")
    model_dir = kaggle_working / "deepseek-ocr-model"

    # Check if model is already downloaded in working directory
    if model_dir.exists():
        print(f"✓ Model found in working directory: {model_dir}")
        os.environ["MODEL_PATH"] = str(model_dir)
        print(f"Model path set to: {model_dir}")
        return True

    # ALWAYS download to /kaggle/working for persistence
    print("Downloading model to /kaggle/working for persistence...")

    try:
        from huggingface_hub import snapshot_download

        print("Downloading DeepSeek-OCR model (this may take a few minutes)...")
        downloaded_path = snapshot_download(
            repo_id="deepseek-ai/DeepSeek-OCR",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        os.environ["MODEL_PATH"] = str(model_dir)
        print(f"✓ Model downloaded to: {model_dir}")
        print(f"Model path set to: {model_dir}")

    except ImportError:
        print("✗ huggingface_hub not available. Installing...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"
        ])
        # Retry download
        return download_model_if_needed()
    except Exception as e:
        print(f"✗ Model download failed: {e}")
        print("Will use HuggingFace hub directly (no persistence)")
        os.environ["MODEL_PATH"] = "deepseek-ai/DeepSeek-OCR"

    return True


def setup_ngrok():
    """Setup ngrok for public access - MANDATORY with token included"""
    print("Setting up ngrok (MANDATORY)...")

    try:
        from pyngrok import ngrok

        # SET THE TOKEN FIRST - don't try to get tunnels before authentication
        print("Setting ngrok token...")
        ngrok.set_auth_token(
            "2Xggvjlzi2yhVSoKzaxbqGdw3hq_hu2s9JyNg54nyvSaEhai"
        )
        print("✓ Ngrok token configured")

        # Now check if ngrok is working
        try:
            tunnels = ngrok.get_tunnels()
            print("✓ ngrok is working")
            return True
        except Exception as e:
            print(f"⚠ ngrok tunnel check failed: {e}")
            print("But token is set - ngrok should work when server starts")
            return True

    except ImportError as e:
        print("✗ pyngrok not installed - installing now...")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "pyngrok==7.0.0",
                ]
            )
            print("✓ pyngrok installed")
            # Retry setup
            return setup_ngrok()
        except subprocess.CalledProcessError:
            print("✗ Failed to install pyngrok")
            raise Exception(
                "NGROK SETUP FAILED - Could not install pyngrok"
            ) from e


def main():
    """Main setup function for Kaggle - robust with mandatory ngrok"""
    print("DeepSeek OCR - Kaggle Environment Setup")
    print("=" * 50)

    # Run setup steps
    steps = [
        ("Environment Setup", setup_kaggle_environment),
        ("Dependencies", install_kaggle_dependencies),
        ("Model Check", download_model_if_needed),
        ("Ngrok Setup (MANDATORY)", setup_ngrok),
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
                print(
                    "Please fix the ngrok authentication issue and run setup again."
                )
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

