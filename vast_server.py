#!/usr/bin/env python3
"""
DeepSeek OCR Server for Vast.ai
Optimized for vast.ai instances with proper GPU support
"""

import os
import sys
import subprocess
import asyncio
import base64
import io
import json
from pathlib import Path
from datetime import datetime

# Vast.ai workspace paths
WORKSPACE = Path("/opt/workspace-internal")
MODEL_PATH = WORKSPACE / "deepseek-ocr-model"
UPLOAD_FOLDER = WORKSPACE / "uploads"
OUTPUT_FOLDER = WORKSPACE / "outputs"

# Import numpy - setup script already installed correct version
import numpy as np
print(f"✓ Using numpy version: {np.__version__}")

import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps, ImageFont

# Check for required modules
VLLM_AVAILABLE = False
OCR_AVAILABLE = False

try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.models.registry import ModelRegistry
    VLLM_AVAILABLE = True
    print("✓ vLLM modules imported successfully")
except ImportError as e:
    print(f"✗ vLLM import failed: {e}")

try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    from process.image_process import DeepseekOCRProcessor
    OCR_AVAILABLE = True
    print("✓ DeepSeek OCR modules imported successfully")
except ImportError as e:
    print(f"✗ DeepSeek OCR import failed: {e}")
    print("Note: flash_attn is optional and not required for basic functionality")

# Configuration
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
CROP_MODE = True

if VLLM_AVAILABLE and OCR_AVAILABLE:
    # Register model
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
else:
    print("⚠ Warning: Some modules not available, but continuing with available functionality")


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global engine instance
engine = None


def initialize_model():
    """Initialize the vLLM engine once during server startup"""
    global engine

    if not VLLM_AVAILABLE:
        print("✗ Required vLLM modules not available - cannot initialize model")
        return

    if not OCR_AVAILABLE:
        print("⚠ OCR modules not available, but continuing with vLLM initialization")

    print("\nInitializing DeepSeek-OCR model for Vast.ai...")
    print(f"Using model path: {MODEL_PATH}")
    print(f"Model path exists: {MODEL_PATH.exists()}")

    if MODEL_PATH.exists():
        contents = list(MODEL_PATH.iterdir())
        print(f"Model directory contents: {[f.name for f in contents[:10]]}")
    else:
        print("Model directory does not exist!")

    try:
        # Initialize engine - auto-detect best dtype for GPU
        engine_args = AsyncEngineArgs(
            model=str(MODEL_PATH),
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
            dtype='auto',  # Let vLLM choose optimal dtype for GPU
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        print("✓ Model initialization complete!")

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        print("Exiting...")
        sys.exit(1)


async def process_image_async(image_path, prompt=PROMPT, crop_mode=CROP_MODE):
    """Process image using DeepSeek OCR model"""
    if engine is None:
        raise ValueError("Model engine not initialized")

    try:
        # Create processor instance (not global)
        processor = DeepSeekOCRProcessor.from_pretrained(str(MODEL_PATH))

        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Prepare inputs
        inputs = processor.process_image(
            images=image,
            text=prompt,
            return_tensors='pt',
            crop_mode=crop_mode
        )

        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            stop_token_ids=[processor.tokenizer.eos_token_id]
        )

        # Generate response
        result_generator = engine.generate(
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=inputs['input_ids'],
            multi_modal_data=inputs['multi_modal_data']
        )

        # Get the result
        async for result in result_generator:
            if result.outputs:
                output_text = result.outputs[0].text

                # Extract bounding boxes if available
                boxes = []
                if hasattr(result.outputs[0], 'boxes') and result.outputs[0].boxes:
                    boxes = result.outputs[0].boxes

                return {
                    'text': output_text,
                    'boxes': boxes,
                    'success': True
                }

        return {'text': '', 'boxes': [], 'success': False}

    except Exception as e:
        print(f"Error processing image: {e}")
        return {'text': f'Error: {str(e)}', 'boxes': [], 'success': False}


def create_boxes_image(image_path, boxes, output_path):
    """Create image with bounding boxes drawn"""
    try:
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)

        # Try to load font, fallback to default
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            if len(box) >= 4:  # Ensure we have coordinates
                x1, y1, x2, y2 = box[:4]
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1, y1-15), str(i), fill='red', font=font)

        image.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating boxes image: {e}")
        return False


def image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': engine is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/ocr/image', methods=['POST'])
def ocr_image():
    """OCR endpoint for images"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"upload_{timestamp}_{image_file.filename}"
        image_path = UPLOAD_FOLDER / image_filename
        image_file.save(image_path)

        print(f"Processing image: {image_filename}")

        # Process image
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_image_async(image_path))
        loop.close()

        if not result['success']:
            return jsonify({'error': result['text']}), 500

        # Create boxes image if boxes are available
        boxes_image_base64 = ""
        if result['boxes']:
            boxes_filename = f"boxes_{timestamp}.jpg"
            boxes_path = OUTPUT_FOLDER / boxes_filename
            if create_boxes_image(image_path, result['boxes'], boxes_path):
                boxes_image_base64 = image_to_base64(boxes_path)

        return jsonify({
            'success': True,
            'markdown': result['text'],
            'boxes_image': boxes_image_base64,
            'image_name': image_filename
        })

    except Exception as e:
        print(f"OCR error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/image/<image_name>', methods=['GET'])
def serve_image(image_name):
    """Serve uploaded images"""
    try:
        image_path = UPLOAD_FOLDER / image_name
        if image_path.exists():
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'DeepSeek OCR Server - Vast.ai',
        'endpoints': {
            'health': '/health',
            'ocr': '/ocr/image',
            'images': '/image/<image_name>'
        },
        'model_loaded': engine is not None
    })


def start_ngrok():
    """Start ngrok tunnel and print URL"""
    try:
        from pyngrok import ngrok

        # Create HTTP tunnel
        tunnel = ngrok.connect(5000, "http")
        public_url = tunnel.public_url

        print("\n" + "=" * 50)
        print("NGROK PUBLIC URL:")
        print(public_url)
        print("=" * 50)
        print("\nCopy this URL to access your server from anywhere!")

        return public_url
    except Exception as e:
        print(f"✗ Ngrok tunnel failed: {e}")
        return None


def main():
    """Main server startup"""
    # Initialize model
    initialize_model()

    if engine is None:
        print("✗ Cannot start server - model not initialized")
        sys.exit(1)

    # Start ngrok tunnel
    ngrok_url = start_ngrok()

    print("\n" + "=" * 50)
    print("DeepSeek OCR Server - Vast.ai")
    print("=" * 50)
    print(f"Server running on: http://localhost:5000")
    if ngrok_url:
        print(f"Public access: {ngrok_url}")
    print("=" * 50)
    print("\nEndpoints:")
    print("  GET  /health     - Health check")
    print("  POST /ocr/image  - OCR image endpoint")
    print("  GET  /image/*    - Serve images")
    print("=" * 50)

    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()