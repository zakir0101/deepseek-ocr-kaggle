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
import time
from pathlib import Path
from datetime import datetime

# Set vLLM to use legacy API (compatible with DeepSeek OCR)
os.environ['VLLM_USE_V1'] = '0'

# Vast.ai workspace paths
WORKSPACE = Path("/home/zakir/deepseek-ocr-kaggle")
MODEL_PATH = WORKSPACE / "models" / "deepseek-ocr"
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
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
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
        # Use the correct architecture for DeepSeek OCR
        engine_args = AsyncEngineArgs(
            model=str(MODEL_PATH),
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        print("✓ Model initialization complete!")

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        print("⚠ Continuing without model - server will start but OCR functionality will be limited")
        # Don't exit - let the server start without model


async def process_image_async(image_path, prompt=PROMPT, crop_mode=CROP_MODE):
    """Process image using DeepSeek OCR model"""
    if engine is None:
        return {
            'text': 'Model not available - server is running but OCR functionality is disabled',
            'boxes': [],
            'success': False,
            'error': 'Model engine not initialized'
        }

    print(f"Engine available: {engine is not None}")

    # Test if model is actually loaded by checking tokenizer
    try:
        from config import TOKENIZER
        print(f"Tokenizer available: {TOKENIZER is not None}")
    except:
        print("Tokenizer not available in config")

    # Debug: Check if this is a subsequent request
    print(f"Processing request at timestamp: {time.time()}")

    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Prepare inputs using the correct method from official code
        if '<image>' in prompt:
            image_features = DeepseekOCRProcessor().tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=crop_mode
            )
            request = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features}
            }
        else:
            request = {
                "prompt": prompt
            }

        # Prepare sampling parameters with logits processor
        logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})]
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            logits_processors=logits_processors,
            skip_special_tokens=False
        )

        # Generate response
        request_id = f"request-{int(time.time())}"

        print(f"Starting generation with request_id: {request_id}")

        try:
            # Generate with timeout
            async def generate_ocr():
                printed_length = 0
                final_output = ""

                async for request_output in engine.generate(
                    request,
                    sampling_params,
                    request_id
                ):
                    if request_output.outputs:
                        full_text = request_output.outputs[0].text
                        # Stream the output like official code
                        new_text = full_text[printed_length:]
                        if new_text:
                            print(new_text, end='', flush=True)
                        printed_length = len(full_text)
                        final_output = full_text
                print('\n')  # New line after generation completes
                return final_output

            # Wait for generation with 120 second timeout
            final_output = await asyncio.wait_for(generate_ocr(), timeout=120.0)

        except asyncio.TimeoutError:
            print("Generation timed out after 120 seconds")
            return {'text': 'Generation timed out', 'boxes': [], 'success': False}
        except Exception as e:
            print(f"Error during generation: {e}")
            return {'text': f'Generation error: {str(e)}', 'boxes': [], 'success': False}

        # Extract bounding boxes if available
        boxes = []
        if final_output:
            return {
                'text': final_output,
                'boxes': boxes,
                'success': True
            }
        else:
            return {'text': '', 'boxes': [], 'success': False}

    except Exception as e:
        print(f"Error processing image: {e}")
        return {'text': f'Error: {str(e)}', 'boxes': [], 'success': False}


def create_boxes_image(image_path, boxes, output_path):
    """Create image with bounding boxes drawn using original implementation logic"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        # Create semi-transparent overlay (original implementation)
        overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)

        # Try to load font, fallback to default
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        for i, box_info in enumerate(boxes):
            try:
                coordinates = box_info['coordinates']
                label = box_info['label']

                if len(coordinates) == 4:
                    x1, y1, x2, y2 = coordinates

                    # Normalize coordinates from 0-999 range to actual image dimensions (original implementation)
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    # Generate random color for each box (original implementation)
                    import numpy as np
                    color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                    color_a = color + (20,)  # Semi-transparent version

                    # Draw bounding box with semi-transparent fill (original implementation)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                    # Add label text with background (original implementation)
                    text_x = x1
                    text_y = max(0, y1 - 15)

                    try:
                        text_bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30))
                        draw.text((text_x, text_y), label, font=font, fill=color)
                    except:
                        # Fallback if font measurement fails
                        draw.text((text_x, text_y), label, font=font, fill=color)
            except Exception as e:
                print(f"Error drawing box {i}: {e}")
                continue

        # Apply the semi-transparent overlay (original implementation)
        img_draw.paste(overlay, (0, 0), overlay)
        img_draw.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating boxes image: {e}")
        return False


def re_match(text):
    """Extract <|ref|> and <|det|> tags from OCR output (official implementation)"""
    import re

    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """Extract coordinates and label from <|ref|> and <|det|> tags (official implementation)"""
    import re

    try:
        # Extract the pattern: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
        pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
        match = re.search(pattern, ref_text, re.DOTALL)
        if match:
            label_type = match.group(1)
            coords_text = match.group(2)

            # Extract coordinates from [[x1,y1,x2,y2]]
            if coords_text.startswith('[[') and coords_text.endswith(']]'):
                coords_list = eval(coords_text)
                return (label_type, coords_list)
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None

    return None


def crop_and_save_images(image_path, matches_images, output_folder):
    """Crop and save images from bounding boxes (official implementation)"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size

        # Create images directory
        images_dir = output_folder / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for idx, match_image in enumerate(matches_images):
            result = extract_coordinates_and_label(match_image, image_width, image_height)
            if result:
                label_type, points_list = result
                if label_type == 'image':
                    for points in points_list:
                        x1, y1, x2, y2 = points

                        # Normalize coordinates from 0-999 range to actual image dimensions
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)

                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(images_dir / f"{idx}.jpg")
                        except Exception as e:
                            print(f"Error cropping image {idx}: {e}")
                            continue

        return True
    except Exception as e:
        print(f"Error in crop_and_save_images: {e}")
        return False


def extract_boxes_from_ocr(raw_text):
    """Extract bounding boxes from <|ref|> and <|det|> tags in OCR output"""
    import re

    boxes = []
    # Find all <|ref|>...</|ref|><|det|>...</|det|> patterns (original implementation)
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, raw_text, re.DOTALL)

    for match in matches:
        try:
            ref_text = match[1]  # Content between <|ref|> and <|/ref|>
            det_text = match[2]  # Content between <|det|> and <|/det|>

            # Extract coordinates from <|det|>[[x1,y1,x2,y2]]<|/det|>
            if det_text.startswith('[[') and det_text.endswith(']]'):
                coords_text = det_text[2:-2]  # Remove [[ and ]]
                coords = [int(x.strip()) for x in coords_text.split(',')]
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    boxes.append({
                        'coordinates': [x1, y1, x2, y2],
                        'label': ref_text if ref_text else 'text'
                    })
        except (ValueError, IndexError, SyntaxError):
            continue

    return boxes


def process_ocr_output(raw_text):
    """Process raw OCR output to remove <|ref|> tags and clean up the markdown
    Preserves HTML tables and LaTeX equations for proper rendering"""
    import re

    # Remove all <|ref|>...</|ref|> and <|det|>...</|det|> tags
    processed = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', raw_text)
    processed = re.sub(r'<\|det\|>.*?<\|/det\|>', '', processed)

    # Clean up extra whitespace
    processed = re.sub(r'\n\s*\n', '\n\n', processed)  # Multiple newlines to double newlines
    processed = processed.strip()

    return processed


def process_ocr_for_rendering(raw_text, image_filename=None):
    """Process OCR output specifically for rendered HTML view
    This version preserves image references with proper <img> tags and handles line breaks"""
    import re

    # Extract image references using official implementation
    matches_ref, matches_images, matches_other = re_match(raw_text)

    # Start with the raw text
    processed = raw_text

    # Replace image references with proper HTML <img> tags
    for idx, a_match_image in enumerate(matches_images):
        # Extract coordinates from the image reference
        result = extract_coordinates_and_label(a_match_image, 1000, 1000)  # Use dummy dimensions for calculation
        if result:
            label_type, coords_list = result
            if label_type == 'image' and coords_list:
                # Get the first bounding box coordinates
                x1, y1, x2, y2 = coords_list[0]
                # Calculate width and height from coordinates
                width = x2 - x1
                height = y2 - y1
                # Create proper HTML img tag with server URL and dimensions
                img_tag = f'<img src="http://localhost:5000/images/{idx}.jpg" width="{width}" height="{height}" alt="Extracted image"><br>'
                processed = processed.replace(a_match_image, img_tag)

    # Remove other <|ref|> and <|det|> tags (non-image)
    for a_match_other in matches_other:
        processed = processed.replace(a_match_other, '')

    # Clean up extra whitespace and handle line breaks for HTML rendering
    processed = re.sub(r'\n\s*\n', '\n\n', processed)

    # Convert newlines to <br> tags for proper HTML rendering
    processed = processed.replace('\n', '<br>')

    processed = processed.strip()

    return processed


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

        # Process image with proper async task management
        # Use asyncio.run() for each request to avoid event loop conflicts
        result = asyncio.run(process_image_async(image_path))

        if not result['success']:
            return jsonify({'error': result['text']}), 500

        # Process the raw OCR output
        raw_result = result['text']

        # Extract image references and crop images
        matches_ref, matches_images, matches_other = re_match(raw_result)

        # Crop and save images from bounding boxes
        if matches_images:
            crop_and_save_images(image_path, matches_images, OUTPUT_FOLDER)

        # Process markdown output (clean version without <|ref|> tags)
        processed_markdown = process_ocr_output(raw_result)

        # Process for rendered view (with image references and line breaks)
        source_markdown = process_ocr_for_rendering(raw_result, image_filename)

        # Extract bounding boxes from OCR output
        boxes = extract_boxes_from_ocr(raw_result)

        # Create boxes image if boxes are available
        boxes_image_base64 = ""
        if boxes:
            boxes_filename = f"boxes_{timestamp}.jpg"
            boxes_path = OUTPUT_FOLDER / boxes_filename
            if create_boxes_image(image_path, boxes, boxes_path):
                boxes_image_base64 = image_to_base64(boxes_path)

        return jsonify({
            'success': True,
            'raw_result': raw_result,  # Original OCR output with <|ref|> tags
            'markdown': processed_markdown,  # Clean markdown without <|ref|> tags
            'source_markdown': source_markdown,  # Source markdown with HTML tables and LaTeX preserved
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


@app.route('/images/<image_id>.jpg', methods=['GET'])
def serve_cropped_image(image_id):
    """Serve cropped images from OCR processing"""
    try:
        image_path = OUTPUT_FOLDER / "images" / f"{image_id}.jpg"
        if image_path.exists():
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Cropped image not found'}), 404
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
    # Create necessary directories
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Initialize model
    initialize_model()

    if engine is None:
        print("⚠ Model not initialized - server will start but OCR functionality will be limited")

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