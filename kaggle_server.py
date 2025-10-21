#!/usr/bin/env python3
"""
Kaggle-compatible Flask Server for DeepSeek OCR
Optimized for Kaggle notebook environment with ngrok tunneling
"""

import asyncio
import base64
import io
import os
import re
import time
from pathlib import Path

# Import numpy first to check version
import numpy as np
print(f"Using numpy version: {np.__version__}")

import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps, ImageFont

# Import vLLM components with error handling
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.models.registry import ModelRegistry
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM not available. Running in demo mode.")
    VLLM_AVAILABLE = False

try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from process.image_process import DeepseekOCRProcessor
    OCR_AVAILABLE = True
    print("✓ DeepSeek OCR modules imported successfully")
except ImportError as e:
    print(f"Warning: DeepSeek OCR modules not available: {e}")
    print("Running in demo mode")
    OCR_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins for ngrok access

# Kaggle-specific paths
KAGGLE_WORKING = Path('/kaggle/working/deepseek-ocr')
UPLOAD_FOLDER = KAGGLE_WORKING / 'uploads'
OUTPUT_FOLDER = KAGGLE_WORKING / 'outputs'
IMAGES_FOLDER = OUTPUT_FOLDER / 'images'

# Create necessary directories
KAGGLE_WORKING.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
IMAGES_FOLDER.mkdir(exist_ok=True)

print(f"Kaggle working directory: {KAGGLE_WORKING}")
print(f"Upload folder: {UPLOAD_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")

# Global variables for model and engine
engine = None
ngrok_url = None

# Configuration - Use environment variable or fallback
MODEL_PATH = os.environ.get('MODEL_PATH', '/kaggle/working/deepseek-ocr-model')
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
CROP_MODE = True

if VLLM_AVAILABLE and OCR_AVAILABLE:
    # Register model
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def initialize_model():
    """Initialize the vLLM engine once during server startup"""
    global engine

    if not VLLM_AVAILABLE or not OCR_AVAILABLE:
        print("✗ Required dependencies not available")
        print(f"VLLM_AVAILABLE: {VLLM_AVAILABLE}")
        print(f"OCR_AVAILABLE: {OCR_AVAILABLE}")
        print("Exiting...")
        import sys
        sys.exit(1)

    print("Initializing DeepSeek-OCR model for Kaggle...")
    print(f"Using model path: {MODEL_PATH}")
    print(f"Model path exists: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        print(f"Model directory contents: {os.listdir(MODEL_PATH)}")
    else:
        print("Model directory does not exist!")

    try:
        # Initialize engine - use float16 for Tesla T4 compatibility
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
            dtype='half',  # Use float16 instead of bfloat16 for Tesla T4
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        print("✓ Model initialization complete!")

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        print("Exiting...")
        import sys
        sys.exit(1)


def start_ngrok_tunnel(port=5000):
    """Start ngrok tunnel for public access"""
    global ngrok_url

    try:
        from pyngrok import ngrok

        # Start ngrok tunnel
        public_url = ngrok.connect(port)
        ngrok_url = public_url
        print(f"✓ Ngrok tunnel started: {public_url}")

        # Print easy-to-copy URL
        print("\n" + "="*60)
        print("🌐 COPY THIS URL FOR CLIENT ACCESS:")
        print(f"{public_url}")
        print("="*60 + "\n")

        return public_url

    except Exception as e:
        print(f"✗ Ngrok tunnel failed: {e}")
        print("Server will only be accessible locally")
        return None


def get_base_url():
    """Get base URL for image links - prefer ngrok if available"""
    global ngrok_url

    if ngrok_url:
        return ngrok_url
    else:
        return request.host_url.rstrip('/')


def load_image(image_data):
    """Load image from base64 or file data"""
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Handle base64 image data
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Handle file upload
            image = Image.open(image_data)

        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image.convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def re_match(text):
    """Extract references from OCR output"""
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
    """Extract coordinates and labels from reference text"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, base_url):
    """Draw bounding boxes on image and save extracted images"""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    img_idx = 0
    image_urls = []

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20,)

                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            image_filename = f"{img_idx}.jpg"
                            image_path = IMAGES_FOLDER / image_filename
                            cropped.save(image_path)
                            image_urls.append(f"{base_url}/images/{image_filename}")
                        except Exception as e:
                            print(f"Error saving cropped image: {e}")
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, image_urls


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"




async def generate_ocr_real(image_features, prompt):
    """Generate OCR output using vLLM engine"""
    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    request_id = f"request-{int(time.time())}"
    printed_length = 0
    final_output = ""

    if image_features and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        raise ValueError("Prompt cannot be empty")

    async for request_output in engine.generate(
        request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text

    print('\n')
    return final_output


async def generate_ocr(image_features, prompt):
    """Generate OCR output - engine should always be available"""
    if engine is None:
        raise RuntimeError("Engine not initialized - server should have exited")
    return await generate_ocr_real(image_features, prompt)


@app.route('/api/ocr/image', methods=['POST'])
def ocr_image():
    """OCR endpoint for images"""
    try:
        # Get base URL for image links
        base_url = get_base_url()

        # Get image data
        if 'image' not in request.files and 'image_data' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        if 'image' in request.files:
            image_file = request.files['image']
            image = load_image(image_file)
        else:
            image_data = request.form['image_data']
            image = load_image(image_data)

        if image is None:
            return jsonify({'error': 'Failed to load image'}), 400

        # Get prompt (optional)
        prompt = request.form.get('prompt', PROMPT)

        # Process image
        image_features = None
        print(f"OCR_AVAILABLE: {OCR_AVAILABLE}, prompt contains <image>: {'<image>' in prompt}")
        if OCR_AVAILABLE and '<image>' in prompt:
            try:
                print("Creating DeepseekOCRProcessor instance...")
                # Create new processor instance like in official code
                image_features = DeepseekOCRProcessor().tokenize_with_images(
                    images=[image], bos=True, eos=True, cropping=CROP_MODE
                )
                print("✓ Image processing successful")
            except Exception as e:
                print(f"Image processing failed: {e}")
        else:
            print("Skipping image processing")

        # Generate OCR
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_out = loop.run_until_complete(generate_ocr(image_features, prompt))

        # Process results
        matches_ref, matches_images, matches_other = re_match(result_out)

        # Draw bounding boxes and save extracted images
        image_with_boxes, image_urls = draw_bounding_boxes(image, matches_ref, base_url)

        # Convert image with boxes to base64
        boxes_base64 = image_to_base64(image_with_boxes)

        # Process markdown output
        processed_markdown = result_out

        # Replace image references with web URLs
        for idx, a_match_image in enumerate(matches_images):
            if idx < len(image_urls):
                processed_markdown = processed_markdown.replace(
                    a_match_image,
                    f'![]({image_urls[idx]})\n'
                )

        # Clean up other references
        for a_match_other in matches_other:
            processed_markdown = processed_markdown.replace(a_match_other, '')

        # Additional cleanup
        processed_markdown = processed_markdown.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

        return jsonify({
            'success': True,
            'markdown': processed_markdown,
            'original_markdown': result_out,
            'image_with_boxes': boxes_base64,
            'extracted_images': image_urls,
            })

    except Exception as e:
        print(f"Error in OCR endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ocr/pdf', methods=['POST'])
def ocr_pdf():
    """Placeholder for PDF OCR endpoint"""
    return jsonify({
        'success': False,
        'message': 'PDF OCR endpoint not yet implemented'
    }), 501


@app.route('/images/<filename>')
def serve_image(filename):
    """Serve extracted images"""
    try:
        image_path = IMAGES_FOLDER / filename
        if image_path.exists():
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    print(f"Health check request from: {request.remote_addr}")
    print(f"Request headers: {dict(request.headers)}")
    return jsonify({
        'status': 'healthy',
        'model_loaded': engine is not None,
        'vllm_available': VLLM_AVAILABLE,
        'ocr_available': OCR_AVAILABLE,
        'kaggle_environment': True,
        'ngrok_url': ngrok_url,
        'timestamp': time.time()
    })


@app.route('/')
def index():
    """Simple index page"""
    return """
    <html>
        <head><title>DeepSeek OCR - Kaggle Server</title></head>
        <body>
            <h1>DeepSeek OCR Server</h1>
            <p>Server is running in Kaggle environment</p>
            <p><a href="/health">Health Check</a></p>
            <p>Use the React client to interact with the OCR API.</p>
        </body>
    </html>
    """


if __name__ == '__main__':
    print("DeepSeek OCR - Kaggle Server")
    print("=" * 50)

    # Check if we're in the right directory
    current_dir = Path(__file__).parent
    if not (current_dir / "deepseek_ocr.py").exists():
        print("Error: Not in deepseek-ocr-kaggle directory")
        print("Please ensure you're in the correct directory:")
        print("!git clone https://github.com/zakir0101/deepseek-ocr-kaggle.git")
        print("%cd deepseek-ocr-kaggle")
        sys.exit(1)

    # Initialize model on startup
    initialize_model()

    # Auto-start ngrok if available
    try:
        from pyngrok import ngrok

        # ALWAYS CREATE A NEW TUNNEL AND PRINT THE FUCKING URL
        print("Starting ngrok tunnel...")
        public_url = ngrok.connect(5000)

        print("\n" + "="*60)
        print("🌐 COPY THIS URL FOR CLIENT ACCESS:")
        print(f"{public_url}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"✗ Ngrok failed: {e}")
        print("Server will only be accessible locally at http://localhost:5000")

    print("\nStarting Flask server on http://localhost:5000")
    print("Server will be accessible at the ngrok URL if configured")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)