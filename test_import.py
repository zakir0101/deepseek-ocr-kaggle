#!/usr/bin/env python3
"""Test DeepSeek OCR import"""

try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    print("✓ DeepSeek OCR import successful")
except Exception as e:
    print(f"✗ DeepSeek OCR import failed: {e}")
    import traceback
    traceback.print_exc()