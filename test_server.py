#!/usr/bin/env python3
"""
Simple test server to verify basic Flask functionality
"""

import os
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': False,
        'timestamp': datetime.now().isoformat(),
        'message': 'Basic Flask server is running'
    })

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'DeepSeek OCR Test Server',
        'endpoints': {
            'health': '/health'
        },
        'model_loaded': False
    })

if __name__ == "__main__":
    print("Starting test Flask server...")
    print("Server running on: http://localhost:5000")
    print("Try: curl http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=False)