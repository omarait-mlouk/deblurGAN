#!/usr/bin/env python3
"""
DeblurGAN-v2 Flask Web App with Mobile Access Info
=================================================
"""

import os
import sys
import socket
import uuid
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# Get the correct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Add parent directory to path for imports
sys.path.insert(0, parent_dir)

# Import our modules
from dynamic_inference import DeblurPredictor, ImageProcessor

# Flask app configuration with correct template path
app = Flask(__name__, 
           template_folder=os.path.join(script_dir, 'templates'),
           static_folder=os.path.join(script_dir, 'static'))

app.config['SECRET_KEY'] = 'deblurgan-v2-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Setup directories
UPLOAD_FOLDER = os.path.join(script_dir, 'uploads')
RESULTS_FOLDER = os.path.join(script_dir, 'results')
WEIGHTS_PATH = os.path.join(parent_dir, 'fpn_inception.h5')
CONFIG_PATH = os.path.join(parent_dir, 'config', 'config.yaml')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# Global predictor instance
predictor = None

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        try:
            # Alternative method
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except:
            return "Unable to determine IP"

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_predictor():
    """Initialize the DeblurGAN-v2 predictor"""
    global predictor
    try:
        print("🚀 Initializing DeblurGAN-v2 model...")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🏗️  Model weights: {WEIGHTS_PATH}")
        print(f"⚙️  Config file: {CONFIG_PATH}")
        
        # Verify files exist
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        
        # Change to parent directory for model loading
        original_cwd = os.getcwd()
        os.chdir(parent_dir)
        
        try:
            predictor = DeblurPredictor(
                weights_path='fpn_inception.h5',
                device='auto'
            )
            print("✅ Model loaded successfully!")
            return True
        finally:
            # Change back to original directory
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def process_uploaded_image(image_path, output_dir):
    """Process uploaded image and return result paths"""
    try:
        # Load image
        original_rgb = ImageProcessor.load_image(image_path)
        if original_rgb is None:
            raise ValueError("Could not load image")
        
        # Run deblurring
        deblurred_rgb = predictor.predict(original_rgb)
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        base_name = f"result_{unique_id}"
        
        # Save deblurred image
        deblurred_path = os.path.join(output_dir, f"{base_name}_deblurred.png")
        ImageProcessor.save_image(deblurred_rgb, deblurred_path)
        
        # Create comparison image
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        deblurred_bgr = cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR)
        comparison = ImageProcessor.create_comparison(original_bgr, deblurred_bgr)
        
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        ImageProcessor.save_image(comparison, comparison_path, from_rgb=False)
        
        return {
            'success': True,
            'deblurred_path': deblurred_path,
            'comparison_path': comparison_path,
            'base_name': base_name
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def image_to_base64(image_path):
    """Convert image to base64 for web display"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode('utf-8')
    except:
        return None

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload JPG, PNG, BMP, or TIFF images.'})
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(upload_path)
        
        # Process image
        result = process_uploaded_image(upload_path, RESULTS_FOLDER)
        
        # Clean up uploaded file
        try:
            os.remove(upload_path)
        except:
            pass
        
        if result['success']:
            # Convert images to base64 for display
            deblurred_b64 = image_to_base64(result['deblurred_path'])
            comparison_b64 = image_to_base64(result['comparison_path'])
            
            return jsonify({
                'success': True,
                'deblurred_image': deblurred_b64,
                'comparison_image': comparison_b64,
                'download_url': url_for('download_result', filename=f"{result['base_name']}_deblurred.png")
            })
        else:
            return jsonify({'success': False, 'error': result['error']})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'})

@app.route('/download/<filename>')
def download_result(filename):
    """Download processed image"""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Download failed: {str(e)}", 500

@app.route('/status')
def status():
    """Get model status"""
    global predictor
    return jsonify({
        'model_loaded': predictor is not None,
        'device': str(predictor.device) if predictor else 'Not loaded',
        'weights_path': WEIGHTS_PATH,
        'config_path': CONFIG_PATH
    })

if __name__ == '__main__':
    print("🌐 DeblurGAN-v2 Web Application")
    print("=" * 50)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Create required directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Results folder: {RESULTS_FOLDER}")
    print(f"🏗️  Model weights: {WEIGHTS_PATH}")
    print(f"⚙️  Config file: {CONFIG_PATH}")
    print(f"🌐 Port: {port}")
    
    # Initialize predictor
    if init_predictor():
        # Get local IP address
        local_ip = get_local_ip()
        
        print("\n🌐 Starting Flask web server...")
        print("=" * 50)
        print(f"💻 Computer Access: http://localhost:{port}")
        print(f"📱 Mobile Access:   http://{local_ip}:{port}")
        print("=" * 50)
        print("📱 MOBILE SETUP INSTRUCTIONS:")
        print("1. Make sure your phone is on the same WiFi network")
        print("2. Open your mobile browser (Safari, Chrome, etc.)")
        print(f"3. Navigate to: http://{local_ip}:{port}")
        print("4. Enjoy deblurring images on your phone! 📸")
        print("=" * 50)
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print("❌ Cannot start web server - model failed to load")
        print("\n🔧 Troubleshooting:")
        print(f"   • Check if model weights exist: {WEIGHTS_PATH}")
        print(f"   • Check if config file exists: {CONFIG_PATH}")
        print("   • Ensure you're running from the webapp directory")
        sys.exit(1)
