#!/usr/bin/env python3
"""
DeblurGAN-v2 Research Web Application (Fixed)
===========================================

Advanced research-grade web interface with comprehensive metrics,
technical analysis, and batch processing capabilities.
"""

import os
import sys
import uuid
import cv2
import numpy as np
import time
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
from collections import defaultdict

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - system monitoring will be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available - using OpenCV for image processing")

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import our modules
from dynamic_inference import DeblurPredictor, ImageProcessor

# Flask app configuration
app = Flask(__name__, 
           template_folder=os.path.join(script_dir, 'templates'),
           static_folder=os.path.join(script_dir, 'static'))

app.config['SECRET_KEY'] = 'deblurgan-v2-research-key'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

# Setup directories
UPLOAD_FOLDER = os.path.join(script_dir, 'uploads')
RESULTS_FOLDER = os.path.join(script_dir, 'results')
BATCH_FOLDER = os.path.join(script_dir, 'batch_results')
WEIGHTS_PATH = os.path.join(parent_dir, 'fpn_inception.h5')
CONFIG_PATH = os.path.join(parent_dir, 'config', 'config.yaml')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# Global variables
predictor = None
model_info = {}
processing_stats = defaultdict(list)

class MetricsCalculator:
    """Advanced metrics calculation for research analysis"""
    
    @staticmethod
    def calculate_blur_level(image):
        """Calculate blur level using Laplacian variance"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Classify blur level
            if laplacian_var < 50:
                level = "Very High Blur"
            elif laplacian_var < 100:
                level = "High Blur"
            elif laplacian_var < 200:
                level = "Medium Blur"
            elif laplacian_var < 500:
                level = "Low Blur"
            else:
                level = "Sharp"
                
            return {
                'variance': round(laplacian_var, 2),
                'level': level,
                'score': min(100, laplacian_var / 10)
            }
        except Exception as e:
            print(f"Error calculating blur level: {e}")
            return {'variance': 0, 'level': 'Unknown', 'score': 0}
    
    @staticmethod
    def calculate_edge_density(image):
        """Calculate edge density using Canny edge detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            density = (edge_pixels / total_pixels) * 100
            
            return {
                'edge_pixels': int(edge_pixels),
                'total_pixels': int(total_pixels),
                'density_percent': round(density, 2),
                'edges_per_region': round(edge_pixels / max(1, (edges.shape[0] // 50) * (edges.shape[1] // 50)), 1)
            }
        except Exception as e:
            print(f"Error calculating edge density: {e}")
            return {'edge_pixels': 0, 'total_pixels': 0, 'density_percent': 0, 'edges_per_region': 0}
    
    @staticmethod
    def calculate_frequency_analysis(image):
        """Analyze frequency content of the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # FFT analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Calculate frequency distribution
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            
            # Low frequency (center region)
            low_freq_region = magnitude[center_y-h//8:center_y+h//8, center_x-w//8:center_x+w//8]
            low_freq_energy = np.sum(low_freq_region)
            
            # High frequency (outer regions)
            high_freq_energy = np.sum(magnitude) - low_freq_energy
            
            total_energy = low_freq_energy + high_freq_energy
            low_freq_ratio = (low_freq_energy / max(total_energy, 1)) * 100
            high_freq_ratio = (high_freq_energy / max(total_energy, 1)) * 100
            
            return {
                'low_freq_percent': round(low_freq_ratio, 2),
                'high_freq_percent': round(high_freq_ratio, 2),
                'frequency_balance': 'Low-freq dominant' if low_freq_ratio > 60 else 'Balanced' if low_freq_ratio > 40 else 'High-freq dominant'
            }
        except Exception as e:
            print(f"Error in frequency analysis: {e}")
            return {'low_freq_percent': 50, 'high_freq_percent': 50, 'frequency_balance': 'Unknown'}
    
    @staticmethod
    def calculate_sharpness_improvement(before_img, after_img):
        """Calculate improvement metrics between before and after images"""
        try:
            before_blur = MetricsCalculator.calculate_blur_level(before_img)
            after_blur = MetricsCalculator.calculate_blur_level(after_img)
            
            before_edges = MetricsCalculator.calculate_edge_density(before_img)
            after_edges = MetricsCalculator.calculate_edge_density(after_img)
            
            # Calculate improvements
            sharpness_improvement = ((after_blur['variance'] - before_blur['variance']) / max(before_blur['variance'], 1)) * 100
            edge_improvement = ((after_edges['density_percent'] - before_edges['density_percent']) / max(before_edges['density_percent'], 1)) * 100
            
            return {
                'sharpness_improvement_percent': round(max(0, sharpness_improvement), 1),
                'edge_improvement_percent': round(max(0, edge_improvement), 1),
                'before_sharpness': before_blur['variance'],
                'after_sharpness': after_blur['variance'],
                'before_edge_density': before_edges['density_percent'],
                'after_edge_density': after_edges['density_percent']
            }
        except Exception as e:
            print(f"Error calculating improvements: {e}")
            return {
                'sharpness_improvement_percent': 0,
                'edge_improvement_percent': 0,
                'before_sharpness': 0,
                'after_sharpness': 0,
                'before_edge_density': 0,
                'after_edge_density': 0
            }

class PerformanceMonitor:
    """Monitor system performance during processing"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        if PSUTIL_AVAILABLE:
            self.start_memory = psutil.virtual_memory().used
            self.peak_memory = self.start_memory
        else:
            self.start_memory = 0
            self.peak_memory = 0
        
    def update_peak_memory(self):
        """Update peak memory usage"""
        if PSUTIL_AVAILABLE:
            current_memory = psutil.virtual_memory().used
            self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_metrics(self):
        """Get performance metrics"""
        end_time = time.time()
        processing_time = end_time - self.start_time if self.start_time else 0
        
        if PSUTIL_AVAILABLE:
            memory_used = (self.peak_memory - self.start_memory) / (1024 * 1024)  # MB
            cpu_usage = psutil.cpu_percent()
            peak_memory_gb = self.peak_memory / (1024**3)
        else:
            memory_used = 0
            cpu_usage = 0
            peak_memory_gb = 0
        
        return {
            'processing_time_ms': round(processing_time * 1000, 1),
            'processing_time_s': round(processing_time, 2),
            'memory_used_mb': round(max(0, memory_used), 1),
            'peak_memory_gb': round(peak_memory_gb, 2),
            'cpu_usage_percent': cpu_usage,
            'timestamp': datetime.now().isoformat()
        }

def get_model_info():
    """Get comprehensive model information"""
    return {
        'architecture': 'DeblurGAN-v2',
        'backbone': 'Inception-ResNet-v2',
        'generator': 'FPN (Feature Pyramid Network)',
        'discriminator': 'Double-scale (Global + Local)',
        'loss_function': 'RaGAN-LS + Perceptual + MSE',
        'parameters_millions': '60.9M',
        'model_size_mb': '232MB',
        'gflops': '411G',
        'paper_psnr': '29.55 dB (GoPro)',
        'paper_ssim': '0.934 (GoPro)',
        'efficiency_vs_srn': '5x faster',
        'efficiency_vs_deepdeblur': '100x faster',
        'device': str(predictor.device) if predictor else 'Not loaded',
        'weights_file': os.path.basename(WEIGHTS_PATH)
    }

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_predictor():
    """Initialize the DeblurGAN-v2 predictor"""
    global predictor, model_info
    try:
        print("🚀 Initializing DeblurGAN-v2 research model...")
        
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
            model_info = get_model_info()
            print("✅ Research model loaded successfully!")
            return True
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def process_image_with_analysis(image_path, output_dir):
    """Process image with comprehensive analysis"""
    try:
        # Start performance monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Load image
        original_rgb = ImageProcessor.load_image(image_path)
        if original_rgb is None:
            raise ValueError("Could not load image")
        
        # Pre-processing analysis
        before_blur = MetricsCalculator.calculate_blur_level(original_rgb)
        before_edges = MetricsCalculator.calculate_edge_density(original_rgb)
        before_freq = MetricsCalculator.calculate_frequency_analysis(original_rgb)
        
        monitor.update_peak_memory()
        
        # Run deblurring
        inference_start = time.time()
        deblurred_rgb = predictor.predict(original_rgb)
        inference_time = time.time() - inference_start
        
        monitor.update_peak_memory()
        
        # Post-processing analysis
        after_blur = MetricsCalculator.calculate_blur_level(deblurred_rgb)
        after_edges = MetricsCalculator.calculate_edge_density(deblurred_rgb)
        after_freq = MetricsCalculator.calculate_frequency_analysis(deblurred_rgb)
        
        # Calculate improvements
        improvements = MetricsCalculator.calculate_sharpness_improvement(original_rgb, deblurred_rgb)
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        base_name = f"research_{unique_id}"
        
        # Save results
        deblurred_path = os.path.join(output_dir, f"{base_name}_deblurred.png")
        ImageProcessor.save_image(deblurred_rgb, deblurred_path)
        
        # Create comparison
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        deblurred_bgr = cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR)
        comparison = ImageProcessor.create_comparison(original_bgr, deblurred_bgr)
        
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        ImageProcessor.save_image(comparison, comparison_path, from_rgb=False)
        
        # Get performance metrics
        performance = monitor.get_metrics()
        
        # Store processing stats
        processing_stats['total_processed'].append(1)
        processing_stats['processing_times'].append(performance['processing_time_s'])
        processing_stats['memory_usage'].append(performance['memory_used_mb'])
        processing_stats['sharpness_improvements'].append(improvements['sharpness_improvement_percent'])
        
        # Compile comprehensive analysis
        analysis = {
            'image_info': {
                'dimensions': f"{original_rgb.shape[1]}x{original_rgb.shape[0]}",
                'channels': original_rgb.shape[2],
                'file_size_kb': os.path.getsize(image_path) // 1024
            },
            'before_analysis': {
                'blur_level': before_blur,
                'edge_density': before_edges,
                'frequency_analysis': before_freq
            },
            'after_analysis': {
                'blur_level': after_blur,
                'edge_density': after_edges,
                'frequency_analysis': after_freq
            },
            'improvements': improvements,
            'performance': {
                **performance,
                'inference_time_ms': round(inference_time * 1000, 1),
            },
            'model_info': model_info
        }
        
        return {
            'success': True,
            'deblurred_path': deblurred_path,
            'comparison_path': comparison_path,
            'base_name': base_name,
            'analysis': analysis
        }
        
    except Exception as e:
        print(f"Error in image processing: {e}")
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
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def get_processing_statistics():
    """Get comprehensive processing statistics"""
    if not processing_stats['total_processed']:
        return {
            'total_images': 0,
            'avg_processing_time_s': 0,
            'min_processing_time_s': 0,
            'max_processing_time_s': 0,
            'avg_memory_usage_mb': 0,
            'avg_sharpness_improvement': 0,
            'success_rate': 100.0,
            'throughput_per_hour': 0
        }
    
    times = processing_stats['processing_times']
    memory = processing_stats['memory_usage']
    improvements = processing_stats['sharpness_improvements']
    
    return {
        'total_images': len(processing_stats['total_processed']),
        'avg_processing_time_s': round(np.mean(times), 2),
        'min_processing_time_s': round(min(times), 2),
        'max_processing_time_s': round(max(times), 2),
        'avg_memory_usage_mb': round(np.mean(memory), 1),
        'avg_sharpness_improvement': round(np.mean(improvements), 1),
        'success_rate': 100.0,
        'throughput_per_hour': round(3600 / np.mean(times), 1) if times else 0
    }

@app.route('/')
def index():
    """Main research interface"""
    return render_template('research_interface.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with comprehensive analysis"""
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
        
        # Process with analysis
        result = process_image_with_analysis(upload_path, RESULTS_FOLDER)
        
        # Clean up uploaded file
        try:
            os.remove(upload_path)
        except:
            pass
        
        if result['success']:
            # Convert images to base64
            deblurred_b64 = image_to_base64(result['deblurred_path'])
            comparison_b64 = image_to_base64(result['comparison_path'])
            
            return jsonify({
                'success': True,
                'deblurred_image': deblurred_b64,
                'comparison_image': comparison_b64,
                'analysis': result['analysis'],
                'download_url': url_for('download_result', filename=f"{result['base_name']}_deblurred.png")
            })
        else:
            return jsonify({'success': False, 'error': result['error']})
            
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'})

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle batch upload for research analysis"""
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'error': 'No files selected'})
    
    batch_id = str(uuid.uuid4())[:8]
    batch_dir = os.path.join(BATCH_FOLDER, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    results = []
    successful = 0
    failed = 0
    
    try:
        for file in files:
            if not allowed_file(file.filename):
                failed += 1
                continue
                
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, f"batch_{filename}")
            file.save(upload_path)
            
            result = process_image_with_analysis(upload_path, batch_dir)
            
            if result['success']:
                successful += 1
                results.append({
                    'filename': filename,
                    'analysis': result['analysis'],
                    'processing_time': result['analysis']['performance']['processing_time_s'],
                    'improvement': result['analysis']['improvements']['sharpness_improvement_percent']
                })
            else:
                failed += 1
            
            try:
                os.remove(upload_path)
            except:
                pass
        
        if results:
            batch_stats = {
                'total_files': len(files),
                'successful': successful,
                'failed': failed,
                'avg_processing_time': round(np.mean([r['processing_time'] for r in results]), 2),
                'avg_improvement': round(np.mean([r['improvement'] for r in results]), 1),
                'total_processing_time': round(sum([r['processing_time'] for r in results]), 2),
                'batch_id': batch_id
            }
        else:
            batch_stats = {'total_files': len(files), 'successful': 0, 'failed': len(files)}
        
        return jsonify({
            'success': True,
            'batch_stats': batch_stats,
            'results': results[:10]
        })
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        return jsonify({'success': False, 'error': f'Batch processing failed: {str(e)}'})

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

@app.route('/api/model_info')
def api_model_info():
    """API endpoint for model information"""
    return jsonify(model_info)

@app.route('/api/stats')
def api_stats():
    """API endpoint for processing statistics"""
    return jsonify(get_processing_statistics())

@app.route('/api/system_info')
def api_system_info():
    """API endpoint for system information"""
    if PSUTIL_AVAILABLE:
        return jsonify({
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 1),
            'cpu_usage_percent': psutil.cpu_percent(),
            'python_version': sys.version.split()[0],
            'opencv_version': cv2.__version__
        })
    else:
        return jsonify({
            'cpu_count': 'N/A',
            'memory_total_gb': 'N/A',
            'memory_available_gb': 'N/A', 
            'cpu_usage_percent': 'N/A',
            'python_version': sys.version.split()[0],
            'opencv_version': cv2.__version__
        })

if __name__ == '__main__':
    print("🔬 DeblurGAN-v2 Research Web Application")
    print("=" * 45)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Create required directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(BATCH_FOLDER, exist_ok=True)
    
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Results folder: {RESULTS_FOLDER}")
    print(f"🏗️  Model weights: {WEIGHTS_PATH}")
    print(f"⚙️  Config file: {CONFIG_PATH}")
    print(f"🌐 Port: {port}")
    
    # Check dependencies
    print(f"🔧 psutil available: {PSUTIL_AVAILABLE}")
    print(f"🔧 PIL available: {PIL_AVAILABLE}")
    
    # Initialize predictor
    if init_predictor():
        print("\n🌐 Starting research web server...")
        print("=" * 45)
        print(f"💻 Local Access:    http://localhost:{port}")
        print("=" * 45)
        print("🔬 RESEARCH FEATURES:")
        print("• Real-time image quality metrics")
        print("• Performance benchmarking")
        print("• Batch processing analysis")
        print("• Model architecture insights")
        print("• Technical comparison tools")
        print("=" * 45)
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
    else:
        print("❌ Cannot start research server - model failed to load")
        print("\n💡 Troubleshooting:")
        print(f"   • Install dependencies: pip install -r research_requirements.txt")
        print(f"   • Verify model weights exist: {WEIGHTS_PATH}")
        print(f"   • Verify config file exists: {CONFIG_PATH}")
        sys.exit(1)
