#!/usr/bin/env python3
"""
DeblurGAN-v2 Dynamic Inference Script
====================================

A comprehensive, user-friendly inference script for DeblurGAN-v2 with flexible input options
and CPU/GPU support.

Author: Refactored for optimal performance and usability
Features:
- Multiple input methods (single image, multiple images, folders, patterns)
- CPU/GPU device selection
- Progress tracking and error handling
- Batch processing with comparison images
- Clean, modular code structure

Usage Examples:
    python dynamic_inference.py image.jpg                    # Single image
    python dynamic_inference.py img1.jpg img2.png           # Multiple images  
    python dynamic_inference.py --folder ./images           # Folder processing
    python dynamic_inference.py --pattern "*.jpg"           # Pattern matching
    python dynamic_inference.py --auto                      # Auto-discover
    python dynamic_inference.py --device gpu                # Force GPU usage
"""

import os
import sys
import cv2
import numpy as np
import torch
import yaml
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple

# Import project modules
from aug import get_normalize
from models.networks import get_generator

class DeblurPredictor:
    """
    Main predictor class for DeblurGAN-v2 inference
    Handles model loading, preprocessing, and prediction
    """
    
    def __init__(self, weights_path: str = 'fpn_inception.h5', 
                 model_name: str = '', device: str = 'auto'):
        """
        Initialize the predictor
        
        Args:
            weights_path: Path to model weights file
            model_name: Model architecture name (from config if empty)
            device: Device to use ('auto', 'cpu', 'gpu')
        """
        self.weights_path = weights_path
        self.device = self._setup_device(device)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model(model_name or self.config['model'])
        self.normalize_fn = get_normalize()
        
        print(f"✅ DeblurGAN-v2 loaded on device: {self.device}")
        print(f"🏗️  Architecture: {self.config['model']['g_name']}")
    
    def _setup_device(self, device_preference: str) -> torch.device:
        """Setup and return the appropriate device"""
        if device_preference.lower() == 'gpu':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("⚠️  GPU requested but CUDA not available. Using CPU.")
                return torch.device('cpu')
        elif device_preference.lower() == 'cpu':
            return torch.device('cpu')
        else:  # auto
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                print(f"🚀 Auto-detected GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("💻 Using CPU (GPU not available)")
            return device
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open('config/config.yaml', encoding='utf-8') as cfg:
                return yaml.load(cfg, Loader=yaml.FullLoader)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def _load_model(self, model_config: dict) -> torch.nn.Module:
        """Load and initialize the model"""
        try:
            # Get model architecture
            model = get_generator(model_config)
            
            # Load weights with device mapping
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
            
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            
            # Move to device and set to evaluation mode for inference
            model = model.to(self.device)
            model.train(True)  # Keep in train mode for proper batch norm behavior
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    @staticmethod
    def _array_to_batch(x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to batch tensor"""
        x = np.transpose(x, (2, 0, 1))  # HWC to CHW
        x = np.expand_dims(x, 0)        # Add batch dimension
        return torch.from_numpy(x)
    
    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Preprocess image for model input
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Tuple of (image_tensor, mask_tensor, original_height, original_width)
        """
        # Normalize image
        normalized_img, _ = self.normalize_fn(image, image)
        
        # Create mask (all ones for no masking)
        mask = np.ones_like(normalized_img, dtype=np.float32)
        
        # Get original dimensions
        h, w, _ = normalized_img.shape
        
        # Pad to multiples of 32 (required by model architecture)
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size
        
        pad_params = {
            'mode': 'constant',
            'constant_values': 0,
            'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
        }
        
        padded_img = np.pad(normalized_img, **pad_params)
        padded_mask = np.pad(mask, **pad_params)
        
        # Convert to tensors
        img_tensor = self._array_to_batch(padded_img)
        mask_tensor = self._array_to_batch(padded_mask)
        
        return img_tensor, mask_tensor, h, w
    
    @staticmethod
    def _postprocess(prediction: torch.Tensor, original_h: int, original_w: int) -> np.ndarray:
        """
        Postprocess model output to final image
        
        Args:
            prediction: Model output tensor
            original_h: Original image height
            original_w: Original image width
            
        Returns:
            Processed image as numpy array (H, W, C)
        """
        # Move to CPU and convert to numpy
        pred_np = prediction.detach().cpu().float().numpy()
        
        # Remove batch dimension and transpose to HWC
        pred_np = pred_np[0]  # Remove batch dim
        pred_np = np.transpose(pred_np, (1, 2, 0))  # CHW to HWC
        
        # Denormalize from [-1, 1] to [0, 255]
        pred_np = (pred_np + 1) / 2.0 * 255.0
        
        # Crop to original size and convert to uint8
        pred_np = pred_np[:original_h, :original_w, :]
        return pred_np.astype('uint8')
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run deblurring prediction on a single image
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            Deblurred image as numpy array (H, W, C) in RGB format
        """
        # Preprocess
        img_tensor, mask_tensor, h, w = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            img_input = img_tensor.to(self.device)
            prediction = self.model(img_input)
        
        # Postprocess
        return self._postprocess(prediction, h, w)

class ImageProcessor:
    """Handles image I/O and batch processing"""
    
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    @classmethod
    def find_images_in_folder(cls, folder_path: str) -> List[str]:
        """Find all supported images in a folder"""
        folder = Path(folder_path)
        if not folder.exists():
            return []
        
        images = []
        for ext in cls.SUPPORTED_EXTENSIONS:
            images.extend(folder.glob(f"*{ext}"))
            images.extend(folder.glob(f"*{ext.upper()}"))
        
        return sorted([str(img) for img in images])
    
    @classmethod
    def find_images_by_pattern(cls, pattern: str) -> List[str]:
        """Find images using glob pattern"""
        return sorted(glob.glob(pattern))
    
    @classmethod
    def auto_discover_images(cls) -> List[str]:
        """Auto-discover images in common directories"""
        search_paths = [".", "test_img/", "images/", "input/", "data/"]
        found_images = []
        
        for path in search_paths:
            if os.path.exists(path):
                for ext in cls.SUPPORTED_EXTENSIONS:
                    pattern = os.path.join(path, f"*{ext}")
                    found_images.extend(glob.glob(pattern))
                    found_images.extend(glob.glob(pattern.upper()))
        
        return sorted(list(set(found_images)))
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image and convert to RGB"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str, from_rgb: bool = True) -> bool:
        """Save image to file"""
        try:
            if from_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(output_path, image)
        except Exception:
            return False
    
    @staticmethod
    def create_comparison(original: np.ndarray, deblurred: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison image"""
        # Ensure same dimensions
        h1, w1 = original.shape[:2]
        h2, w2 = deblurred.shape[:2]
        
        if h1 != h2 or w1 != w2:
            deblurred = cv2.resize(deblurred, (w1, h1))
        
        return np.hstack((original, deblurred))

def process_images(image_paths: List[str], predictor: DeblurPredictor, 
                  output_dir: str = 'output', create_comparisons: bool = True) -> dict:
    """
    Process a list of images with the predictor
    
    Args:
        image_paths: List of image file paths
        predictor: Initialized DeblurPredictor instance
        output_dir: Output directory for results
        create_comparisons: Whether to create side-by-side comparisons
        
    Returns:
        Dictionary with processing statistics
    """
    if not image_paths:
        return {'successful': 0, 'failed': 0, 'total': 0}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    successful = 0
    failed = 0
    total = len(image_paths)
    
    print(f"📸 Processing {total} image(s)...")
    
    # Process each image with progress bar
    for img_path in tqdm(image_paths, desc="Deblurring", unit="img"):
        try:
            # Load image
            original_rgb = ImageProcessor.load_image(img_path)
            if original_rgb is None:
                print(f"❌ Could not load: {Path(img_path).name}")
                failed += 1
                continue
                
            # Run deblurring
            deblurred_rgb = predictor.predict(original_rgb)
            
            # Prepare output paths
            input_path = Path(img_path)
            output_name = input_path.stem
            
            # Save deblurred image
            deblurred_path = Path(output_dir) / f"{output_name}_deblurred{input_path.suffix}"
            if not ImageProcessor.save_image(deblurred_rgb, str(deblurred_path)):
                print(f"❌ Could not save: {deblurred_path.name}")
                failed += 1
                continue
            
            # Create and save comparison if requested
            if create_comparisons:
                original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
                deblurred_bgr = cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR)
                comparison = ImageProcessor.create_comparison(original_bgr, deblurred_bgr)
                
                comparison_path = Path(output_dir) / f"{output_name}_comparison{input_path.suffix}"
                ImageProcessor.save_image(comparison, str(comparison_path), from_rgb=False)
            
            successful += 1
            
        except Exception as e:
            print(f"❌ Error processing {Path(img_path).name}: {e}")
            failed += 1
    
    return {'successful': successful, 'failed': failed, 'total': total}

def print_summary(stats: dict, output_dir: str, create_comparisons: bool):
    """Print processing summary"""
    print(f"\n🎉 Processing Complete!")
    print(f"   ✅ Successful: {stats['successful']}")
    print(f"   ❌ Failed: {stats['failed']}")
    print(f"   📊 Success Rate: {stats['successful']/stats['total']*100:.1f}%")
    print(f"   📁 Output Directory: {output_dir}")
    
    if stats['successful'] > 0:
        print(f"\n📋 Generated Files:")
        print(f"   • *_deblurred.* ({stats['successful']} deblurred images)")
        if create_comparisons:
            print(f"   • *_comparison.* ({stats['successful']} side-by-side comparisons)")

def interactive_mode() -> List[str]:
    """Run interactive mode to select images"""
    print("🎯 DeblurGAN-v2 Interactive Mode")
    print("=" * 35)
    print("Scanning for images...")
    
    discovered_images = ImageProcessor.auto_discover_images()
    
    if not discovered_images:
        print("\n❌ No images found in common directories.")
        print("\nTry specifying images directly:")
        print("   python dynamic_inference.py image.jpg")
        print("   python dynamic_inference.py --folder /path/to/images")
        return []
    
    print(f"\n🔍 Found {len(discovered_images)} image(s):")
    for i, img in enumerate(discovered_images[:10], 1):
        print(f"   {i}. {Path(img).name}")
    if len(discovered_images) > 10:
        print(f"   ... and {len(discovered_images) - 10} more")
    
    while True:
        response = input(f"\nProcess all {len(discovered_images)} images? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            return discovered_images
        elif response in ['n', 'no']:
            print("❌ Cancelled by user")
            return []
        else:
            print("Please enter 'y' for yes or 'n' for no")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="DeblurGAN-v2 Dynamic Inference - Professional image deblurring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dynamic_inference.py image.jpg                    # Single image
  python dynamic_inference.py img1.jpg img2.png          # Multiple images  
  python dynamic_inference.py --folder ./images          # Folder processing
  python dynamic_inference.py --pattern "*.jpg"          # Pattern matching
  python dynamic_inference.py --auto                     # Auto-discover
  python dynamic_inference.py --device gpu               # Force GPU usage
  python dynamic_inference.py --folder ./images --device cpu --no-comparison
        """
    )
    
    # Input options
    parser.add_argument('images', nargs='*', help='Image file(s) to process')
    parser.add_argument('--folder', '-f', help='Folder containing images')
    parser.add_argument('--pattern', '-p', help='Glob pattern for images (e.g., "*.jpg")')
    parser.add_argument('--auto', '-a', action='store_true', help='Auto-discover images')
    
    # Processing options
    parser.add_argument('--device', '-d', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Device to use for inference (default: auto)')
    parser.add_argument('--output', '-o', default='output', 
                       help='Output directory (default: output)')
    parser.add_argument('--weights', '-w', default='fpn_inception.h5',
                       help='Path to model weights (default: fpn_inception.h5)')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip creating side-by-side comparison images')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine input images (priority: folder -> pattern -> auto -> images -> interactive)
    image_paths = []
    
    if args.folder:
        image_paths = ImageProcessor.find_images_in_folder(args.folder)
        print(f"📁 Scanning folder: {args.folder}")
        
    elif args.pattern:
        image_paths = ImageProcessor.find_images_by_pattern(args.pattern)
        print(f"🔍 Using pattern: {args.pattern}")
        
    elif args.auto:
        print("🔍 Auto-discovering images...")
        image_paths = ImageProcessor.auto_discover_images()
        
    elif args.images:
        print(f"📝 Processing specified images: {len(args.images)} file(s)")
        for img in args.images:
            if os.path.exists(img):
                image_paths.append(img)
            else:
                print(f"⚠️  Image not found: {img}")
    else:
        image_paths = interactive_mode()
    
    # Validate images found
    if not image_paths:
        print("❌ No images to process!")
        return 1
    
    print(f"🔍 Found {len(image_paths)} image(s) to process")
    
    # Initialize predictor
    try:
        predictor = DeblurPredictor(
            weights_path=args.weights,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return 1
    
    # Process images
    stats = process_images(
        image_paths=image_paths,
        predictor=predictor,
        output_dir=args.output,
        create_comparisons=not args.no_comparison
    )
    
    # Print summary
    print_summary(stats, args.output, not args.no_comparison)
    
    return 0 if stats['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
