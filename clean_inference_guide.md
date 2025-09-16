# Clean DeblurGAN-v2 Inference Guide

## 🎉 **Refactored & Clean!**

I've completely refactored your DeblurGAN-v2 inference into a single, professional-grade script with clean architecture and optimal performance.

## 🗂️ **Cleaned Directory Structure**

```
DeblurGANv2/
├── dynamic_inference.py          # ✅ Main inference script (CPU/GPU)
├── predict.py                     # 📚 Original script (kept for reference)
├── fpn_inception.h5              # 🧠 Pre-trained model weights
├── config/config.yaml            # ⚙️  Model configuration
├── models/                       # 🏗️  Model architectures
├── test_img/                     # 🖼️  Test images
├── output/                       # 📁 Results directory
└── cleanup.sh                    # 🧹 Cleanup script (run once)
```

## 🚀 **Quick Start**

### Clean up temporary files first:
```bash
chmod +x cleanup.sh && ./cleanup.sh
```

### Test the new script:
```bash
# Auto-discover and process images
python dynamic_inference.py --auto

# Process specific folder  
python dynamic_inference.py --folder test_img

# See all options
python dynamic_inference.py --help
```

## 🔧 **New Features**

### **1. Device Selection**
Choose your processing device:

```bash
# Auto-detect (default - uses GPU if available, falls back to CPU)
python dynamic_inference.py --device auto image.jpg

# Force CPU usage
python dynamic_inference.py --device cpu --folder ./images

# Force GPU usage (if available)
python dynamic_inference.py --device gpu --pattern "*.jpg"
```

### **2. Flexible Input Methods**

#### Single Image
```bash
python dynamic_inference.py photo.jpg
```

#### Multiple Images
```bash
python dynamic_inference.py img1.jpg img2.png img3.jpeg
```

#### Folder Processing
```bash
python dynamic_inference.py --folder ./vacation_photos
python dynamic_inference.py -f /path/to/images
```

#### Pattern Matching
```bash
python dynamic_inference.py --pattern "*.jpg"
python dynamic_inference.py -p "wedding_*.png"
```

#### Auto-Discovery
```bash
python dynamic_inference.py --auto
python dynamic_inference.py -a
```

#### Interactive Mode
```bash
python dynamic_inference.py
# Scans for images and asks for confirmation
```

### **3. Advanced Options**

#### Custom Output Directory
```bash
python dynamic_inference.py --folder ./images --output ./results
python dynamic_inference.py image.jpg -o ./my_output