# 🌐 DeblurGAN-v2 Web Application

## 🎉 **What I Built for You**

A beautiful, user-friendly web interface for your DeblurGAN-v2 model with:

### ✨ **Features**
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **🖱️ Drag & Drop**: Simply drag images onto the upload area
- **⚡ Real-time Processing**: Live status updates during deblurring
- **👀 Side-by-side Comparison**: See before/after results clearly
- **💾 Download Results**: Get your deblurred images instantly
- **🖥️ CPU/GPU Support**: Automatic device detection and optimization
- **🔒 Secure**: Files are processed securely and cleaned up automatically

### 🎨 **Clean Interface**
- Modern gradient design
- Intuitive upload area with visual feedback
- Professional loading animations
- Clear error handling and success messages

## 🗂️ **Directory Structure**

```
webapp/
├── app.py                     # 🐍 Flask application
├── templates/
│   └── index.html            # 🎨 Web interface
├── static/                   # 📁 Static files (auto-created)
├── uploads/                  # 📤 Temporary uploads (auto-created)
├── results/                  # 📥 Processed results (auto-created)
├── webapp_requirements.txt   # 📦 Python dependencies
├── start_webapp.sh          # 🚀 Linux/Mac startup script
└── start_webapp.bat         # 🚀 Windows startup script
```

## 🚀 **Quick Start**

### **Method 1: Automatic Startup (Recommended)**

#### On Mac/Linux:
```bash
cd webapp
chmod +x start_webapp.sh
./start_webapp.sh
```

#### On Windows:
```cmd
cd webapp
start_webapp.bat
```

### **Method 2: Manual Setup**

```bash
# 1. Navigate to webapp directory
cd webapp

# 2. Install Flask requirements
pip install -r webapp_requirements.txt

# 3. Start the web server
python app.py
```

### **Method 3: From Your Virtual Environment**

```bash
# 1. Activate your environment
source myenv/bin/activate  # or conda activate myenv

# 2. Navigate and start
cd webapp
pip install flask pillow
python app.py
```

## 🌐 **Using the Web App**

### **1. Start the Server**
Run one of the startup methods above. You'll see:
```
🚀 Initializing DeblurGAN-v2 model...
✅ Model loaded successfully!
🌐 Starting Flask web server...
📱 Access the app at: http://localhost:5000
```

### **2. Open Your Browser**
Navigate to: **http://localhost:5000**

### **3. Upload an Image**
- **Drag & Drop**: Drag any image onto the upload area
- **Click to Browse**: Click "Choose Image" button
- **Supported Formats**: JPG, PNG, BMP, TIFF (max 16MB)

### **4. Watch the Magic Happen**
- ⏳ **Loading Screen**: AI processing indicator with spinner
- 🧠 **Real-time Status**: "AI is processing your image..."
- ⚡ **Processing Time**: Usually 10-30 seconds depending on CPU/GPU

### **5. View Results**
- ✨ **Deblurred Image**: Your enhanced result
- 📊 **Before/After Comparison**: Side-by-side view
- 📥 **Download Button**: Get your processed image
- 🔄 **Process Another**: Upload more images

## 🛠️ **Technical Details**

### **Backend Architecture**
- **Flask Framework**: Lightweight web server
- **Integration**: Uses your `dynamic_inference.py` script
- **Model Loading**: Loads DeblurGAN-v2 once at startup
- **File Handling**: Secure upload/download with cleanup
- **Error Handling**: Comprehensive error messages

### **Performance Optimization**
- **Single Model Instance**: Loaded once, reused for all requests
- **Memory Management**: Automatic cleanup of temporary files
- **Device Detection**: Auto-uses GPU if available, falls back to CPU
- **Image Optimization**: Efficient base64 encoding for web display

### **Security Features**
- **File Validation**: Only allows image formats
- **Size Limits**: 16MB maximum file size
- **Secure Filenames**: Prevents directory traversal attacks
- **Temporary Storage**: Files deleted after processing
- **Input Sanitization**: Safe handling of user uploads

## 🎯 **Usage Examples**

### **Desktop Users**
1. Open http://localhost:5000
2. Drag photo from desktop to upload area
3. Wait for processing
4. Right-click to save deblurred result

### **Mobile Users**
1. Open browser on phone
2. Navigate to your computer's IP (e.g., http://192.168.1.100:5000)
3. Tap "Choose Image" to access camera/gallery
4. View results and download

### **Batch Processing Workflow**
1. Process one image
2. Click "Process Another Image"
3. Upload next image
4. Repeat as needed

## 🔧 **Configuration Options**

### **Change Server Settings**
Edit `app.py` to modify:
```python
# Server configuration
app.run(debug=False, host='0.0.0.0', port=8080)

# File size limit (currently 16MB)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

# Model device preference
predictor = DeblurPredictor(device='cpu')  # Force CPU
```

### **Custom Styling**
Modify the CSS in `templates/index.html` to change:
- Colors and gradients
- Upload area appearance
- Button styles
- Mobile responsiveness

## 📱 **Mobile Access**

### **Local Network Access**
To access from other devices on your network:

1. **Find Your IP Address**:
```bash
# Mac/Linux
ifconfig | grep "inet "

# Windows
ipconfig
```

2. **Update Flask Configuration**:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

3. **Access from Mobile**:
Navigate to: `http://YOUR_IP_ADDRESS:5000`

### **Mobile-Optimized Features**
- ✅ Touch-friendly upload area
- ✅ Responsive image display
- ✅ Mobile-optimized buttons
- ✅ Portrait/landscape support

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **"Model failed to load"**
```bash
# Check if model weights exist
ls -la ../fpn_inception.h5

# Verify dependencies
pip install torch torchvision opencv-python-headless
```

#### **"Upload failed"**
- Check file size (must be < 16MB)
- Ensure image format is supported
- Try a different image

#### **"Processing takes too long"**
- **CPU Processing**: 30-60 seconds is normal
- **Large Images**: Try resizing to 1024px max
- **Multiple Uploads**: Process one at a time

#### **"Can't access from mobile"**
```bash
# Make sure server is accessible
python app.py
# Look for: "Running on all addresses (0.0.0.0)"

# Check firewall settings
# Ensure port 5000 is open
```

#### **"Out of memory errors"**
```python
# Reduce batch size or image size
# Or force CPU mode:
predictor = DeblurPredictor(device='cpu')
```

## 🔧 **Advanced Customization**

### **Adding Authentication**
```python
from flask_login import login_required

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    # Your upload logic
```

### **Adding Database Logging**
```python
# Track usage statistics
import sqlite3

def log_processing(filename, processing_time, device):
    # Log to database
```

### **API Endpoints**
The web app also provides API endpoints:
- `GET /status` - Model status
- `POST /upload` - Process image (JSON response)
- `GET /download/<filename>` - Download result

## 🎉 **Ready to Deploy!**

Your web app is production-ready with:
- ✅ **Professional UI/UX**
- ✅ **Robust error handling**
- ✅ **Mobile responsiveness**
- ✅ **Security best practices**
- ✅ **Performance optimization**

### **Quick Test**
```bash
cd webapp
python app.py
```

Then visit: **http://localhost:5000** 🚀

### **Share with Others**
```bash
# Run on all network interfaces
python app.py
# Access from: http://YOUR_IP:5000
```

## 💡 **Next Steps**

1. **Test the web app** with various images
2. **Customize the styling** to match your preferences
3. **Deploy to cloud** (Heroku, AWS, etc.) for public access
4. **Add features** like batch processing or user accounts
5. **Monitor performance** and optimize as needed

Your DeblurGAN-v2 is now accessible to anyone with a web browser! 🌐✨