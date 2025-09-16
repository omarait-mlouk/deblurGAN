#!/bin/bash
# DeblurGAN-v2 Web App Startup Script

echo "🚀 Starting DeblurGAN-v2 Web Application"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "../fpn_inception.h5" ]; then
    echo "❌ Error: Model weights not found!"
    echo "Please run this script from the webapp directory:"
    echo "cd webapp && ./start_webapp.sh"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "Consider activating your environment first:"
    echo "source /path/to/your/venv/bin/activate"
    echo ""
fi

# Install requirements if needed
echo "📦 Checking Flask requirements..."
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask requirements..."
    pip install -r webapp_requirements.txt
fi

echo ""
echo "🌐 Starting web server..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python app.py
