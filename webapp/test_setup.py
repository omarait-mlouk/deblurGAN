#!/usr/bin/env python3
"""
Test script to verify the research web application setup
"""

import os
import sys
import importlib.util

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {file_path}")
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - Available")
        return True
    except ImportError:
        print(f"❌ {module_name} - NOT AVAILABLE")
        return False

def main():
    print("🔬 DeblurGAN-v2 Research Web App Verification")
    print("=" * 50)
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    webapp_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(webapp_dir)
    
    dirs_ok = True
    dirs_ok &= check_file_exists(webapp_dir, "Webapp directory")
    dirs_ok &= check_file_exists(os.path.join(webapp_dir, "templates"), "Templates directory")
    dirs_ok &= check_file_exists(os.path.join(webapp_dir, "templates", "research_interface.html"), "HTML template")
    dirs_ok &= check_file_exists(os.path.join(parent_dir, "fpn_inception.h5"), "Model weights")
    dirs_ok &= check_file_exists(os.path.join(parent_dir, "config", "config.yaml"), "Config file")
    dirs_ok &= check_file_exists(os.path.join(parent_dir, "dynamic_inference.py"), "Dynamic inference")
    
    # Check Python dependencies
    print("\n📦 Python Dependencies:")
    deps_ok = True
    deps_ok &= check_import('flask')
    deps_ok &= check_import('cv2')
    deps_ok &= check_import('numpy')
    deps_ok &= check_import('torch')
    deps_ok &= check_import('yaml')
    
    print("\n🔧 Optional Dependencies:")
    check_import('psutil')
    check_import('PIL')
    
    # Test model loading
    print("\n🧠 Model Loading Test:")
    try:
        sys.path.insert(0, parent_dir)
        from dynamic_inference import DeblurPredictor
        
        # Change to parent directory for model loading
        original_cwd = os.getcwd()
        os.chdir(parent_dir)
        
        predictor = DeblurPredictor(weights_path='fpn_inception.h5', device='auto')
        os.chdir(original_cwd)
        print(f"✅ Model loaded successfully on device: {predictor.device}")
        model_ok = True
    except Exception as e:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        print(f"❌ Model loading failed: {e}")
        model_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY:")
    print(f"📁 Directory Structure: {'✅ OK' if dirs_ok else '❌ ISSUES'}")
    print(f"📦 Core Dependencies: {'✅ OK' if deps_ok else '❌ MISSING'}")
    print(f"🧠 Model Loading: {'✅ OK' if model_ok else '❌ FAILED'}")
    
    if dirs_ok and deps_ok and model_ok:
        print("\n🎉 ALL CHECKS PASSED! Ready to run:")
        print("   cd webapp")
        print("   python app_research_fixed.py")
        return True
    else:
        print("\n⚠️  ISSUES FOUND:")
        if not deps_ok:
            print("   Install missing dependencies: pip install -r research_requirements.txt")
        if not dirs_ok:
            print("   Check file paths and directory structure")
        if not model_ok:
            print("   Verify model weights and config files")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
