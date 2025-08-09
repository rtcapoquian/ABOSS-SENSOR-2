#!/usr/bin/env python3
"""
Quick setup script for BossSensor
"""

import subprocess
import sys
import os

def install_packages():
    """Install required Python packages"""
    packages = [
        "opencv-python",
        "scikit-learn", 
        "numpy",
        "pillow"
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def test_installation():
    """Test if all packages are installed correctly"""
    print("\nTesting installation...")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn not available")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow available")
    except ImportError:
        print("❌ Pillow not available")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["faces/boss", "faces/others", "model", "logs"]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def main():
    print("=== BossSensor Setup ===\n")
    
    # Install packages
    if not install_packages():
        print("❌ Package installation failed!")
        return
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed!")
        return
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Capture boss faces:    python capture_faces.py boss 30")
    print("2. Capture other faces:   python capture_faces.py others 20")  
    print("3. Train the model:       python train_model.py")
    print("4. Test detection:        python boss_detector.py --test")
    print("5. Start monitoring:      python boss_detector.py --monitor")
    print("\nMake sure you have a working_screen.jpg file in the directory!")

if __name__ == "__main__":
    main()
