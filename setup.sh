#!/bin/bash

# BossSensor Setup Script
# Run this script to set up BossSensor on your system

echo "=== BossSensor Setup ==="

# Detect system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ $(uname -m) == "arm"* ]] || [[ $(uname -m) == "aarch64" ]]; then
        echo "Detected: Raspberry Pi"
        PLATFORM="pi"
    else
        echo "Detected: Linux"
        PLATFORM="linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected: macOS"
    PLATFORM="mac"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Detected: Windows"
    PLATFORM="windows"
else
    echo "Detected: Unknown system"
    PLATFORM="unknown"
fi

# Install system dependencies
echo "Installing system dependencies..."

if [[ $PLATFORM == "pi" ]]; then
    echo "Setting up for Raspberry Pi..."
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev python3-setuptools
    sudo apt-get install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
    sudo apt-get install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
    sudo apt-get install -y libatlas-base-dev
    sudo apt-get install -y libjasper-dev
    sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
    sudo apt-get install -y libxvidcore-dev libx264-dev
    
    echo "Installing Python packages for Pi..."
    pip3 install --upgrade pip
    pip3 install -r requirements_pi.txt
    
elif [[ $PLATFORM == "linux" ]]; then
    echo "Setting up for Linux..."
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    
elif [[ $PLATFORM == "mac" ]]; then
    echo "Setting up for macOS..."
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install python3
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    
elif [[ $PLATFORM == "windows" ]]; then
    echo "For Windows, please manually install:"
    echo "1. Python 3.8+ from python.org"
    echo "2. Run: pip install -r requirements.txt"
    echo "3. If you encounter issues, use: pip install opencv-python scikit-learn numpy pillow"
    
else
    echo "Please manually install Python 3.8+ and run:"
    echo "pip install -r requirements.txt"
fi

# Create directories
echo "Creating directories..."
mkdir -p training_data
mkdir -p logs

# Test installation
echo "Testing installation..."
python3 -c "import cv2, numpy, sklearn; print('✓ All packages installed successfully')" || echo "❌ Installation failed"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick Start:"
echo "1. Collect boss data:    python3 boss_sensor_lite.py collect boss 20"
echo "2. Collect other data:   python3 boss_sensor_lite.py collect colleague 15"
echo "3. Train model:          python3 boss_sensor_lite.py train"
echo "4. Start monitoring:     python3 boss_sensor_lite.py monitor"
echo ""
echo "For Raspberry Pi, use boss_sensor_lite.py (optimized version)"
echo "For desktop/laptop, you can use either version"
