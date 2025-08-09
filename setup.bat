@echo off
echo === BossSensor Setup for Windows ===
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add to PATH" during installation
    pause
    exit /b 1
)

echo Python found. Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages...
python -m pip install opencv-python==4.7.1.72
python -m pip install scikit-learn==1.2.2
python -m pip install numpy==1.24.3
python -m pip install pillow==9.5.0

echo Creating directories...
if not exist "training_data" mkdir training_data
if not exist "logs" mkdir logs

echo Testing installation...
python -c "import cv2, numpy, sklearn; print('✓ All packages installed successfully')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Installation test failed
    echo Trying alternative installation...
    python -m pip install opencv-python scikit-learn numpy pillow
    python -c "import cv2, numpy, sklearn; print('✓ Installation successful')" 2>nul
)

echo.
echo === Setup Complete ===
echo.
echo Quick Start:
echo 1. Collect boss data:    python boss_sensor_lite.py collect boss 20
echo 2. Collect other data:   python boss_sensor_lite.py collect colleague 15  
echo 3. Train model:          python boss_sensor_lite.py train
echo 4. Start monitoring:     python boss_sensor_lite.py monitor
echo.
echo Use boss_sensor_lite.py (optimized for lower-end hardware)
pause
