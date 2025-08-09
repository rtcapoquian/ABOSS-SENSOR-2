# Step 1: Capture boss training data
python capture_faces.py boss 30

# Step 2: Capture other people data  
python capture_faces.py others 25

# Step 3: Train the model
python train_model.py

# Step 4: Start monitoring (THE MAGIC HAPPENS HERE!)
python boss_detector.py --monitor


# BossSensor 👁️

**Hide your screen when your boss is approaching!**

BossSensor is a lightweight computer vision system that uses your webcam to detect when your boss is nearby and automatically switches your screen to a "working" image. Perfect for maintaining productivity appearances! 😉

## Features

- 🎯 **Lightweight**: Optimized for Raspberry Pi 3B (1GB RAM)
- 🚀 **Fast Detection**: Real-time face recognition with low latency
- 🔧 **Easy Setup**: Simple CLI interface for training and monitoring
- 📸 **Custom Training**: Train on your specific boss and colleagues
- 🖥️ **Screen Hiding**: Automatically displays working screen when boss detected
- ⚙️ **Configurable**: Adjustable sensitivity and detection parameters

## System Requirements

### Minimum Requirements (Raspberry Pi 3B)

- Raspberry Pi 3B with 1GB RAM
- USB webcam or Pi Camera
- Python 3.7+
- 2GB free storage space

### Recommended (Desktop/Laptop)

- 4GB+ RAM
- Webcam
- Python 3.8+
- 4GB free storage space

## Quick Start

### 1. Installation

**Windows:**

```cmd
setup.bat
```

**Linux/macOS/Raspberry Pi:**

```bash
chmod +x setup.sh
./setup.sh
```

**Manual Installation:**

```bash
pip install opencv-python scikit-learn numpy pillow
```

### 2. Collect Training Data

First, collect images of your boss:

```bash
python boss_sensor_lite.py collect boss 20
```

Then collect images of colleagues (for comparison):

```bash
python boss_sensor_lite.py collect colleague 15
python boss_sensor_lite.py collect friend 10
```

**Tips for data collection:**

- Ensure good lighting
- Capture different angles and expressions
- Include images with and without glasses
- Take photos at different times of day

### 3. Train the Model

```bash
python boss_sensor_lite.py train
```

This creates a lightweight model optimized for your Raspberry Pi.

### 4. Start Monitoring

```bash
python boss_sensor_lite.py monitor
```

**Controls during monitoring:**

- Press `q` to quit
- Press `h` to manually hide screen
- ESC to restore hidden screen

## File Structure

```
ABOSS-SENSOR-2/
├── boss_sensor.py          # Full-featured version
├── boss_sensor_lite.py     # Raspberry Pi optimized version
├── working_screen.jpg      # Image displayed when boss is detected
├── requirements.txt        # Python dependencies (desktop)
├── requirements_pi.txt     # Python dependencies (Pi)
├── setup.sh               # Linux/macOS/Pi setup script
├── setup.bat              # Windows setup script
├── config.json            # Configuration file (auto-generated)
├── training_data/         # Training images directory
│   ├── boss/             # Boss training images
│   └── colleague/        # Other people's images
└── logs/                 # Application logs
```

## Configuration

The system auto-generates a `config.json` file with these settings:

```json
{
  "detection_threshold": 0.7,
  "required_consecutive": 2,
  "camera_width": 160,
  "camera_height": 120,
  "fps": 5,
  "boss_name": "boss",
  "face_size": 96
}
```

**Key Parameters:**

- `detection_threshold`: Confidence level needed for boss detection (0.0-1.0)
- `required_consecutive`: Number of consecutive detections before hiding screen
- `camera_width/height`: Camera resolution (lower = better Pi performance)
- `fps`: Frames per second (lower = better Pi performance)

## Raspberry Pi Deployment

### 1. Transfer Files

Copy the entire project to your Raspberry Pi:

```bash
scp -r ABOSS-SENSOR-2/ pi@your-pi-ip:~/
```

### 2. Setup on Pi

```bash
ssh pi@your-pi-ip
cd ABOSS-SENSOR-2
chmod +x setup.sh
./setup.sh
```

### 3. Enable Camera

```bash
sudo raspi-config
# Navigate to: Interfacing Options > Camera > Enable
sudo reboot
```

### 4. Test Camera

```bash
python3 boss_sensor_lite.py test
```

### 5. Start Monitoring

```bash
python3 boss_sensor_lite.py monitor
```

## Performance Optimization

### For Raspberry Pi 3B (1GB RAM):

**Use the lite version:**

- `boss_sensor_lite.py` is specifically optimized for Pi
- Uses simple feature extraction instead of deep learning
- Reduced image resolution (160x120)
- Lower frame rate (5 fps)
- Smaller face detection windows

**System Optimizations:**

```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 128

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-country

# Add to /boot/config.txt for better performance
gpu_mem=128
dtoverlay=vc4-fkms-v3d
```

**Running as Service (Optional):**
Create `/etc/systemd/system/bosssensor.service`:

```ini
[Unit]
Description=BossSensor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ABOSS-SENSOR-2
ExecStart=/usr/bin/python3 boss_sensor_lite.py monitor
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable bosssensor
sudo systemctl start bosssensor
```

## Troubleshooting

### Common Issues:

**1. Camera not detected:**

```bash
# Check camera
lsusb  # Should show your USB camera
# or for Pi camera:
vcgencmd get_camera
```

**2. Low performance on Pi:**

- Reduce camera resolution in config.json
- Lower fps setting
- Ensure adequate power supply (2.5A+)
- Use fast SD card (Class 10+)

**3. False positives:**

- Collect more diverse training data
- Increase detection_threshold
- Increase required_consecutive detections

**4. Model not detecting boss:**

- Collect more boss training data
- Ensure good lighting during training
- Lower detection_threshold (carefully)
- Retrain with more varied boss images

**5. Python package conflicts:**
Use virtual environment:

```bash
python3 -m venv bosssensor_env
source bosssensor_env/bin/activate  # Linux/macOS
# or
bosssensor_env\Scripts\activate     # Windows
pip install -r requirements_pi.txt
```

## Advanced Usage

### Multiple Boss Detection:

```bash
python boss_sensor_lite.py collect boss1 20
python boss_sensor_lite.py collect boss2 15
python boss_sensor_lite.py train
```

### Custom Working Screen:

Replace `working_screen.jpg` with your desired image. Recommended:

- High resolution (1920x1080 or higher)
- Professional appearance
- Code editor, spreadsheet, or document

### Logging:

Enable detailed logging by modifying the script:

```python
import logging
logging.basicConfig(level=logging.INFO,
                   filename='logs/bosssensor.log',
                   format='%(asctime)s - %(levelname)s - %(message)s')
```

## Security & Privacy

- All processing is done locally on your device
- No data is transmitted to external servers
- Training images are stored locally only
- You have full control over your data

## Legal Disclaimer

This software is for educational and entertainment purposes only. Please:

- Respect privacy laws in your jurisdiction
- Obtain consent before recording others
- Use responsibly in workplace environments
- Follow your organization's IT policies

## Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use and modify as needed.

## Support

If you encounter issues:

1. Check the troubleshooting section
2. Review the configuration settings
3. Test with different lighting conditions
4. Ensure adequate training data

---

**Remember**: The effectiveness of boss detection depends on quality training data and proper system setup. Happy monitoring! 👨‍💼👀
