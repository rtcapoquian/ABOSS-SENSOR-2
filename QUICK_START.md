# BossSensor - Quick Start Guide 🚀

## Step 1: Setup Environment

Run the setup script to install dependencies:

```bash
python setup_simple.py
```

## Step 2: Capture Training Data

### Capture Boss Faces

```bash
python capture_faces.py boss 30
```

- Position your boss clearly in the camera
- Press SPACE to capture each image
- Capture 25-50 images for best results
- Try different angles and lighting

### Capture Other People's Faces

```bash
python capture_faces.py others 25
```

- Capture colleagues, friends, family members
- This helps the model distinguish between boss and others
- More diverse data = better accuracy

## Step 3: Train the Model

```bash
python train_model.py
```

- This creates a high-accuracy machine learning model
- The script will show training progress and accuracy
- Model is saved to `model/boss_detector.pkl`

## Step 4: Test the System

```bash
python boss_detector.py --test
```

- Tests face detection and classification
- Shows confidence scores in real-time
- Make sure boss detection works correctly

## Step 5: Start Monitoring

```bash
python boss_detector.py --monitor
```

- Starts real-time boss monitoring
- Automatically switches to working_screen.jpg when boss detected
- Press 'q' to quit, 'h' to manually hide screen

## Files Structure

```
ABOSS-SENSOR-2/
├── capture_faces.py      # Script to capture training faces
├── train_model.py        # Script to train the ML model
├── boss_detector.py      # Main detection and monitoring script
├── setup_simple.py       # Quick setup script
├── working_screen.jpg    # Image shown when boss is detected
├── faces/
│   ├── boss/            # Boss training images
│   └── others/          # Other people training images
└── model/
    ├── boss_detector.pkl # Trained model
    └── config.json       # Configuration settings
```

## Tips for Best Results

### Training Data Quality

- ✅ Good lighting conditions
- ✅ Clear face visibility
- ✅ Multiple angles and expressions
- ✅ Similar lighting to where you'll use the system
- ❌ Avoid blurry or dark images

### Detection Accuracy

- Capture 30+ boss images and 25+ other people images
- Include images with/without glasses if applicable
- Test in the same lighting conditions you'll use it
- Adjust detection threshold in `model/config.json` if needed

### System Performance

- Use good lighting for real-time detection
- Position camera at eye level
- Keep consistent distance from camera
- Close other applications for better performance

## Troubleshooting

### "No module named 'cv2'"

```bash
pip install opencv-python
```

### "Camera not found"

- Check if camera is connected
- Close other applications using the camera
- Try different camera index: modify `cv2.VideoCapture(1)` in scripts

### "Low accuracy"

- Capture more training images
- Ensure good lighting during capture
- Add more diverse "others" category images
- Retrain the model

### "Boss not detected"

- Check detection confidence scores in test mode
- Lower the detection_threshold in config.json
- Capture more boss training images in current lighting

### "False positives"

- Increase detection_threshold in config.json
- Add more "others" training images
- Increase consecutive_detections in config.json

## Configuration (model/config.json)

```json
{
  "detection_threshold": 0.75, // Higher = more strict detection
  "consecutive_detections": 3, // Detections needed before hiding
  "camera_width": 640, // Camera resolution
  "camera_height": 480,
  "fps": 10, // Frames per second
  "cooldown_time": 5.0 // Seconds before allowing new detection
}
```

## Advanced Usage

### Multiple Boss Detection

Capture images for multiple bosses in the same folder:

```bash
python capture_faces.py boss 50  # Include all bosses
```

### Custom Working Screen

Replace `working_screen.jpg` with your preferred image:

- Use high resolution (1920x1080+)
- Professional appearance (code, spreadsheet, etc.)
- Make sure it looks convincing!

### Running as Background Service

For continuous monitoring, you can set up the script to run automatically on startup (system-specific setup required).

---

**Remember**: This is for educational/entertainment purposes. Use responsibly and respect privacy laws! 👨‍💼👀
