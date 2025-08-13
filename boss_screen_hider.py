#!/usr/bin/env python3
"""
Boss Screen Hider - Background Monitoring with Screen Overlay
Overlays working_screen.png on top of current screen when boss is detected.
Runs completely in background without requiring window focus.
"""

import cv2
import numpy as np
import os
import pickle
import json
import time
import threading
from datetime import datetime
import sys
import tkinter as tk
from PIL import Image, ImageTk

class BossScreenHider:
    def __init__(self, model_path="model/boss_detector.pkl", config_path="model/config.json", screen_path="working_screen.png"):
        self.model_path = model_path
        self.config_path = config_path
        self.screen_path = screen_path
        
        # Model components
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.model_loaded = False
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Configuration
        self.detection_threshold = 0.75
        self.camera_width = 640
        self.camera_height = 480
        self.fps = 10
        
        # Background monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.screen_hidden = False
        self.last_detection_time = 0
        
        print("🎯 Boss Screen Hider initialized - Background Mode")
    
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.detection_threshold = config.get('detection_threshold', 0.75)
                    self.camera_width = config.get('camera_width', 640)
                    self.camera_height = config.get('camera_height', 480)
                    self.fps = config.get('fps', 10)
                print("✓ Configuration loaded")
            except Exception as e:
                print(f"⚠ Config load error: {e}, using defaults")
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("Please train the model first.")
            return False
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.model_loaded = True
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_lbp_features(self, image):
        def get_lbp(image, radius=1, n_points=8):
            h, w = image.shape
            lbp_image = np.zeros_like(image)
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = image[i, j]
                    pattern = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if image[x, y] >= center:
                            pattern |= (1 << k)
                    lbp_image[i, j] = pattern
            return lbp_image
        image_resized = cv2.resize(image, (128, 128))
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_resized
        equalized = cv2.equalizeHist(gray)
        lbp = get_lbp(equalized)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist_lbp = hist_lbp.astype(np.float32)
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        features = list(hist_lbp)
        grad_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        h, w = magnitude.shape
        grid_size = 8
        for i in range(0, h, h//grid_size):
            for j in range(0, w, w//grid_size):
                roi = magnitude[i:i+h//grid_size, j:j+w//grid_size]
                features.extend([np.mean(roi), np.std(roi), np.max(roi)])
        features.extend([
            np.mean(equalized), np.std(equalized), np.median(equalized),
            np.min(equalized), np.max(equalized),
            np.percentile(equalized, 25), np.percentile(equalized, 75)
        ])
        offsets = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for offset in offsets:
            cooc = self._calculate_glcm(equalized, offset)
            features.append(np.sum(cooc * cooc))
            features.append(np.sum(cooc**2))
        return np.array(features, dtype=np.float32)
    
    def _calculate_glcm(self, image, offset, levels=16):
        image_reduced = (image // (256 // levels)).astype(np.uint8)
        h, w = image_reduced.shape
        glcm = np.zeros((levels, levels), dtype=np.float32)
        dy, dx = offset
        for i in range(h - abs(dy)):
            for j in range(w - abs(dx)):
                i2, j2 = i + dy, j + dx
                if 0 <= i2 < h and 0 <= j2 < w:
                    glcm[image_reduced[i, j], image_reduced[i2, j2]] += 1
        return glcm / (glcm.sum() + 1e-7)
    
    def classify_face(self, face_roi):
        if not self.model_loaded:
            return "no_model", 0.0
        try:
            features = self.extract_lbp_features(face_roi)
            features_scaled = self.scaler.transform([features])
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            predicted_class = self.classifier.predict(features_scaled)[0]
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = max(probabilities)
            return predicted_name, confidence
        except Exception as e:
            print(f"Classification error: {e}")
            return "error", 0.0
    
    def show_working_screen(self):
        """Display working screen as full screen overlay on top of everything"""
        if not os.path.exists(self.screen_path):
            print(f"❌ Working screen not found: {self.screen_path}")
            return
        
        print("🖥️ Creating screen overlay...")
        
        # Create tkinter overlay window
        overlay_root = tk.Tk()
        overlay_root.title("Working Screen")
        
        # Make window fullscreen and always on top
        overlay_root.attributes('-fullscreen', True)
        overlay_root.attributes('-topmost', True)
        overlay_root.attributes('-alpha', 1.0)  # Fully opaque
        overlay_root.configure(bg='black')
        
        # Get screen dimensions
        screen_width = overlay_root.winfo_screenwidth()
        screen_height = overlay_root.winfo_screenheight()
        
        try:
            # Load and resize image to fit screen
            if self.screen_path.lower().endswith('.png'):
                pil_image = Image.open(self.screen_path)
            else:
                # Convert other formats using OpenCV first
                cv_image = cv2.imread(self.screen_path)
                cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image_rgb)
            
            # Resize to screen size
            pil_image = pil_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
            
            # Convert to tkinter format
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Create label to display image
            image_label = tk.Label(overlay_root, image=tk_image, bg='black')
            image_label.pack(fill=tk.BOTH, expand=True)
            
            # Keep reference to prevent garbage collection
            image_label.image = tk_image
            
            print("✅ Screen overlay active - boss screen displayed")
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            # Fallback to solid color overlay
            fallback_label = tk.Label(overlay_root, text="WORKING...", 
                                    font=("Arial", 48), fg="white", bg="black")
            fallback_label.pack(expand=True)
        
        # Function to close overlay
        def close_overlay():
            overlay_root.destroy()
            print("📱 Screen overlay closed")
        
        # Function to check if boss is still detected
        def check_boss_status():
            if not self.screen_hidden:
                close_overlay()
                return
            
            # Auto-close after 5 seconds of no boss detection
            if time.time() - self.last_detection_time > 5.0:
                print("� Boss gone, removing overlay...")
                self.screen_hidden = False
                close_overlay()
                return
            
            # Check again in 1 second
            overlay_root.after(1000, check_boss_status)
        
        # Bind escape key to close (emergency exit)
        overlay_root.bind('<Escape>', lambda e: close_overlay())
        
        # Set window to stay hidden
        self.screen_hidden = True
        
        # Start checking boss status
        overlay_root.after(1000, check_boss_status)
        
        # Run the overlay (this blocks until window is closed)
        try:
            overlay_root.mainloop()
        except:
            pass
        
        self.screen_hidden = False
    
    def background_monitor(self):
        """Background monitoring thread - runs independently"""
        print("🎥 Starting background camera monitoring...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return
        
        consecutive_detections = 0
        required_detections = 3  # Need 3 consecutive detections to trigger
        
        try:
            while self.monitoring:
                if self.screen_hidden:
                    # While screen is hidden, check if boss is still there
                    time.sleep(0.5)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(60, 60), 
                    maxSize=(300, 300)
                )
                
                boss_detected_this_frame = False
                
                # Check each face
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    label, confidence = self.classify_face(face_roi)
                    
                    if label == "boss" and confidence > self.detection_threshold:
                        boss_detected_this_frame = True
                        self.last_detection_time = time.time()
                        break
                
                # Count consecutive detections for stability
                if boss_detected_this_frame:
                    consecutive_detections += 1
                    print(f"� Boss detected ({consecutive_detections}/{required_detections})")
                else:
                    consecutive_detections = 0
                
                # Trigger screen hide after consecutive detections
                if consecutive_detections >= required_detections and not self.screen_hidden:
                    print(f"🚨 BOSS APPROACHING! Activating screen overlay at {datetime.now().strftime('%H:%M:%S')}")
                    # Start screen overlay in separate thread so monitoring continues
                    overlay_thread = threading.Thread(target=self.show_working_screen, daemon=False)
                    overlay_thread.start()
                    consecutive_detections = 0
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        
        finally:
            cap.release()
            print("📹 Camera monitoring stopped.")
    
    def start_monitoring(self):
        """Start background monitoring"""
        self.load_config()
        
        if not self.load_model():
            print("❌ Cannot start without trained model")
            return False
        
        print("\n🎯 Starting BossScreenHider - Background Mode")
        print(f"📊 Detection threshold: {self.detection_threshold}")
        print(f"📸 Camera resolution: {self.camera_width}x{self.camera_height}")
        print(f"🖼️ Working screen: {self.screen_path}")
        print("=" * 50)
        print("🔥 BOSS SENSOR ACTIVE - MONITORING IN BACKGROUND")
        print("=" * 50)
        print("💡 Tips:")
        print("   • The program runs in background")
        print("   • No preview window needed")
        print("   • Works while you're using other apps")
        print("   • Press Ctrl+C to stop")
        print()
        
        # Start background monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.background_monitor, daemon=True)
        self.monitor_thread.start()
        
        # Main thread handles user input
        try:
            print("🎮 Boss Sensor is now monitoring...")
            print("Press Ctrl+C to stop monitoring.")
            
            while self.monitoring:
                # Keep main thread alive and show status updates
                time.sleep(5)  # Update every 5 seconds
                status = "HIDDEN" if self.screen_hidden else "MONITORING"
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Status: {status}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping Boss Sensor...")
            self.monitoring = False
            
            # Wait for monitoring thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
            
            # Close any open windows
            cv2.destroyAllWindows()
            print("✅ Boss Sensor stopped successfully!")
        
        return True

def main():
    print("=== 🎯 BOSS SCREEN HIDER - Background Mode ===")
    print("🚀 Advanced Boss Detection System")
    print("📱 Runs in background - no window focus required!")
    print()
    
    hider = BossScreenHider()
    
    if not hider.start_monitoring():
        print("❌ Failed to start boss monitoring system")
        print("\nMake sure you have:")
        print("1. Trained model: python train_model.py")
        print("2. Working screen image: working_screen.png")
        print("3. Camera access available")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
