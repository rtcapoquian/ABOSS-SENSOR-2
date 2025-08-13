#!/usr/bin/env python3
"""
Boss Screen Hider - INSTANT Background Monitoring with Screen Overlay
Overlays working_screen.png on top of current screen when boss is detected.
INSTANT RESPONSE - No counting delays!
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
        self.fps = 15
        
        # Background monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.screen_hidden = False
        self.last_detection_time = 0
        self.last_trigger_time = 0
        
        print("🎯 Boss Screen Hider initialized - INSTANT MODE")
    
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.detection_threshold = config.get('detection_threshold', 0.75)
                    self.camera_width = config.get('camera_width', 640)
                    self.camera_height = config.get('camera_height', 480)
                    self.fps = config.get('fps', 15)
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
        
        print("🖥️ Creating INSTANT screen overlay...")
        
        # Create tkinter overlay window
        overlay_root = tk.Tk()
        overlay_root.title("Working")
        
        # Get screen dimensions first
        screen_width = overlay_root.winfo_screenwidth()
        screen_height = overlay_root.winfo_screenheight()
        
        # Configure window for true overlay
        overlay_root.geometry(f"{screen_width}x{screen_height}+0+0")
        overlay_root.attributes('-fullscreen', True)
        overlay_root.attributes('-topmost', True)
        overlay_root.attributes('-alpha', 1.0)
        overlay_root.configure(bg='black')
        overlay_root.overrideredirect(True)  # Remove window decorations
        overlay_root.focus_force()  # Force focus to ensure it's on top
        
        try:
            # Load and resize image to fit screen
            if self.screen_path.lower().endswith('.png'):
                pil_image = Image.open(self.screen_path)
            else:
                # Convert other formats using OpenCV first
                cv_image = cv2.imread(self.screen_path)
                if cv_image is not None:
                    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cv_image_rgb)
                else:
                    raise Exception("Could not load image file")
            
            # Resize to exact screen size
            pil_image = pil_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
            
            # Convert to tkinter format
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Create label to display image (fill entire screen)
            image_label = tk.Label(overlay_root, image=tk_image, bd=0, highlightthickness=0)
            image_label.place(x=0, y=0, width=screen_width, height=screen_height)
            
            # Keep reference to prevent garbage collection
            image_label.image = tk_image
            
            print("✅ Screen overlay ACTIVE - working screen displayed")
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            # Fallback to solid color overlay with text
            fallback_label = tk.Label(overlay_root, text="📊 WORKING...\n\nAnalyzing Data\n━━━━━━━━━━", 
                                    font=("Courier New", 32, "bold"), fg="white", bg="#1e1e1e",
                                    bd=0, highlightthickness=0)
            fallback_label.place(x=0, y=0, width=screen_width, height=screen_height)
        
        # Variables to track overlay state
        overlay_active = True
        
        # Function to close overlay ONLY on keypress
        def close_overlay(event=None):
            nonlocal overlay_active
            if overlay_active:
                overlay_active = False
                print("\n� Keypress detected - Closing overlay...")
                print("✅ Program will exit now.")
                try:
                    overlay_root.quit()
                    overlay_root.destroy()
                except:
                    pass
        
        # Make window focusable and bind ALL possible key events
        overlay_root.configure(highlightthickness=0)
        overlay_root.focus_set()
        overlay_root.grab_set()  # Grab all events
        
        # Bind EVERY possible key event to close
        overlay_root.bind('<Key>', close_overlay)           # Any key
        overlay_root.bind('<KeyPress>', close_overlay)      # Key press
        overlay_root.bind('<Button-1>', close_overlay)      # Left mouse click
        overlay_root.bind('<Button-2>', close_overlay)      # Middle mouse click  
        overlay_root.bind('<Button-3>', close_overlay)      # Right mouse click
        overlay_root.bind('<Escape>', close_overlay)        # Escape key
        overlay_root.bind('<Return>', close_overlay)        # Enter key
        overlay_root.bind('<space>', close_overlay)         # Spacebar
        overlay_root.bind('<Control-c>', close_overlay)     # Ctrl+C
        
        # Make ALL child widgets also respond to keypresses
        def bind_to_all_children(widget):
            widget.bind('<Key>', close_overlay)
            widget.bind('<KeyPress>', close_overlay) 
            widget.bind('<Button-1>', close_overlay)
            for child in widget.winfo_children():
                bind_to_all_children(child)
        
        bind_to_all_children(overlay_root)
        
        # Set screen as hidden - NO AUTO CLOSE
        self.screen_hidden = True
        
        # Add instruction text overlay - VERY VISIBLE
      
        
        print("⌨️ Overlay displayed - WAITING FOR KEYPRESS ONLY")
        print("🔑 The overlay will ONLY close when you press a key!")
        
        # Run the overlay - BLOCKS until keypress ONLY
        try:
            # Force focus multiple times to ensure it works
            overlay_root.after(100, lambda: overlay_root.focus_force())
            overlay_root.after(200, lambda: overlay_root.focus_set())
            overlay_root.after(300, lambda: overlay_root.grab_set())
            
            overlay_root.mainloop()  # This blocks until close_overlay() is called
        except Exception as e:
            print(f"Overlay error: {e}")
        
        # Ensure screen_hidden is reset
        self.screen_hidden = False
        print("✅ Screen overlay ended - Program exiting")
    
    def background_monitor(self):
        """Background monitoring thread - INSTANT DETECTION"""
        print("🎥 Starting INSTANT background camera monitoring...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return

        trigger_cooldown = 0  # Prevent rapid re-triggering for 3 seconds only

        try:
            while self.monitoring:
                if self.screen_hidden:
                    # While screen is hidden, check if boss is still there
                    time.sleep(0.2)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue
                
                # Process frame
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces - optimized for speed
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=3,  # Less strict for faster detection
                    minSize=(40, 40), 
                    maxSize=(500, 500)
                )
                
                current_time = time.time()
                boss_detected_this_frame = False
                
                # Check each face - INSTANT response
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    label, confidence = self.classify_face(face_roi)
                    
                    if label == "boss" and confidence > self.detection_threshold:
                        boss_detected_this_frame = True
                        self.last_detection_time = current_time
                        break
                
                # INSTANT trigger if boss detected and cooldown passed
                if (boss_detected_this_frame and 
                    not self.screen_hidden and 
                    current_time - self.last_trigger_time > trigger_cooldown):
                    
                    print(f"🚨 BOSS DETECTED! SWITCHING TO WORK SCREEN at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                    self.last_trigger_time = current_time
                    
                    # Stop monitoring immediately after detection
                    self.monitoring = False
                    print("🛑 Stopping monitoring - Boss detected, mission complete!")
                    
                    # Show overlay and exit
                    self.show_working_screen()
                    break  # Exit the monitoring loop
                
                # Very fast monitoring for instant response
                time.sleep(0.03)  # ~30 FPS monitoring for maximum responsiveness
        
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
        
        print("\n🎯 Starting BossScreenHider - INSTANT MODE")
        print(f"📊 Detection threshold: {self.detection_threshold}")
        print(f"📸 Camera resolution: {self.camera_width}x{self.camera_height}")
        print(f"🖼️ Working screen: {self.screen_path}")
        print("=" * 60)
        print("🚨 INSTANT BOSS SENSOR ACTIVE - NO DELAYS!")
        print("⚡ IMMEDIATE RESPONSE - BOSS DETECTION INSTANT")
        print("=" * 60)
        print("💡 Features:")
        print("   • Instant detection - no counting delays")
        print("   • Runs in background - no window focus needed")
        print("   • Works while using other apps")
        print("   • 30 FPS monitoring for maximum speed")
        print("   • Press Ctrl+C to stop")
        print()
        
        # Start background monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.background_monitor, daemon=True)
        self.monitor_thread.start()
        
        # Main thread handles user input - USER CONTROLS WHEN TO EXIT
        try:
            print("⚡ REAL-TIME Boss Sensor is now monitoring...")
            print("🎯 Waiting for boss to appear...")
            print("👁️ Camera is watching continuously...")
            print("Press Ctrl+C to stop monitoring.")
            
            while self.monitoring:
                # Keep main thread alive and show status updates (every 20 minutes)
                time.sleep(1200)  # 20 minutes
                status = "🖥️ SCREEN HIDDEN" if self.screen_hidden else "👁️ MONITORING"
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
    print("=== 🚨 INSTANT BOSS SCREEN HIDER ===")
    print("⚡ Lightning-Fast Boss Detection System")
    print("🎯 NO DELAYS - INSTANT RESPONSE!")
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
