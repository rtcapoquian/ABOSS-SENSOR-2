#!/usr/bin/env python3
"""
Boss Screen Hider
Displays working_screen.png when boss is detected in real-time.
"""

import cv2
import numpy as np
import os
import pickle
import json
import time
from datetime import datetime

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
        
        print("Boss Screen Hider initialized")
    
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
        if not os.path.exists(self.screen_path):
            print(f"❌ Working screen not found: {self.screen_path}")
            return
        img = cv2.imread(self.screen_path)
        if img is None:
            print(f"❌ Could not load image: {self.screen_path}")
            return
        cv2.namedWindow('WORKING SCREEN', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('WORKING SCREEN', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('WORKING SCREEN', img)
        print("🖥️ Displaying working screen. Press ESC to close.")
        while True:
            if cv2.waitKey(100) & 0xFF == 27:
                break
        cv2.destroyWindow('WORKING SCREEN')
    
    def start_monitoring(self):
        self.load_config()
        if not self.load_model():
            print("❌ Cannot start without trained model")
            return False
        print("\n🎯 Starting BossScreenHider monitoring...")
        print(f"Detection threshold: {self.detection_threshold}")
        print("Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return False
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), maxSize=(300, 300)
                )
                boss_detected = False
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    label, confidence = self.classify_face(face_roi)
                    if label == "boss" and confidence > self.detection_threshold:
                        boss_detected = True
                        break
                if boss_detected:
                    print(f"🚨 Boss detected! Hiding screen at {datetime.now().strftime('%H:%M:%S')}")
                    cap.release()
                    self.show_working_screen()
                    # After hiding, re-initialize camera
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    if not cap.isOpened():
                        print("❌ Cannot access camera after hiding!")
                        return False
                # Show small preview window
                preview = cv2.resize(frame, (320, 240))
                cv2.imshow('Boss Monitor Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        print("✓ Monitoring session ended.")
        return True

def main():
    print("=== Boss Screen Hider ===\n")
    hider = BossScreenHider()
    hider.start_monitoring()

if __name__ == "__main__":
    main()
