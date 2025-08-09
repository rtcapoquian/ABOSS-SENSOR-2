#!/usr/bin/env python3
"""
Real-time Face Labeling Script
Shows live camera feed with face classification (boss vs others)
"""

import cv2
import numpy as np
import os
import pickle
import json
import time
from datetime import datetime

class RealTimeFaceLabeler:
    def __init__(self, model_path="model/boss_detector.pkl", config_path="model/config.json"):
        self.model_path = model_path
        self.config_path = config_path
        
        # Model components
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.model_loaded = False
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Configuration
        self.detection_threshold = 0.7
        self.camera_width = 640
        self.camera_height = 480
        self.fps = 15
        
        # Statistics
        self.frame_count = 0
        self.boss_detections = 0
        self.other_detections = 0
        
        print("Real-time Face Labeler initialized")
    
    def load_config(self):
        """Load configuration if available"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.detection_threshold = config.get('detection_threshold', 0.7)
                    self.camera_width = config.get('camera_width', 640)
                    self.camera_height = config.get('camera_height', 480)
                    self.fps = config.get('fps', 15)
                print("✓ Configuration loaded")
            except Exception as e:
                print(f"⚠ Config load error: {e}, using defaults")
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("Please train the model first:")
            print("1. python capture_faces.py others 25")
            print("2. python train_model.py")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            
            print("✓ Model loaded successfully")
            print(f"  Classes: {list(self.label_encoder.classes_)}")
            print(f"  Training samples: {model_data.get('training_samples', 'Unknown')}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_lbp_features(self, image):
        """Extract features from face image (same as training)"""
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
        
        # Resize and preprocess
        image_resized = cv2.resize(image, (128, 128))
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_resized
            
        equalized = cv2.equalizeHist(gray)
        
        # LBP features
        lbp = get_lbp(equalized)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist_lbp = hist_lbp.astype(np.float32)
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        
        features = list(hist_lbp)
        
        # Gradient features
        grad_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        h, w = magnitude.shape
        grid_size = 8
        for i in range(0, h, h//grid_size):
            for j in range(0, w, w//grid_size):
                roi = magnitude[i:i+h//grid_size, j:j+w//grid_size]
                features.extend([np.mean(roi), np.std(roi), np.max(roi)])
        
        # Statistical features
        features.extend([
            np.mean(equalized), np.std(equalized), np.median(equalized),
            np.min(equalized), np.max(equalized),
            np.percentile(equalized, 25), np.percentile(equalized, 75)
        ])
        
        # Texture features
        offsets = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for offset in offsets:
            cooc = self._calculate_glcm(equalized, offset)
            features.append(np.sum(cooc * cooc))
            features.append(np.sum(cooc**2))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_glcm(self, image, offset, levels=16):
        """Calculate GLCM"""
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
        """Classify a face region"""
        if not self.model_loaded:
            return "no_model", 0.0
        
        try:
            # Extract features
            features = self.extract_lbp_features(face_roi)
            features_scaled = self.scaler.transform([features])
            
            # Predict
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            predicted_class = self.classifier.predict(features_scaled)[0]
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = max(probabilities)
            
            return predicted_name, confidence
            
        except Exception as e:
            print(f"Classification error: {e}")
            return "error", 0.0
    
    def draw_face_info(self, frame, x, y, w, h, label, confidence):
        """Draw face detection box and label"""
        # Choose color based on prediction
        if label == "boss":
            if confidence > self.detection_threshold:
                color = (0, 0, 255)  # Red for boss (high confidence)
                text_color = (255, 255, 255)
                bg_color = (0, 0, 255)
            else:
                color = (0, 100, 255)  # Orange for boss (low confidence)
                text_color = (255, 255, 255) 
                bg_color = (0, 100, 255)
        elif label == "others":
            if confidence > self.detection_threshold:
                color = (0, 255, 0)  # Green for others (high confidence)
                text_color = (0, 0, 0)
                bg_color = (0, 255, 0)
            else:
                color = (100, 255, 100)  # Light green for others (low confidence)
                text_color = (0, 0, 0)
                bg_color = (100, 255, 100)
        else:
            color = (128, 128, 128)  # Gray for error/no model
            text_color = (255, 255, 255)
            bg_color = (128, 128, 128)
        
        # Draw face rectangle
        thickness = 3 if confidence > self.detection_threshold else 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Prepare label text
        if label in ["boss", "others"]:
            label_text = f"{label.upper()}: {confidence:.3f}"
            status = "HIGH CONF" if confidence > self.detection_threshold else "LOW CONF"
        else:
            label_text = label.upper()
            status = ""
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness_text = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness_text)
        (status_w, status_h), _ = cv2.getTextSize(status, font, 0.4, 1)
        
        # Draw background rectangles for text
        cv2.rectangle(frame, (x, y-text_h-10), (x+max(text_w, status_w)+10, y), bg_color, -1)
        
        # Draw main label
        cv2.putText(frame, label_text, (x+5, y-text_h-2), font, font_scale, text_color, thickness_text)
        
        # Draw status if available
        if status:
            cv2.putText(frame, status, (x+5, y-2), font, 0.4, text_color, 1)
        
        return label == "boss" and confidence > self.detection_threshold
    
    def start_labeling(self):
        """Start real-time face labeling"""
        # Load configuration and model
        self.load_config()
        
        if not self.load_model():
            print("❌ Cannot start without trained model")
            return False
        
        print("\n🎯 Starting real-time face labeling...")
        print(f"Detection threshold: {self.detection_threshold}")
        print("Press 'q' to quit, 't' to adjust threshold, 's' for statistics")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return False
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self.frame_count += 1
                fps_counter += 1
                
                # Calculate FPS every second
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, 
                    minSize=(60, 60), maxSize=(300, 300)
                )
                
                boss_detected_this_frame = False
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Classify face
                    label, confidence = self.classify_face(face_roi)
                    
                    # Draw face info
                    is_boss = self.draw_face_info(frame, x, y, w, h, label, confidence)
                    
                    # Update statistics
                    if is_boss:
                        boss_detected_this_frame = True
                        self.boss_detections += 1
                    elif label == "others" and confidence > self.detection_threshold:
                        self.other_detections += 1
                
                # Draw header info
                header_height = 100
                cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (0, 0, 0), -1)
                
                # Title
                cv2.putText(frame, "REAL-TIME FACE LABELING", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Statistics
                cv2.putText(frame, f"FPS: {current_fps} | Faces: {len(faces)} | Threshold: {self.detection_threshold:.2f}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(frame, f"Boss detections: {self.boss_detections} | Others: {self.other_detections}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Controls
                cv2.putText(frame, "Q:Quit | T:Threshold | S:Stats", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Alert if boss detected
                if boss_detected_this_frame:
                    cv2.putText(frame, "🚨 BOSS DETECTED! 🚨", 
                               (frame.shape[1]//2 - 100, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('Real-time Face Labeling', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    # Adjust threshold
                    print(f"\nCurrent threshold: {self.detection_threshold:.2f}")
                    try:
                        new_threshold = float(input("Enter new threshold (0.1-0.9): "))
                        if 0.1 <= new_threshold <= 0.9:
                            self.detection_threshold = new_threshold
                            print(f"Threshold updated to: {self.detection_threshold:.2f}")
                        else:
                            print("Invalid threshold. Must be between 0.1 and 0.9")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                elif key == ord('s'):
                    # Show statistics
                    print(f"\n=== Statistics ===")
                    print(f"Frames processed: {self.frame_count}")
                    print(f"Boss detections: {self.boss_detections}")
                    print(f"Others detections: {self.other_detections}")
                    print(f"Current FPS: {current_fps}")
                    print(f"Detection threshold: {self.detection_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n✓ Session completed!")
            print(f"Final statistics:")
            print(f"- Frames processed: {self.frame_count}")
            print(f"- Boss detections: {self.boss_detections}")
            print(f"- Others detections: {self.other_detections}")
        
        return True

def main():
    print("=== Real-time Face Labeling System ===\n")
    
    labeler = RealTimeFaceLabeler()
    
    if not labeler.start_labeling():
        print("❌ Failed to start labeling system")
        print("\nMake sure you have:")
        print("1. Captured boss training data: python capture_faces.py boss 30")
        print("2. Captured others training data: python capture_faces.py others 25")
        print("3. Trained the model: python train_model.py")

if __name__ == "__main__":
    main()
