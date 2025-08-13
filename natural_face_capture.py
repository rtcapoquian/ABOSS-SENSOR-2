#!/usr/bin/env python3
"""
Natural Face Capture - No Enhancement
Captures faces with completely natural colors and lighting
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse

class NaturalFaceCapture:
    def __init__(self):
        # Initialize face detector
        # Try multiple paths for the Haar cascade file
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else None,
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]
        
        self.face_cascade = None
        for path in cascade_paths:
            if path and os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                break
                
        if self.face_cascade is None or self.face_cascade.empty():
            # Download the cascade file if not found
            import urllib.request
            cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
            cascade_file = 'haarcascade_frontalface_default.xml'
            print(f"📥 Downloading Haar cascade file...")
            urllib.request.urlretrieve(cascade_url, cascade_file)
            self.face_cascade = cv2.CascadeClassifier(cascade_file)
        
        # Create directories
        self.faces_dir = "faces"
        self.boss_dir = os.path.join(self.faces_dir, "boss")
        self.others_dir = os.path.join(self.faces_dir, "others")
        
        os.makedirs(self.boss_dir, exist_ok=True)
        os.makedirs(self.others_dir, exist_ok=True)
        
        # Count existing images
        self.boss_count = len([f for f in os.listdir(self.boss_dir) if f.endswith('.jpg')])
        self.others_count = len([f for f in os.listdir(self.others_dir) if f.endswith('.jpg')])
        
        # Settings
        self.crop_padding = 15
        self.target_size = (128, 128)
        
        print("🌟 Natural Face Capture System - No Enhancement")
        print(f"📊 Current counts - Boss: {self.boss_count}, Others: {self.others_count}")
    
    def detect_faces(self, frame):
        """Simple face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50), 
            maxSize=(300, 300)
        )
        
        return faces
    
    def crop_face_natural(self, frame, face_rect):
        """Crop face with NO enhancement - completely natural"""
        x, y, w, h = face_rect
        
        # Add minimal padding
        x1 = max(0, x - self.crop_padding)
        y1 = max(0, y - self.crop_padding)
        x2 = min(frame.shape[1], x + w + self.crop_padding)
        y2 = min(frame.shape[0], y + h + self.crop_padding)
        
        # Extract face region
        face_crop = frame[y1:y2, x1:x2]
        
        # Resize to standard size - NO OTHER PROCESSING
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_CUBIC)
            return face_resized
        
        return None
    
    def save_natural_face(self, face_crop, category):
        """Save face crop with NO enhancement - pure natural image"""
        if face_crop is None:
            return None
        
        # NO ENHANCEMENT - Save exactly as captured
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if category.lower() == "boss":
            self.boss_count += 1
            filename = f"boss_{self.boss_count:04d}_{timestamp}_natural_crop.jpg"
            filepath = os.path.join(self.boss_dir, filename)
        else:
            self.others_count += 1
            filename = f"others_{self.others_count:04d}_{timestamp}_natural_crop.jpg"
            filepath = os.path.join(self.others_dir, filename)
        
        # Save with high quality, no compression artifacts
        success = cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            return filepath
        return None
    
    def capture_natural(self, category="boss", target_count=25):
        """Natural capture session with no enhancement"""
        print(f"\n🌟 NATURAL FACE CAPTURE STARTED")
        print(f"📋 Mode: {category.upper()} images")
        print(f"🎯 Target: {target_count} natural face images")
        print("✨ No enhancement applied - completely natural colors!")
        print("\n🎮 Controls:")
        print("   • Press SPACE to capture when face is detected")
        print("   • Press 'Q' to quit")
        print("\n🚀 Starting camera...")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access camera!")
            return False
        
        print("✅ Camera ready! You'll see your natural colors...")
        
        captured_this_session = 0
        last_capture_time = 0
        capture_cooldown = 1.0
        
        try:
            while captured_this_session < target_count:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Store clean frame for cropping
                clean_frame = frame.copy()
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Draw minimal interface
                display_frame = frame.copy()
                self.draw_minimal_interface(display_frame, faces, category, captured_this_session, target_count)
                
                # Show frame
                cv2.imshow('Natural Face Capture', display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()
                
                if key == ord('q') or key == ord('Q'):
                    break
                
                # Capture on spacebar
                if key == ord(' ') and len(faces) > 0 and current_time - last_capture_time > capture_cooldown:
                    # Use largest face
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    
                    face_crop = self.crop_face_natural(clean_frame, largest_face)
                    
                    if face_crop is not None:
                        filepath = self.save_natural_face(face_crop, category)
                        
                        if filepath:
                            captured_this_session += 1
                            last_capture_time = current_time
                            
                            print(f"✅ Natural capture {captured_this_session}/{target_count}: {os.path.basename(filepath)}")
                            
                            # Brief green flash
                            flash_frame = display_frame.copy()
                            cv2.rectangle(flash_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), 10)
                            cv2.imshow('Natural Face Capture', flash_frame)
                            cv2.waitKey(200)
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping capture...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🌟 NATURAL CAPTURE COMPLETE!")
            print(f"📈 Results:")
            print(f"   • Natural images captured: {captured_this_session}")
            print(f"   • Total boss images: {self.boss_count}")
            print(f"   • Total others images: {self.others_count}")
            print("✨ All images saved with natural, unprocessed colors!")
        
        return True
    
    def draw_minimal_interface(self, frame, faces, category, captured, target):
        """Draw minimal interface without harsh overlays"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, f"Natural {category.upper()} Capture", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress
        cv2.putText(frame, f"Progress: {captured}/{target}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Face detection - minimal green box
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Instructions
        if len(faces) > 0:
            cv2.putText(frame, "Face detected! Press SPACE to capture", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Look at camera... Q=Quit", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    parser = argparse.ArgumentParser(description="Natural Face Capture - No Enhancement")
    parser.add_argument("category", choices=["boss", "others"], 
                       help="Category to capture (boss or others)")
    parser.add_argument("-n", "--number", type=int, default=25,
                       help="Number of images to capture (default: 25)")
    
    args = parser.parse_args()
    
    print("🌟 Natural Face Capture - No Enhancement")
    print("=" * 50)
    print("Captures faces with completely natural colors and lighting")
    print("No harsh processing that can cause 'bruised' appearance")
    print()
    
    capturer = NaturalFaceCapture()
    capturer.capture_natural(args.category, args.number)

if __name__ == "__main__":
    main()
