#!/usr/bin/env python3
"""
Advanced Face Capture and Crop Script for BossSensor Training
Automatically detects, crops, and saves faces for model training
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse

class FaceCaptureAndCrop:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create directories
        self.faces_dir = "faces"
        self.boss_dir = os.path.join(self.faces_dir, "boss")
        self.others_dir = os.path.join(self.faces_dir, "others")
        
        os.makedirs(self.boss_dir, exist_ok=True)
        os.makedirs(self.others_dir, exist_ok=True)
        
        # Count existing images
        self.boss_count = len([f for f in os.listdir(self.boss_dir) if f.endswith('.jpg')])
        self.others_count = len([f for f in os.listdir(self.others_dir) if f.endswith('.jpg')])
        
        # Settings for quality crops
        self.min_face_size = (50, 50)
        self.max_face_size = (300, 300)
        self.crop_padding = 20
        self.target_size = (128, 128)  # Standard size for training
        
        print("🎯 Face Capture & Crop System Initialized")
        print(f"📊 Current counts - Boss: {self.boss_count}, Others: {self.others_count}")
    
    def detect_faces_multi_method(self, frame):
        """Enhanced face detection with multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        
        # Method 1: Standard detection
        detected = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=self.min_face_size, 
            maxSize=self.max_face_size
        )
        faces.extend(detected)
        
        # Method 2: More sensitive if no faces found
        if len(faces) == 0:
            detected = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=3, 
                minSize=(30, 30), 
                maxSize=(400, 400)
            )
            faces.extend(detected)
        
        # Method 3: With histogram equalization
        if len(faces) == 0:
            equalized = cv2.equalizeHist(gray)
            detected = self.face_cascade.detectMultiScale(
                equalized, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(25, 25), 
                maxSize=(400, 400)
            )
            faces.extend(detected)
        
        # Method 4: With CLAHE enhancement
        if len(faces) == 0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            detected = self.face_cascade.detectMultiScale(
                enhanced, 
                scaleFactor=1.08, 
                minNeighbors=3, 
                minSize=(20, 20), 
                maxSize=(500, 500)
            )
            faces.extend(detected)
        
        return self.remove_overlapping_faces(faces)
    
    def remove_overlapping_faces(self, faces):
        """Remove overlapping face detections"""
        if len(faces) <= 1:
            return faces
        
        # Convert to list for easier manipulation
        faces_list = list(faces)
        filtered_faces = []
        
        for i, (x1, y1, w1, h1) in enumerate(faces_list):
            keep = True
            for j, (x2, y2, w2, h2) in enumerate(faces_list):
                if i != j:
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    area1 = w1 * h1
                    area2 = w2 * h2
                    min_area = min(area1, area2)
                    
                    # If overlap > 50% of smaller face, keep the larger one
                    if overlap_area > 0.5 * min_area:
                        if area1 < area2:
                            keep = False
                            break
            
            if keep:
                filtered_faces.append((x1, y1, w1, h1))
        
        return filtered_faces
    
    def crop_face(self, frame, face_rect):
        """Crop face with padding and resize to standard size"""
        x, y, w, h = face_rect
        
        # Add padding
        x1 = max(0, x - self.crop_padding)
        y1 = max(0, y - self.crop_padding)
        x2 = min(frame.shape[1], x + w + self.crop_padding)
        y2 = min(frame.shape[0], y + h + self.crop_padding)
        
        # Extract face region
        face_crop = frame[y1:y2, x1:x2]
        
        # Resize to standard size
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, self.target_size)
            return face_resized
        
        return None
    
    def enhance_face_quality(self, face_img):
        """Enhance face image quality for better training - gentle enhancement"""
        if face_img is None:
            return None
        
        # Very gentle enhancement to preserve natural skin tones
        if len(face_img.shape) == 3:
            # For color images, apply very mild enhancement
            enhanced = face_img.copy()
            
            # Option 1: No enhancement (natural look)
            # return enhanced
            
            # Option 2: Very gentle brightness/contrast adjustment
            # Slightly increase brightness and contrast without harsh processing
            alpha = 1.05  # Contrast (1.0 = no change)
            beta = 5      # Brightness (0 = no change)
            enhanced = cv2.convertScaleAbs(face_img, alpha=alpha, beta=beta)
            
            # Optional: Very gentle gamma correction for better lighting
            gamma = 0.95  # < 1 = brighter, > 1 = darker
            lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, lookup_table)
            
        else:
            # For grayscale images, just slight enhancement
            enhanced = cv2.convertScaleAbs(face_img, alpha=1.05, beta=5)
        
        return enhanced
    
    def save_face_crop(self, face_crop, category):
        """Save cropped face to appropriate category folder"""
        if face_crop is None:
            return None
        
        # Enhance quality
        enhanced_face = self.enhance_face_quality(face_crop)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if category.lower() == "boss":
            self.boss_count += 1
            filename = f"boss_{self.boss_count:04d}_{timestamp}_crop.jpg"
            filepath = os.path.join(self.boss_dir, filename)
        else:
            self.others_count += 1
            filename = f"others_{self.others_count:04d}_{timestamp}_crop.jpg"
            filepath = os.path.join(self.others_dir, filename)
        
        # Save the enhanced face crop
        success = cv2.imwrite(filepath, enhanced_face)
        
        if success:
            return filepath
        return None
    
    def capture_session(self, category="boss", target_count=25, auto_mode=False):
        """Main capture session"""
        print(f"\n🎯 FACE CAPTURE & CROP SESSION STARTED")
        print(f"📋 Mode: {category.upper()} images")
        print(f"🎪 Target: {target_count} high-quality face crops")
        
        if auto_mode:
            print("🤖 AUTO MODE: Will capture automatically when face detected")
            print("   • Good faces will be saved automatically")
            print("   • Press 'Q' to stop early")
        else:
            print("🎮 MANUAL MODE:")
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
        
        print("✅ Camera ready! Position your face in view...")
        
        captured_this_session = 0
        last_capture_time = 0
        auto_capture_interval = 1.0  # 1 second between auto captures
        quality_threshold = 3000  # Minimum face area for quality
        
        try:
            while captured_this_session < target_count:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Store original clean frame for cropping
                clean_frame = frame.copy()
                
                # Detect faces
                faces = self.detect_faces_multi_method(frame)
                
                # Draw interface on display frame only
                display_frame = frame.copy()
                self.draw_capture_interface(display_frame, faces, category, captured_this_session, target_count)
                
                # Show display frame with UI
                cv2.imshow('Face Capture & Crop', display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()
                
                if key == ord('q') or key == ord('Q'):
                    break
                
                # Capture logic
                if len(faces) > 0:
                    # Find best face (largest and good quality)
                    best_face = None
                    best_quality = 0
                    
                    for face_rect in faces:
                        x, y, w, h = face_rect
                        face_area = w * h
                        
                        # Quality score based on size and position
                        center_x = x + w // 2
                        center_y = y + h // 2
                        frame_center_x = frame.shape[1] // 2
                        frame_center_y = frame.shape[0] // 2
                        
                        # Distance from center (prefer centered faces)
                        distance_from_center = np.sqrt(
                            (center_x - frame_center_x)**2 + 
                            (center_y - frame_center_y)**2
                        )
                        
                        # Quality score (higher is better)
                        quality = face_area / (1 + distance_from_center * 0.01)
                        
                        if quality > best_quality and face_area > quality_threshold:
                            best_quality = quality
                            best_face = face_rect
                    
                    # Capture decision
                    should_capture = False
                    
                    if auto_mode and best_face is not None:
                        # Auto capture if enough time passed
                        if current_time - last_capture_time > auto_capture_interval:
                            should_capture = True
                    elif not auto_mode and (key == ord(' ') or key == 13):  # Space or Enter
                        # Manual capture
                        if best_face is not None:
                            should_capture = True
                    
                    # Perform capture
                    if should_capture and best_face is not None:
                        # Use clean frame without UI elements for cropping
                        face_crop = self.crop_face(clean_frame, best_face)
                        
                        if face_crop is not None:
                            filepath = self.save_face_crop(face_crop, category)
                            
                            if filepath:
                                captured_this_session += 1
                                last_capture_time = current_time
                                
                                print(f"✅ Captured {captured_this_session}/{target_count}: {os.path.basename(filepath)}")
                                
                                # Visual feedback on display frame
                                self.show_capture_feedback(display_frame, f"SAVED! ({captured_this_session}/{target_count})")
                            else:
                                print("⚠ Failed to save image")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping capture...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🎊 CAPTURE SESSION COMPLETE!")
            print(f"📈 Results:")
            print(f"   • Captured this session: {captured_this_session}")
            print(f"   • Total boss images: {self.boss_count}")
            print(f"   • Total others images: {self.others_count}")
            
            # Training readiness check
            if self.boss_count >= 10 and self.others_count >= 10:
                print(f"\n🎯 READY FOR TRAINING! Run:")
                print(f"   python train_model.py")
            else:
                needed = []
                if self.boss_count < 10:
                    needed.append(f"boss: need {10 - self.boss_count} more")
                if self.others_count < 10:
                    needed.append(f"others: need {10 - self.others_count} more")
                
                print(f"\n⚠ Need more training data:")
                print(f"   • {', '.join(needed)}")
        
        return True
    
    def draw_capture_interface(self, frame, faces, category, captured, target):
        """Draw capture interface"""
        h, w = frame.shape[:2]
        
        # Header background
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, f"CAPTURING: {category.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Progress
        progress_text = f"Progress: {captured}/{target}"
        cv2.putText(frame, progress_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x = 10
        bar_y = 70
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Progress bar fill
        progress_ratio = captured / target if target > 0 else 0
        fill_width = int(bar_width * progress_ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Instructions
        if len(faces) > 0:
            cv2.putText(frame, "FACE DETECTED! Press SPACE to capture", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Position face in camera view...", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw face detections with quality indicators
        for i, (x, y, w, h) in enumerate(faces):
            face_area = w * h
            
            # Color based on quality
            if face_area > 5000:
                color = (0, 255, 0)  # Green for good quality
                quality_text = "EXCELLENT"
            elif face_area > 3000:
                color = (0, 255, 255)  # Yellow for medium quality
                quality_text = "GOOD"
            else:
                color = (0, 0, 255)  # Red for poor quality
                quality_text = "POOR"
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw crop preview
            crop_x1 = max(0, x - self.crop_padding)
            crop_y1 = max(0, y - self.crop_padding)
            crop_x2 = min(w, x + w + self.crop_padding)
            crop_y2 = min(h, y + h + self.crop_padding)
            cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color, 1)
            
            # Quality indicator
            cv2.putText(frame, quality_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Size info
            cv2.putText(frame, f"{w}x{h}", (x, y+h+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Controls at bottom
        cv2.putText(frame, "Controls: SPACE=Capture | Q=Quit", 
                   (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def show_capture_feedback(self, frame, message):
        """Show visual feedback when image is captured"""
        feedback_frame = frame.copy()
        
        # Green flash effect
        overlay = feedback_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
        cv2.addWeighted(feedback_frame, 0.7, overlay, 0.3, 0, feedback_frame)
        
        # Message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(feedback_frame, message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        cv2.imshow('Face Capture & Crop', feedback_frame)
        cv2.waitKey(300)  # Show for 300ms

def main():
    parser = argparse.ArgumentParser(description="Face Capture & Crop for BossSensor Training")
    parser.add_argument("category", choices=["boss", "others"], 
                       help="Category to capture (boss or others)")
    parser.add_argument("-n", "--number", type=int, default=25,
                       help="Number of images to capture (default: 25)")
    parser.add_argument("--auto", action="store_true",
                       help="Enable automatic capture mode")
    
    args = parser.parse_args()
    
    print("🎯 Face Capture & Crop for BossSensor Training")
    print("=" * 50)
    print("This tool captures and crops faces for high-quality training data")
    print()
    
    capturer = FaceCaptureAndCrop()
    capturer.capture_session(args.category, args.number, args.auto)

if __name__ == "__main__":
    main()
