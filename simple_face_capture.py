#!/usr/bin/env python3
"""
Simple Face Capture Script
Automatically detects faces and lets you save them as 'boss' or 'others'
"""

import cv2
import os
import time
from datetime import datetime

class SimpleFaceCapture:
    def __init__(self):
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create directories
        os.makedirs("faces/boss", exist_ok=True)
        os.makedirs("faces/others", exist_ok=True)
        
        # Counters
        self.boss_count = len([f for f in os.listdir("faces/boss") if f.endswith('.jpg')])
        self.others_count = len([f for f in os.listdir("faces/others") if f.endswith('.jpg')])
        
        print("✅ Face capture system ready!")
        print(f"📊 Current counts - Boss: {self.boss_count}, Others: {self.others_count}")
    
    def detect_faces(self, frame):
        """Simple but effective face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection parameters
        faces = []
        
        # Method 1: Standard detection
        detected = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), maxSize=(300, 300)
        )
        if len(detected) > 0:
            faces.extend(detected)
        
        # Method 2: More sensitive detection if no faces found
        if len(faces) == 0:
            detected = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30), maxSize=(400, 400)
            )
            faces.extend(detected)
        
        # Method 3: Very sensitive detection
        if len(faces) == 0:
            # Enhance image
            enhanced = cv2.equalizeHist(gray)
            detected = self.face_cascade.detectMultiScale(
                enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25), maxSize=(400, 400)
            )
            faces.extend(detected)
        
        return faces
    
    def save_face(self, frame, face_rect, category):
        """Save detected face to the specified category"""
        x, y, w, h = face_rect
        
        # Add padding around face
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract face with padding
        face_crop = frame[y1:y2, x1:x2]
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        if category == "boss":
            self.boss_count += 1
            filename = f"faces/boss/boss_{self.boss_count:04d}_{timestamp}.jpg"
        else:
            self.others_count += 1
            filename = f"faces/others/others_{self.others_count:04d}_{timestamp}.jpg"
        
        # Save the image
        cv2.imwrite(filename, face_crop)
        return filename
    
    def start_capture(self):
        """Start the face capture session"""
        print("\n🎯 SMART FACE CAPTURE STARTED")
        print("📋 Instructions:")
        print("   • Look at the camera - faces will be detected automatically")
        print("   • When face is detected (green box):")
        print("     - Press 'B' to save as BOSS")
        print("     - Press 'O' to save as OTHERS")
        print("   • Press 'Q' to quit")
        print("   • Press 'S' to see statistics")
        print("\n🚀 Starting camera...")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access camera!")
            return False
        
        print("✅ Camera started! Look at the camera...")
        
        # Variables for smooth capture
        last_save_time = 0
        save_cooldown = 1.0  # 1 second between saves
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ Warning: Cannot read from camera")
                    continue
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Store clean frame for cropping (without UI elements)
                clean_frame = frame.copy()
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Draw interface on display frame only
                display_frame = frame.copy()
                self.draw_interface(display_frame, faces)
                
                # Show display frame with UI
                cv2.imshow('Smart Face Capture', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()
                
                if key == ord('q') or key == ord('Q'):
                    break
                
                elif key == ord('s') or key == ord('S'):
                    # Show statistics
                    print(f"\n📊 Statistics:")
                    print(f"   Boss images: {self.boss_count}")
                    print(f"   Others images: {self.others_count}")
                    print(f"   Total: {self.boss_count + self.others_count}")
                
                elif len(faces) > 0 and current_time - last_save_time > save_cooldown:
                    # Save logic - use clean frame for cropping
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    
                    if key == ord('b') or key == ord('B'):
                        filename = self.save_face(clean_frame, largest_face, "boss")
                        print(f"✅ BOSS image saved: {filename}")
                        last_save_time = current_time
                        
                        # Visual feedback on display frame
                        self.show_save_feedback(display_frame, "BOSS SAVED!", (0, 0, 255))
                        
                    elif key == ord('o') or key == ord('O'):
                        filename = self.save_face(clean_frame, largest_face, "others")
                        print(f"✅ OTHERS image saved: {filename}")
                        last_save_time = current_time
                        
                        # Visual feedback on display frame
                        self.show_save_feedback(display_frame, "OTHERS SAVED!", (0, 255, 0))
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping capture...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🎊 CAPTURE SESSION COMPLETE!")
            print(f"📈 Final Results:")
            print(f"   Boss images: {self.boss_count}")
            print(f"   Others images: {self.others_count}")
            print(f"   Total captured: {self.boss_count + self.others_count}")
            
            # Next steps
            if self.boss_count > 0 and self.others_count > 0:
                print(f"\n🎯 Ready for training! Run:")
                print(f"   python train_model.py")
            elif self.boss_count > 0:
                print(f"\n⚠ Need 'others' images for training!")
            elif self.others_count > 0:
                print(f"\n⚠ Need 'boss' images for training!")
        
        return True
    
    def draw_interface(self, frame, faces):
        """Draw user interface on frame"""
        h, w = frame.shape[:2]
        
        # Draw header background
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "SMART FACE CAPTURE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Counters
        cv2.putText(frame, f"Boss: {self.boss_count} | Others: {self.others_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        if len(faces) > 0:
            cv2.putText(frame, "FACE DETECTED! Press B=Boss, O=Others", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Look at camera to detect face...", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw face detections
        for i, (x, y, w, h) in enumerate(faces):
            # Different colors for different faces
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw face number
            cv2.putText(frame, f"Face {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw size info
            cv2.putText(frame, f"{w}x{h}", (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw controls at bottom
        cv2.putText(frame, "Controls: B=Boss | O=Others | Q=Quit | S=Stats", 
                   (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def show_save_feedback(self, frame, message, color):
        """Show visual feedback when image is saved"""
        feedback_frame = frame.copy()
        
        # Draw colored border
        cv2.rectangle(feedback_frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 15)
        
        # Draw message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(feedback_frame, message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        cv2.imshow('Smart Face Capture', feedback_frame)
        cv2.waitKey(500)  # Show for 500ms

def main():
    print("🎯 Smart Face Capture for BossSensor")
    print("=====================================")
    print("This tool automatically detects faces and lets you")
    print("categorize them as 'boss' or 'others' in real-time!")
    print()
    
    # Create and start capture
    capture = SimpleFaceCapture()
    capture.start_capture()

if __name__ == "__main__":
    main()
