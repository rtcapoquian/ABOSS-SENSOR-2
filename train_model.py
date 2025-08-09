#!/usr/bin/env python3
"""
Model Training Script for BossSensor
Trains a high-accuracy face recognition model
"""

import cv2
import numpy as np
import os
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
from datetime import datetime

class BossModelTrainer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.faces_dir = "faces"
        self.model_dir = "model"
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classifier = None
        
        # Training data
        self.features = []
        self.labels = []
        
    def extract_lbp_features(self, image):
        """Extract Local Binary Pattern features - more robust than simple features"""
        def get_lbp(image, radius=1, n_points=8):
            """Calculate LBP for an image"""
            h, w = image.shape
            lbp_image = np.zeros_like(image)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = image[i, j]
                    pattern = 0
                    
                    # Calculate LBP pattern
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if image[x, y] >= center:
                            pattern |= (1 << k)
                    
                    lbp_image[i, j] = pattern
            
            return lbp_image
        
        # Resize image to standard size
        image_resized = cv2.resize(image, (128, 128))
        
        # Convert to grayscale if needed
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_resized
            
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Extract LBP features
        lbp = get_lbp(equalized)
        
        # Create histogram of LBP patterns
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist_lbp = hist_lbp.astype(np.float32)
        hist_lbp /= (hist_lbp.sum() + 1e-7)  # Normalize
        
        # Extract additional features
        features = []
        
        # LBP histogram
        features.extend(hist_lbp)
        
        # HOG-like features using gradients
        grad_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Divide into 8x8 grid and compute statistics
        h, w = magnitude.shape
        grid_size = 8
        for i in range(0, h, h//grid_size):
            for j in range(0, w, w//grid_size):
                roi = magnitude[i:i+h//grid_size, j:j+w//grid_size]
                features.extend([
                    np.mean(roi),
                    np.std(roi),
                    np.max(roi)
                ])
        
        # Statistical features of the whole image
        features.extend([
            np.mean(equalized),
            np.std(equalized),
            np.median(equalized),
            np.min(equalized),
            np.max(equalized),
            np.percentile(equalized, 25),
            np.percentile(equalized, 75)
        ])
        
        # Texture features using GLCM (simplified version)
        # Calculate co-occurrence patterns
        offsets = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for offset in offsets:
            cooc = self._calculate_glcm(equalized, offset)
            features.append(np.sum(cooc * cooc))  # Energy
            features.append(np.sum(cooc**2))      # Homogeneity
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_glcm(self, image, offset, levels=16):
        """Calculate simplified GLCM (Gray Level Co-occurrence Matrix)"""
        # Reduce gray levels for computation efficiency
        image_reduced = (image // (256 // levels)).astype(np.uint8)
        
        h, w = image_reduced.shape
        glcm = np.zeros((levels, levels), dtype=np.float32)
        
        dy, dx = offset
        for i in range(h - abs(dy)):
            for j in range(w - abs(dx)):
                i1, j1 = i, j
                i2, j2 = i + dy, j + dx
                
                if 0 <= i2 < h and 0 <= j2 < w:
                    gray1 = image_reduced[i1, j1]
                    gray2 = image_reduced[i2, j2]
                    glcm[gray1, gray2] += 1
        
        # Normalize
        glcm = glcm / (glcm.sum() + 1e-7)
        return glcm
    
    def load_training_data(self):
        """Load and process training images"""
        print("Loading training data...")
        
        boss_dir = os.path.join(self.faces_dir, "boss")
        others_dir = os.path.join(self.faces_dir, "others")
        
        # Check if directories exist
        if not os.path.exists(boss_dir) or not os.path.exists(others_dir):
            print("Error: Training directories not found!")
            print("Please run capture_faces.py first to collect training data.")
            return False
        
        # Load boss images
        boss_count = 0
        for filename in os.listdir(boss_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(boss_dir, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Extract faces from image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                    
                    if len(faces) > 0:
                        # Use the largest face
                        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                        x, y, w, h = largest_face
                        face_roi = image[y:y+h, x:x+w]
                        
                        # Extract features
                        features = self.extract_lbp_features(face_roi)
                        self.features.append(features)
                        self.labels.append("boss")
                        boss_count += 1
                    else:
                        # If no face detected in the image, use the whole image as crop
                        if "crop" in filename:  # This is already a face crop
                            features = self.extract_lbp_features(image)
                            self.features.append(features)
                            self.labels.append("boss")
                            boss_count += 1
        
        print(f"Loaded {boss_count} boss images")
        
        # Load others images
        others_count = 0
        for filename in os.listdir(others_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(others_dir, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Extract faces from image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                    
                    if len(faces) > 0:
                        # Use all detected faces for others category
                        for (x, y, w, h) in faces:
                            face_roi = image[y:y+h, x:x+w]
                            features = self.extract_lbp_features(face_roi)
                            self.features.append(features)
                            self.labels.append("others")
                            others_count += 1
                    else:
                        # If no face detected, use whole image if it's a crop
                        if "crop" in filename:
                            features = self.extract_lbp_features(image)
                            self.features.append(features)
                            self.labels.append("others")
                            others_count += 1
        
        print(f"Loaded {others_count} other person images")
        
        if len(self.features) == 0:
            print("Error: No training data found!")
            return False
            
        print(f"Total training samples: {len(self.features)}")
        return True
    
    def train_model(self):
        """Train the classification model"""
        print("Training model...")
        
        # Convert to numpy arrays
        X = np.array(self.features)
        y = np.array(self.labels)
        
        print(f"Feature vector size: {X.shape[1]}")
        print(f"Training samples: {X.shape[0]}")
        print(f"Classes: {np.unique(y)}")
        
        # Check if we have at least 2 classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"\n❌ Error: Need at least 2 classes for training!")
            print(f"Currently have: {unique_classes}")
            print("Please capture 'others' images using:")
            print("python capture_faces.py others 25")
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Try different classifiers and pick the best
        classifiers = {
            'SVM_RBF': SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'),
            'SVM_Linear': SVC(kernel='linear', probability=True, C=1.0),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        }
        
        best_score = 0
        best_classifier = None
        best_name = ""
        
        for name, clf in classifiers.items():
            print(f"\nTesting {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
            mean_cv_score = np.mean(cv_scores)
            print(f"Cross-validation score: {mean_cv_score:.4f} (+/- {np.std(cv_scores) * 2:.4f})")
            
            # Train on full training set
            clf.fit(X_train, y_train)
            
            # Test on test set
            y_pred = clf.predict(X_test)
            test_score = accuracy_score(y_test, y_pred)
            print(f"Test accuracy: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_classifier = clf
                best_name = name
        
        print(f"\nBest classifier: {best_name} with accuracy: {best_score:.4f}")
        
        # Use the best classifier
        self.classifier = best_classifier
        
        # Final evaluation
        y_pred_final = self.classifier.predict(X_test)
        
        print("\n=== Final Model Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final, target_names=self.label_encoder.classes_))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_final))
        
        return True
    
    def save_model(self):
        """Save the trained model"""
        print("Saving model...")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_size': len(self.features[0]) if self.features else 0,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.features)
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, "boss_detector.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save configuration
        config = {
            'model_path': model_path,
            'detection_threshold': 0.75,  # High threshold for accuracy
            'consecutive_detections': 3,   # Require multiple detections
            'camera_width': 640,
            'camera_height': 480,
            'fps': 10
        }
        
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Config saved to {config_path}")
        
        return True

def main():
    print("=== BossSensor Model Training ===\n")
    
    trainer = BossModelTrainer()
    
    # Load training data
    if not trainer.load_training_data():
        print("❌ Failed to load training data!")
        print("\nMake sure you have:")
        print("1. faces/boss/ directory with boss images")
        print("2. faces/others/ directory with other people images")
        print("\nRun capture_faces.py to collect training data first.")
        return
    
    # Check if we have enough data
    boss_count = sum(1 for label in trainer.labels if label == "boss")
    others_count = sum(1 for label in trainer.labels if label == "others")
    
    print(f"Training data summary:")
    print(f"- Boss samples: {boss_count}")
    print(f"- Others samples: {others_count}")
    
    if others_count == 0:
        print("\n❌ ERROR: No 'others' training data found!")
        print("Machine learning needs at least 2 classes to work.")
        print("\nYou must capture 'others' data first:")
        print("python capture_faces.py others 25")
        print("\nThis should include colleagues, friends, or family members.")
        print("The AI needs to learn the difference between boss and others!")
        return
    
    if boss_count < 10 or others_count < 10:
        print("\n⚠ Warning: Not enough training data for high accuracy!")
        print("Recommended: At least 20 samples per class")
        print("Continue anyway? (y/n): ", end="")
        
        if input().lower() != 'y':
            return
    
    # Train model
    if not trainer.train_model():
        print("❌ Failed to train model!")
        return
    
    # Save model
    if not trainer.save_model():
        print("❌ Failed to save model!")
        return
    
    print("\n🎉 Model training completed successfully!")
    print("\nNext steps:")
    print("1. Test the model: python boss_detector.py")
    print("2. Start monitoring: python boss_detector.py --monitor")
    
    print(f"\nModel performance tips:")
    print(f"- Accuracy depends on training data quality")
    print(f"- Add more training images if accuracy is low")
    print(f"- Ensure good lighting during capture and detection")
    print(f"- Model works best with consistent camera positioning")

if __name__ == "__main__":
    main()
