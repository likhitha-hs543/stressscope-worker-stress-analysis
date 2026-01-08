"""
Facial Emotion Recognition Module
Handles face detection, preprocessing, and emotion classification
"""

import cv2
import numpy as np
from tensorflow import keras
from pathlib import Path
import logging
from config import FACE_CONFIG, FACE_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacialEmotionRecognizer:
    """
    CNN-based facial emotion recognition
    
    Signal chain: Frame → Face detection → Preprocessing → CNN → Emotion probabilities
    Stress inference: Angry + Fear + Sad = stress-prone emotions
    """
    
    def __init__(self, model_path=None):
        """
        Initialize facial emotion recognizer
        
        Args:
            model_path: Path to trained model (uses v2 by default)
        """
        self.input_shape = FACE_CONFIG['input_shape']
        self.emotions = FACE_CONFIG['emotions']
        self.stress_emotions = FACE_CONFIG['stress_emotions']
        self.confidence_threshold = FACE_CONFIG['confidence_threshold']
        
        # Load face detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + FACE_CONFIG['face_cascade']
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Model path - defaults to v2 (MobileNetV2 transfer learning)
        if model_path is None:
            model_path = FACE_MODEL_PATH.parent / 'facial_emotion_model_v2.keras'
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self.model = keras.models.load_model(model_path)
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            self.model = self._build_model()
            logger.info("Created new model architecture")
    
    def _build_model(self):
        """
        Build CNN architecture for facial emotion recognition
        
        Architecture:
        - 4 Convolutional blocks (feature learning)
        - Batch normalization (stability)
        - Dropout (regularization)
        - Dense layers (classification)
        
        This learns:
        - Eyebrow tension
        - Eye openness
        - Lip compression
        - Jaw stiffness
        - Cheek muscle activation
        """
        model = keras.Sequential([
            # Block 1
            keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=self.input_shape, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Block 2
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Block 3
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Block 4
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Classification head
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def detect_face(self, frame):
        """
        Step 1: Face detection - find the signal
        
        Purpose: Remove background noise, isolate face region
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            List of face regions (x, y, w, h) or empty list
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def preprocess_face(self, frame, face_coords):
        """
        Step 2: Preprocessing - make data digestible
        
        This standardizes the face so the model isn't confused by:
        - Lighting bias
        - Camera differences
        - Distance variations
        
        Think of it as putting every face under the same lab lighting.
        
        Args:
            frame: Original BGR frame
            face_coords: Tuple (x, y, w, h)
            
        Returns:
            Preprocessed face array ready for model
        """
        x, y, w, h = face_coords
        
        # Crop face region
        face = frame[y:y+h, x:x+w]
        
        # v2 model (MobileNetV2) expects RGB, v1 expects grayscale
        if self.input_shape[2] == 3:  # RGB for v2
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Resize to model input size  
            face_resized = cv2.resize(face_rgb, (self.input_shape[0], self.input_shape[1]))
        else:  # Grayscale for v1
            # Convert to grayscale
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Resize to model input size
            face_resized = cv2.resize(face_gray, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize pixel values [0, 255] → [0, 1]
        face_normalized = face_resized / 255.0
        
        # Reshape for model input (add batch dimension)
        if len(face_normalized.shape) == 2:  # Grayscale needs channel dim
            face_preprocessed = face_normalized.reshape(1, *self.input_shape)
        else:  # RGB already has channels
            face_preprocessed = np.expand_dims(face_normalized, axis=0)
        
        return face_preprocessed
    
    def predict_emotion(self, preprocessed_face):
        """
        Step 3: CNN inference - feature learning and classification
        
        The CNN learns internally:
        - Eyebrow tension
        - Eye openness
        - Lip compression
        - Jaw stiffness
        - Cheek muscle activation
        
        Stress-related emotions show:
        - Tightened jaw
        - Reduced blinking
        - Furrowed brows
        - Asymmetric expressions
        
        Args:
            preprocessed_face: Normalized face array
            
        Returns:
            Dictionary with emotion probabilities
        """
        predictions = self.model.predict(preprocessed_face, verbose=0)[0]
        
        emotion_probs = {
            emotion: float(prob)
            for emotion, prob in zip(self.emotions, predictions)
        }
        
        return emotion_probs
    
    def calculate_facial_stress_score(self, emotion_probs):
        """
        Step 4: Map emotions to stress relevance
        
        Logic:
        - Angry, Fear, Sad → stress-positive (indicate stress)
        - Happy, Neutral → stress-negative (indicate calm)
        
        This becomes a facial stress score (not diagnosis).
        
        Args:
            emotion_probs: Dictionary of emotion probabilities
            
        Returns:
            Facial stress score (0-100)
        """
        # Sum probabilities of stress-related emotions
        stress_prob = sum(
            emotion_probs[emotion]
            for emotion in self.stress_emotions
            if emotion in emotion_probs
        )
        
        # Convert to 0-100 scale
        stress_score = stress_prob * 100
        
        return stress_score
    
    def analyze_frame(self, frame):
        """
        Complete pipeline: Frame → Face → Preprocess → Predict → Score
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Dictionary with analysis results:
            {
                'face_detected': bool,
                'emotion_probs': dict,
                'dominant_emotion': str,
                'facial_stress_score': float,
                'confidence': float
            }
        """
        result = {
            'face_detected': False,
            'emotion_probs': {},
            'dominant_emotion': None,
            'facial_stress_score': 0.0,
            'confidence': 0.0
        }
        
        # Detect faces
        faces = self.detect_face(frame)
        
        if len(faces) == 0:
            logger.debug("No face detected in frame")
            return result
        
        # Use the largest face (assuming primary subject)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        
        # Preprocess face
        preprocessed_face = self.preprocess_face(frame, largest_face)
        
        # Predict emotion
        emotion_probs = self.predict_emotion(preprocessed_face)
        
        # Get dominant emotion
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[dominant_emotion]
        
        # Calculate facial stress score
        facial_stress_score = self.calculate_facial_stress_score(emotion_probs)
        
        result = {
            'face_detected': True,
            'emotion_probs': emotion_probs,
            'dominant_emotion': dominant_emotion,
            'facial_stress_score': facial_stress_score,
            'confidence': confidence,
            'face_location': largest_face.tolist()
        }
        
        return result
    
    def save_model(self, path):
        """Save trained model"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load pre-trained model"""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


# Standalone testing
if __name__ == "__main__":
    recognizer = FacialEmotionRecognizer()
    print(f"Model architecture created with {len(recognizer.emotions)} emotion classes")
    print(f"Input shape: {recognizer.input_shape}")
    print(f"Stress-related emotions: {recognizer.stress_emotions}")
