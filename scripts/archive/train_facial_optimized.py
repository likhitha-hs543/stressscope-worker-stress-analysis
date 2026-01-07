"""
Optimized Facial Emotion Training Script
Enhanced for better accuracy with FER2013 dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging

from modules.facial_recognition import FacialEmotionRecognizer
from config import FACE_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedFacialTrainer:
    """
    Enhanced training pipeline with accuracy optimizations
    """
    
    def __init__(self):
        self.emotions = FACE_CONFIG['emotions']
        self.num_classes = len(self.emotions)
        self.model = None
        self.history = None
        
        logger.info("Optimized facial trainer initialized")
    
    def create_enhanced_augmentation(self):
        """Enhanced data augmentation for better generalization"""
        datagen = ImageDataGenerator(
            rotation_range=20,              # Increased rotation
            width_shift_range=0.15,         # Increased shift
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,                # Increased zoom
            shear_range=0.1,                # Added shear
            brightness_range=[0.8, 1.2],    # Added brightness variation
            fill_mode='nearest'
        )
        return datagen
    
    def build_optimized_model(self):
        """
        Improved CNN architecture with:
        - Deeper network
        - More robust regularization
        - Better normalization
        """
        input_shape = FACE_CONFIG['input_shape']
        
        model = keras.Sequential([
            # Block 1
            keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Block 2
            keras.layers.Conv2D(128, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(128, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Block 3
            keras.layers.Conv2D(256, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(256, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),
            
            # Block 4
            keras.layers.Conv2D(512, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.4),
            
            # Classification head
            keras.layers.Flatten(),
            keras.layers.Dense(512),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Use Adam with custom learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_fer2013_data(self, csv_path):
        """Load and preprocess FER2013 dataset"""
        logger.info(f"Loading FER2013 from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Parse pixels
        pixels = df['pixels'].tolist()
        X = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in pixels])
        X = X.reshape(-1, 48, 48, 1) / 255.0
        
        # Labels
        y = keras.utils.to_categorical(df['emotion'].values, num_classes=self.num_classes)
        
        # Split
        train_mask = df['Usage'] == 'Training'
        val_mask = df['Usage'] == 'PublicTest'
        test_mask = df['Usage'] == 'PrivateTest'
        
        self.X_train = X[train_mask]
        self.y_train = y[train_mask]
        self.X_val = X[val_mask]
        self.y_val = y[val_mask]
        self.X_test = X[test_mask]
        self.y_test = y[test_mask]
        
        logger.info(f"Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
    
    def train_optimized(self, epochs=100, batch_size=64):
        """Train with optimized settings"""
        
        # Build model
        self.model = self.build_optimized_model()
        
        # Calculate class weights for imbalanced data
        y_integers = np.argmax(self.y_train, axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                str(MODELS_DIR / 'best_facial_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = self.create_enhanced_augmentation()
        datagen.fit(self.X_train)
        
        logger.info(f"Training for {epochs} epochs with enhanced augmentation")
        
        # Train
        self.history = self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_comprehensive(self):
        """Comprehensive evaluation with detailed metrics"""
        
        # Predictions
        y_pred_probs = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Overall accuracy
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Per-class metrics
        report = classification_report(
            y_true, y_pred,
            target_names=self.emotions,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print detailed results
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Test Accuracy: {test_acc:.4f}")
        print(f"Overall Test Loss: {test_loss:.4f}")
        print("\nPer-Class Performance:")
        print("-"*60)
        
        for emotion in self.emotions:
            metrics = report[emotion]
            print(f"{emotion:12s}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        print("\nStress-Related Emotions Performance:")
        print("-"*60)
        stress_emotions = FACE_CONFIG['stress_emotions']
        stress_indices = [self.emotions.index(e) for e in stress_emotions]
        stress_accuracy = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true)) if y_true[i] in stress_indices])
        print(f"Stress Detection Accuracy: {stress_accuracy:.4f}")
        
        return {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'report': report,
            'confusion_matrix': cm,
            'stress_accuracy': stress_accuracy
        }
    
    def save_final_model(self):
        """Save the trained model"""
        save_path = MODELS_DIR / 'facial_emotion_model.h5'
        self.model.save(save_path)
        logger.info(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train optimized facial emotion model')
    parser.add_argument('--data', type=str, required=True, help='Path to FER2013 CSV')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    # Train
    trainer = OptimizedFacialTrainer()
    trainer.load_fer2013_data(args.data)
    trainer.train_optimized(epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    results = trainer.evaluate_comprehensive()
    
    # Save
    trainer.save_final_model()
    
    print("\n✅ Training complete! Model ready for deployment.")
