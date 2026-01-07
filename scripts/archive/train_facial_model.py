"""
Model Training Script for Facial Emotion Recognition

Trains CNN on FER2013 or custom dataset
Includes data augmentation, validation, and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

from modules.facial_recognition import FacialEmotionRecognizer
from config import FACE_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacialEmotionTrainer:
    """
    Training pipeline for facial emotion recognition
    
    Supports:
    - Custom dataset loading
    - Data augmentation
    - Training with callbacks
    - Validation and testing
    - Performance visualization
    """
    
    def __init__(self, data_path=None):
        """
        Initialize trainer
        
        Args:
            data_path: Path to training data (CSV or image folders)
        """
        self.data_path = data_path
        self.model = None
        self.history = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.emotions = FACE_CONFIG['emotions']
        self.num_classes = len(self.emotions)
        
        logger.info("Facial emotion trainer initialized")
    
    def load_fer2013_data(self, csv_path):
        """
        Load FER2013 dataset from CSV
        
        FER2013 format:
        - emotion: 0-6 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
        - pixels: space-separated pixel values
        - Usage: Training/PublicTest/PrivateTest
        
        Args:
            csv_path: Path to fer2013.csv
        """
        logger.info(f"Loading FER2013 data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Parse pixel data
        pixels = df['pixels'].tolist()
        X = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in pixels])
        
        # Reshape to 48x48
        X = X.reshape(-1, 48, 48, 1)
        
        # Normalize
        X = X / 255.0
        
        # One-hot encode labels
        y = keras.utils.to_categorical(df['emotion'].values, num_classes=self.num_classes)
        
        # Split by usage
        train_mask = df['Usage'] == 'Training'
        val_mask = df['Usage'] == 'PublicTest'
        test_mask = df['Usage'] == 'PrivateTest'
        
        self.X_train = X[train_mask]
        self.y_train = y[train_mask]
        
        self.X_val = X[val_mask]
        self.y_val = y[val_mask]
        
        self.X_test = X[test_mask]
        self.y_test = y[test_mask]
        
        logger.info(f"Data loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
    
    def load_custom_data(self, images, labels):
        """
        Load custom dataset
        
        Args:
            images: Numpy array of images (N, H, W, C)
            labels: Numpy array of labels (N,)
        """
        # Normalize images
        X = images / 255.0
        
        # One-hot encode labels
        y = keras.utils.to_categorical(labels, num_classes=self.num_classes)
        
        # Split data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=labels
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42
        )
        
        logger.info(f"Custom data loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator
        
        Augmentations:
        - Rotation
        - Width/height shift
        - Horizontal flip
        - Zoom
        """
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        return datagen
    
    def train(self, epochs=50, batch_size=64, use_augmentation=True):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
        
        Returns:
            Training history
        """
        if self.X_train is None:
            raise ValueError("No training data loaded. Call load_*_data() first.")
        
        logger.info("Building model...")
        recognizer = FacialEmotionRecognizer()
        self.model = recognizer.model
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                str(MODELS_DIR / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
        
        if use_augmentation:
            logger.info("Using data augmentation")
            datagen = self.create_data_augmentation()
            datagen.fit(self.X_train)
            
            self.history = self.model.fit(
                datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self):
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with metrics
        """
        if self.model is None or self.X_test is None:
            raise ValueError("Model not trained or test data not available")
        
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred_probs = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Metrics
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.emotions,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true
        }
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (accuracy and loss curves)
        
        Args:
            save_path: Path to save figure
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.emotions,
            yticklabels=self.emotions
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Facial Emotion Recognition')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path=None):
        """
        Save trained model
        
        Args:
            path: Save path (uses default if None)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = MODELS_DIR / 'facial_emotion_model.h5'
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")


# Command-line training script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train facial emotion recognition model')
    parser.add_argument('--data', type=str, help='Path to FER2013 CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = FacialEmotionTrainer()
    
    # Load data
    if args.data:
        trainer.load_fer2013_data(args.data)
    else:
        # Generate dummy data for demo
        logger.warning("No data provided. Using dummy data for demo.")
        X_dummy = np.random.rand(1000, 48, 48, 1) * 255
        y_dummy = np.random.randint(0, 6, size=1000)
        trainer.load_custom_data(X_dummy, y_dummy)
    
    # Train
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation
    )
    
    # Evaluate
    results = trainer.evaluate()
    
    # Plot results
    trainer.plot_training_history(save_path='training_history.png')
    trainer.plot_confusion_matrix(results['confusion_matrix'], save_path='confusion_matrix.png')
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print("\nPer-class Performance:")
    for emotion in trainer.emotions:
        metrics = results['classification_report'][emotion]
        print(f"  {emotion:12s}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
