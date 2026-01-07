"""
Train Facial Emotion Model from Image Directories
Specifically for your FER 2013 image dataset structure
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging

from config import FACE_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_images_from_directory(data_dir):
    """
    Load FER images from directory structure
    
    Expected structure:
    data_dir/
      train/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
      test/
        angry/
        ...
    """
    data_path = Path(data_dir)
    
    # Emotion mapping
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    
    def load_split(split_dir):
        images = []
        labels = []
        
        for emotion_name, emotion_id in emotion_map.items():
            emotion_dir = split_dir / emotion_name
            if not emotion_dir.exists():
                logger.warning(f"Directory not found: {emotion_dir}")
                continue
            
            # Load all images
            for img_path in emotion_dir.glob('*.jpg'):
                try:
                    # Read image
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize to 48x48
                    img = cv2.resize(img, (48, 48))
                    
                    images.append(img)
                    labels.append(emotion_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
            
            logger.info(f"Loaded {len([l for l in labels if l == emotion_id])} images for '{emotion_name}'")
        
        return np.array(images), np.array(labels)
    
    # Load train and test
    logger.info("Loading training images...")
    X_train, y_train = load_split(data_path / 'train')
    
    logger.info("Loading test images...")
    X_test, y_test = load_split(data_path / 'test')
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
    X_test = X_test.reshape(-1, 48, 48, 1) / 255.0
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    logger.info(f"✅ Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model():
    """Build optimized CNN for facial emotion recognition"""
    model = keras.Sequential([
        # Block 1
        keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
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
        
        # Classification
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(7, activation='softmax')  # 7 emotions
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X_train, X_val, y_train, y_val, epochs=50):
    """Train the model with data augmentation"""
    
    # Build model
    model = build_model()
    logger.info(f"Model built with {model.count_params():,} parameters")
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Callbacks
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
    
    # Train
    logger.info(f"Starting training for {epochs} epochs...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_val, y_val),
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("="*60 + "\n")
    
    return test_acc, test_loss


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train facial emotion model from images')
    parser.add_argument('--data-dir', type=str, 
                       default='data/face/FER 2013',
                       help='Path to FER image directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION - TRAINING")
    print("="*60 + "\n")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_images_from_directory(args.data_dir)
    
    # Train
    model, history = train_model(X_train, X_val, y_train, y_val, epochs=args.epochs)
    
    # Evaluate
    test_acc, test_loss = evaluate_model(model, X_test, y_test)
    
    # Save final model
    save_path = MODELS_DIR / 'facial_emotion_model.h5'
    model.save(save_path)
    logger.info(f"✅ Model saved to {save_path}")
    
    print(f"\n🎉 Training complete! Accuracy: {test_acc:.2%}")
