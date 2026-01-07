"""
Facial Emotion Recognition v2 - Transfer Learning with MobileNetV2
Goal: Improve accuracy from 13% (v1) to 55-70% (v2)

This script demonstrates iterative ML engineering:
- v1: Custom CNN from scratch (baseline)
- v2: Transfer learning (improvement)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging

from config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FacialModelV2Trainer:
    """
    Transfer Learning Trainer for Facial Emotion Recognition
    Uses MobileNetV2 pretrained on ImageNet
    """
    
    def __init__(self, data_dir='data/face/FER 2013'):
        self.data_dir = Path(data_dir)
        self.img_size = (224, 224)  # MobileNetV2 standard input
        self.batch_size = 32
        self.num_classes = 7
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        logger.info("Facial Model v2 Trainer initialized")
        logger.info(f"Dataset: {self.data_dir}")
        logger.info(f"Target size: {self.img_size}")
    
    def create_data_generators(self):
        """
        Create data generators with STRONG augmentation
        Critical for FER2013 to prevent overfitting
        """
        
        # Training: aggressive augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation & test: only rescaling
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load from directories
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='rgb',  # Convert grayscale to RGB
            shuffle=True
        )
        
        # Check if val directory exists, otherwise use test
        val_dir = self.data_dir / 'val'
        if not val_dir.exists():
            val_dir = self.data_dir / 'test'
            logger.info("No 'val' directory found, using 'test' for validation")
        
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False
        )
        
        logger.info(f"✅ Train samples: {train_generator.samples}")
        logger.info(f"✅ Validation samples: {val_generator.samples}")
        logger.info(f"✅ Test samples: {test_generator.samples}")
        
        return train_generator, val_generator, test_generator
    
    def build_transfer_learning_model(self):
        """
        Build model using MobileNetV2 pretrained on ImageNet
        
        Architecture:
        - MobileNetV2 (frozen) - pretrained features
        - GlobalAveragePooling2D
        - BatchNormalization
        - Dense(256) + Dropout
        - Dense(7) - emotion classification
        """
        
        logger.info("Building MobileNetV2 transfer learning model...")
        
        # Load pretrained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # CRITICAL: Freeze base model
        base_model.trainable = False
        
        logger.info(f"Base model loaded: {len(base_model.layers)} layers (frozen)")
        
        # Build classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        total_params = sum([tf.size(w).numpy() for w in model.weights])
        
        logger.info(f"✅ Model built:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Frozen parameters: {total_params - trainable_params:,}")
        
        return model, base_model
    
    def train_phase1_frozen(self, model, train_gen, val_gen, epochs=15):
        """
        Phase 1: Train classification head with frozen base
        
        This teaches the model emotion mapping using pretrained features
        """
        
        logger.info("="*60)
        logger.info("PHASE 1: Training classification head (base frozen)")
        logger.info("="*60)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                str(MODELS_DIR / 'best_facial_v2_phase1.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def train_phase2_finetune(self, model, base_model, train_gen, val_gen, epochs=10):
        """
        Phase 2 (OPTIONAL): Fine-tune top layers
        
        Unfreezes last 30 layers for fine-tuning
        +5-10% accuracy improvement
        """
        
        logger.info("="*60)
        logger.info("PHASE 2: Fine-tuning (unfreezing top layers)")
        logger.info("="*60)
        
        # Unfreeze last 30 layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
        logger.info(f"Unfrozen {trainable_layers} layers for fine-tuning")
        
        # Recompile with LOWER learning rate
        model.compile(
            optimizer=Adam(learning_rate=1e-5),  # 10x lower!
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                str(MODELS_DIR / 'best_facial_v2_phase2.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, model, test_gen):
        """Evaluate model on test set"""
        
        logger.info("="*60)
        logger.info("EVALUATION ON TEST SET")
        logger.info("="*60)
        
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        
        print(f"\n✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"✅ Test Loss: {test_loss:.4f}\n")
        
        return test_acc, test_loss
    
    def save_model(self, model, version='v2'):
        """Save final model"""
        
        save_path = MODELS_DIR / f'facial_emotion_model_{version}.keras'
        model.save(save_path)
        
        logger.info(f"✅ Model saved to: {save_path}")
        
        return save_path


def main():
    """Main training pipeline"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Facial Model v2 with MobileNetV2')
    parser.add_argument('--data-dir', type=str, default='data/face/FER 2013',
                       help='Path to FER dataset')
    parser.add_argument('--phase1-epochs', type=int, default=15,
                       help='Epochs for phase 1 (frozen base)')
    parser.add_argument('--phase2-epochs', type=int, default=10,
                       help='Epochs for phase 2 (fine-tuning) - set to 0 to skip')
    parser.add_argument('--skip-phase2', action='store_true',
                       help='Skip fine-tuning phase 2')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION v2 - TRANSFER LEARNING")
    print("="*60)
    print(f"Dataset: {args.data_dir}")
    print(f"Phase 1 epochs: {args.phase1_epochs}")
    print(f"Phase 2 epochs: {args.phase2_epochs if not args.skip_phase2 else 'SKIPPED'}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = FacialModelV2Trainer(data_dir=args.data_dir)
    
    # Create data generators
    train_gen, val_gen, test_gen = trainer.create_data_generators()
    
    # Build model
    model, base_model = trainer.build_transfer_learning_model()
    
    # Phase 1: Train with frozen base
    history1 = trainer.train_phase1_frozen(
        model, train_gen, val_gen,
        epochs=args.phase1_epochs
    )
    
    # Phase 2: Optional fine-tuning
    if not args.skip_phase2 and args.phase2_epochs > 0:
        history2 = trainer.train_phase2_finetune(
            model, base_model, train_gen, val_gen,
            epochs=args.phase2_epochs
        )
    
    # Evaluate
    test_acc, test_loss = trainer.evaluate(model, test_gen)
    
    # Save
    save_path = trainer.save_model(model, version='v2')
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - MODEL COMPARISON")
    print("="*60)
    print(f"Facial v1 (scratch):        13.21%")
    print(f"Facial v2 (MobileNetV2):    {test_acc*100:.2f}%")
    print(f"Improvement:                 +{(test_acc*100 - 13.21):.2f}pp")
    print("="*60)
    print(f"\n✅ Model saved: {save_path}")
    print("✅ Ready to update fusion weights and test system!")
    print("\nNext steps:")
    print("  1. Run: python validate_system.py")
    print("  2. Update fusion to use v2 model")
    print("  3. Test improved system!")
    print("\n")


if __name__ == "__main__":
    main()
