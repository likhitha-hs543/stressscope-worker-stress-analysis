"""
Optimized Speech Stress Training Script
Enhanced for better accuracy with ensemble methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import librosa
import pickle
import logging

from modules.speech_recognition import SpeechStressRecognizer
from config import SPEECH_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedSpeechTrainer:
    """Enhanced speech stress trainer with ensemble methods"""
    
    def __init__(self):
        self.recognizer = SpeechStressRecognizer()
        self.model = None
        self.scaler = StandardScaler()
        self.stress_levels = SPEECH_CONFIG['stress_levels']
        
        logger.info("Optimized speech trainer initialized")
    
    def extract_features_from_files(self, audio_files, labels):
        """Extract features with progress tracking"""
        logger.info(f"Extracting features from {len(audio_files)} files...")
        
        features = []
        valid_labels = []
        failed = 0
        
        for i, (audio_file, label) in enumerate(zip(audio_files, labels)):
            try:
                audio, sr = librosa.load(audio_file, sr=SPEECH_CONFIG['sample_rate'])
                audio = self.recognizer.preprocess_audio(audio, sr)
                feature_vector = self.recognizer.extract_all_features(audio)
                
                features.append(feature_vector)
                valid_labels.append(label)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{len(audio_files)} ({(i+1)/len(audio_files)*100:.1f}%)")
                    
            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                failed += 1
        
        X = np.array(features)
        y = np.array(valid_labels)
        
        logger.info(f"✅ Extracted {len(X)} features ({failed} failed)")
        logger.info(f"Feature shape: {X.shape}")
        
        return X, y
    
    def load_data_from_directory(self, data_dir):
        """Load and prepare data"""
        data_path = Path(data_dir)
        audio_files = []
        labels = []
        
        label_map = {'low': 0, 'medium': 1, 'high': 2}
        
        for stress_level, label_idx in label_map.items():
            level_dir = data_path / stress_level
            if not level_dir.exists():
                logger.warning(f"Directory not found: {level_dir}")
                continue
            
            for ext in ['*.wav', '*.mp3', '*.flac']:
                files = list(level_dir.glob(ext))
                audio_files.extend([str(f) for f in files])
                labels.extend([label_idx] * len(files))
        
        logger.info(f"Found {len(audio_files)} files across {len(set(labels))} classes")
        
        # Extract features
        X, y = self.extract_features_from_files(audio_files, labels)
        
        # Split with stratification
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
        )
        
        # Fit scaler
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Data split: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
    
    def build_ensemble_model(self):
        """Build ensemble of multiple classifiers"""
        logger.info("Building ensemble model (RF + GradientBoosting)")
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def train_with_grid_search(self):
        """Train with hyperparameter tuning"""
        logger.info("Training with grid search (this may take a while)...")
        
        # Define model
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [150, 200, 250],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        
        # Train
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        
        # Validation accuracy
        val_acc = self.model.score(self.X_val, self.y_val)
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        return self.model
    
    def train_ensemble(self):
        """Train ensemble model"""
        self.model = self.build_ensemble_model()
        
        logger.info("Training ensemble...")
        self.model.fit(self.X_train, self.y_train)
        
        # Validation
        val_acc = self.model.score(self.X_val, self.y_val)
        logger.info(f"✅ Ensemble validation accuracy: {val_acc:.4f}")
        
        return self.model
    
    def evaluate_comprehensive(self):
        """Comprehensive evaluation"""
        y_pred = self.model.predict(self.X_test)
        y_pred_probs = self.model.predict_proba(self.X_test)
        
        test_acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.stress_levels,
            output_dict=True
        )
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Print results
        print("\n" + "="*60)
        print("SPEECH STRESS RECOGNITION - EVALUATION RESULTS")
        print("="*60)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print("\nPer-Class Performance:")
        print("-"*60)
        
        for level in self.stress_levels:
            metrics = report[level]
            print(f"{level:10s}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-10:]
            print("\nTop 10 Most Important Features:")
            print("-"*60)
            for idx in reversed(top_features):
                print(f"Feature {idx}: {self.model.feature_importances_[idx]:.4f}")
        
        return {
            'test_accuracy': test_acc,
            'report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self):
        """Save model and scaler"""
        save_path = MODELS_DIR / 'speech_stress_model.pkl'
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': {
                'sample_rate': SPEECH_CONFIG['sample_rate'],
                'n_mfcc': SPEECH_CONFIG['n_mfcc'],
                'stress_levels': self.stress_levels
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train optimized speech stress model')
    parser.add_argument('--data-dir', type=str, required=True, help='Audio data directory')
    parser.add_argument('--method', type=str, default='ensemble', 
                       choices=['ensemble', 'gridsearch'], help='Training method')
    
    args = parser.parse_args()
    
    # Train
    trainer = OptimizedSpeechTrainer()
    trainer.load_data_from_directory(args.data_dir)
    
    if args.method == 'ensemble':
        trainer.train_ensemble()
    else:
        trainer.train_with_grid_search()
    
    # Evaluate
    results = trainer.evaluate_comprehensive()
    
    # Save
    trainer.save_model()
    
    print("\n✅ Training complete! Model ready for deployment.")
