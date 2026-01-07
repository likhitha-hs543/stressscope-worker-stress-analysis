"""
Model Training Script for Speech Stress Recognition

Trains Random Forest (or other ML model) on speech features
Includes feature extraction, training, and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import librosa
import pickle
import logging

from modules.speech_recognition import SpeechStressRecognizer
from config import SPEECH_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechStressTrainer:
    """
    Training pipeline for speech stress recognition
    
    Supports:
    - Feature extraction from audio files
    - Multiple classifiers (RandomForest, SVM)
    - Cross-validation
    - Performance evaluation
    """
    
    def __init__(self):
        """Initialize trainer"""
        self.recognizer = SpeechStressRecognizer()
        self.model = None
        self.scaler = StandardScaler()
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.stress_levels = SPEECH_CONFIG['stress_levels']
        
        logger.info("Speech stress trainer initialized")
    
    def extract_features_from_files(self, audio_files, labels):
        """
        Extract features from audio files
        
        Args:
            audio_files: List of audio file paths
            labels: List of stress level labels (0=Low, 1=Medium, 2=High)
        
        Returns:
            Feature array, label array
        """
        logger.info(f"Extracting features from {len(audio_files)} audio files...")
        
        features = []
        valid_labels = []
        
        for i, (audio_file, label) in enumerate(zip(audio_files, labels)):
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=SPEECH_CONFIG['sample_rate'])
                
                # Preprocess
                audio = self.recognizer.preprocess_audio(audio, sr)
                
                # Extract features
                feature_vector = self.recognizer.extract_all_features(audio)
                
                features.append(feature_vector)
                valid_labels.append(label)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(audio_files)} files")
            
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
                continue
        
        X = np.array(features)
        y = np.array(valid_labels)
        
        logger.info(f"Feature extraction complete: {X.shape}")
        
        return X, y
    
    def load_data_from_directory(self, data_dir):
        """
        Load audio data from directory structure
        
        Expected structure:
        data_dir/
          low/
            file1.wav
            file2.wav
          medium/
            file1.wav
          high/
            file1.wav
        
        Args:
            data_dir: Path to data directory
        """
        data_path = Path(data_dir)
        
        audio_files = []
        labels = []
        
        # Map directory names to labels
        label_map = {
            'low': 0,
            'medium': 1,
            'high': 2
        }
        
        for stress_level, label_idx in label_map.items():
            level_dir = data_path / stress_level
            
            if not level_dir.exists():
                logger.warning(f"Directory not found: {level_dir}")
                continue
            
            # Get all audio files
            for ext in ['*.wav', '*.mp3', '*.flac']:
                for audio_file in level_dir.glob(ext):
                    audio_files.append(str(audio_file))
                    labels.append(label_idx)
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Extract features
        X, y = self.extract_features_from_files(audio_files, labels)
        
        # Split data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
        )
        
        # Fit scaler on training data
        self.scaler.fit(self.X_train)
        
        # Scale all sets
        self.X_train = self.scaler.transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(
            f"Data loaded and split: "
            f"Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}"
        )
    
    def load_custom_data(self, X, y):
        """
        Load custom feature data
        
        Args:
            X: Feature array
            y: Label array
        """
        # Split data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
        )
        
        # Fit and transform
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Custom data loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
    
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """
        Train Random Forest classifier
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        
        Returns:
            Trained model
        """
        logger.info(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Validation accuracy
        val_acc = self.model.score(self.X_val, self.y_val)
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        return self.model
    
    def train_svm(self, kernel='rbf', C=1.0):
        """
        Train SVM classifier
        
        Args:
            kernel: Kernel type
            C: Regularization parameter
        
        Returns:
            Trained model
        """
        logger.info(f"Training SVM (kernel={kernel}, C={C})")
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,  # Enable probability estimates
            random_state=42,
            verbose=True
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Validation accuracy
        val_acc = self.model.score(self.X_val, self.y_val)
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        return self.model
    
    def cross_validate(self, cv=5):
        """
        Perform cross-validation
        
        Args:
            cv: Number of folds
        
        Returns:
            Cross-validation scores
        """
        if self.model is None:
            raise ValueError("No model trained")
        
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Combine train and val for CV
        X_cv = np.vstack([self.X_train, self.X_val])
        y_cv = np.concatenate([self.y_train, self.y_val])
        
        scores = cross_val_score(self.model, X_cv, y_cv, cv=cv, scoring='accuracy')
        
        logger.info(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
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
        y_pred = self.model.predict(self.X_test)
        y_pred_probs = self.model.predict_proba(self.X_test)
        
        # Metrics
        test_acc = accuracy_score(self.y_test, y_pred)
        
        # Classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.stress_levels,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Feature importance (if Random Forest)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        
        results = {
            'test_accuracy': test_acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_probs,
            'true_labels': self.y_test,
            'feature_importance': feature_importance
        }
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.stress_levels,
            yticklabels=self.stress_levels
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Speech Stress Recognition')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance, top_n=20, save_path=None):
        """
        Plot feature importance (for Random Forest)
        
        Args:
            feature_importance: Feature importance array
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        if feature_importance is None:
            logger.warning("No feature importance available")
            return
        
        # Get top features
        indices = np.argsort(feature_importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), feature_importance[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xticks(range(top_n), indices)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path=None):
        """
        Save trained model and scaler
        
        Args:
            path: Save path (uses default if None)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = MODELS_DIR / 'speech_stress_model.pkl'
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': {
                'sample_rate': SPEECH_CONFIG['sample_rate'],
                'n_mfcc': SPEECH_CONFIG['n_mfcc'],
                'stress_levels': self.stress_levels
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")


# Command-line training script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train speech stress recognition model')
    parser.add_argument('--data-dir', type=str, help='Path to audio data directory')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'svm'], help='Model type')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees (RF only)')
    parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SpeechStressTrainer()
    
    # Load data
    if args.data_dir:
        trainer.load_data_from_directory(args.data_dir)
    else:
        # Generate dummy data for demo
        logger.warning("No data provided. Using dummy data for demo.")
        X_dummy = np.random.rand(500, 75)  # Typical feature count
        y_dummy = np.random.randint(0, 3, size=500)
        trainer.load_custom_data(X_dummy, y_dummy)
    
    # Train
    if args.model == 'rf':
        trainer.train_random_forest(n_estimators=args.n_estimators)
    else:
        trainer.train_svm()
    
    # Cross-validate
    cv_scores = trainer.cross_validate(cv=args.cv)
    
    # Evaluate
    results = trainer.evaluate()
    
    # Plot results
    trainer.plot_confusion_matrix(results['confusion_matrix'], save_path='speech_confusion_matrix.png')
    
    if results['feature_importance'] is not None:
        trainer.plot_feature_importance(results['feature_importance'], save_path='feature_importance.png')
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Model Type: {args.model.upper()}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nPer-class Performance:")
    for stress_level in trainer.stress_levels:
        metrics = results['classification_report'][stress_level]
        print(f"  {stress_level:10s}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
