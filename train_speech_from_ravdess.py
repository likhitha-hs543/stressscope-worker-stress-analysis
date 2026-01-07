"""
Train Speech Stress Model from RAVDESS Dataset
Uses the prepared/organized RAVDESS audio files
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import librosa
import pickle
import logging

from modules.speech_recognition import SpeechStressRecognizer
from config import SPEECH_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_ravdess_data(data_dir):
    """Load and extract features from organized RAVDESS data"""
    
    recognizer = SpeechStressRecognizer()
    data_path = Path(data_dir)
    
    X = []
    y = []
    
    label_map = {'low': 0, 'medium': 1, 'high': 2}
    
    for stress_level, label in label_map.items():
        level_dir = data_path / stress_level
        
        if not level_dir.exists():
            logger.warning(f"Directory not found: {level_dir}")
            continue
        
        wav_files = list(level_dir.glob('*.wav'))
        logger.info(f"Processing {len(wav_files)} files from '{stress_level}' category...")
        
        for i, wav_file in enumerate(wav_files):
            try:
                # Load audio
                audio, sr = librosa.load(str(wav_file), sr=SPEECH_CONFIG['sample_rate'])
                
                # Preprocess
                audio = recognizer.preprocess_audio(audio, sr)
                
                # Extract features
                features = recognizer.extract_all_features(audio)
                
                X.append(features)
                y.append(label)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(wav_files)}")
                    
            except Exception as e:
                logger.warning(f"Failed to process {wav_file.name}: {e}")
        
        logger.info(f"✅ Loaded {len([l for l in y if l == label])} samples for '{stress_level}'")
    
    return np.array(X), np.array(y)


def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Train ensemble model for better accuracy"""
    
    logger.info("Building ensemble model...")
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
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
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    logger.info(f"Class weights: {dict(enumerate(class_weights))}")
    
    # Train
    logger.info("Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Validate
    val_acc = ensemble.score(X_val, y_val)
    logger.info(f"✅ Validation accuracy: {val_acc:.4f}")
    
    return ensemble


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    stress_levels = ['Low', 'Medium', 'High']
    report = classification_report(y_test, y_pred, target_names=stress_levels)
    
    print("\n" + "="*60)
    print("SPEECH STRESS RECOGNITION - EVALUATION")
    print("="*60)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("="*60 + "\n")
    
    return test_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train speech stress model from RAVDESS')
    parser.add_argument('--data-dir', type=str,
                       default='data/speech/SER/prepared',
                       help='Directory with organized audio files')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SPEECH STRESS RECOGNITION - TRAINING")
    print("="*60 + "\n")
    
    # Load data
    logger.info("Loading and extracting features...")
    X, y = load_ravdess_data(args.data_dir)
    logger.info(f"Total samples: {len(X)}, Feature dimension: {X.shape[1]}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Train
    model = train_ensemble_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_acc = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_path = MODELS_DIR / 'speech_stress_model.pkl'
    model_data = {
        'model': model,
        'scaler': scaler,
        'config': {
            'sample_rate': SPEECH_CONFIG['sample_rate'],
            'n_mfcc': SPEECH_CONFIG['n_mfcc'],
            'stress_levels': SPEECH_CONFIG['stress_levels']
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"✅ Model saved to {save_path}")
    
    print(f"\n🎉 Training complete! Accuracy: {test_acc:.2%}")
