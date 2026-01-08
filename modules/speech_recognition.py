"""
Speech Stress Recognition Module
Handles audio capture, preprocessing, feature extraction, and stress detection
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from config import SPEECH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechStressRecognizer:
    """
    MFCC-based speech stress recognition
    
    Signal chain: Audio → Preprocessing → Feature extraction → ML → Stress level
    
    Why speech is powerful:
    Stress physically alters voice production:
    - Vocal cord tension → pitch changes
    - Breathing irregularity → energy variations
    - Muscle strain → timbre changes
    
    This is biomechanics, not psychology.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the speech stress recognizer
        
        Args:
            model_path: Path to pre-trained model (if None, creates new classifier)
        """
        self.sample_rate = SPEECH_CONFIG['sample_rate']
        self.audio_duration = SPEECH_CONFIG['audio_duration']
        self.n_mfcc = SPEECH_CONFIG['n_mfcc']
        self.n_mels = SPEECH_CONFIG['n_mels']
        self.stress_levels = SPEECH_CONFIG['stress_levels']
        
        # Track whether a trained model is loaded
        self.model_loaded = False
        
        # Load or create model and scaler
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
            self.model_loaded = True
            logger.info(f"✓ Loaded pre-trained speech model from {model_path}")
        else:
            self.model = self._build_model()
            self.scaler = StandardScaler()
            self.model_loaded = False
            logger.warning(f"⚠️  Speech model not found at {model_path}. Running in FALLBACK mode (always returns 0.0)")
            logger.warning("   To enable speech stress detection, train the model: python train_speech_from_ravdess.py")
    
    def _build_model(self):
        """
        Build Random Forest classifier for stress detection
        
        Why Random Forest:
        - Handles non-linear relationships
        - Robust to noise
        - Interpretable (feature importance)
        - Fast inference
        
        Alternative: SVM, or deep learning (CNN/LSTM on spectrograms)
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        return model
    
    def preprocess_audio(self, audio_data, sr=None):
        """
        Step 1: Audio preprocessing - clean before you think
        
        Raw audio is chaos. This creates structure.
        
        Operations:
        - Resample to target sample rate
        - Remove silence
        - Normalize amplitude
        
        Why this matters:
        - Reduces noise
        - Makes ML stable
        - Enables consistent feature extraction
        
        Args:
            audio_data: Raw audio array
            sr: Sample rate (if None, uses config)
            
        Returns:
            Preprocessed audio array
        """
        if sr is None:
            sr = self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sr, 
                target_sr=self.sample_rate
            )
        
        # Remove leading/trailing silence
        audio_trimmed, _ = librosa.effects.trim(
            audio_data, 
            top_db=20
        )
        
        # Normalize amplitude
        if len(audio_trimmed) > 0:
            audio_normalized = librosa.util.normalize(audio_trimmed)
        else:
            audio_normalized = audio_data
        
        # Ensure minimum length
        min_length = int(self.sample_rate * 1.0)  # 1 second minimum
        if len(audio_normalized) < min_length:
            audio_normalized = np.pad(
                audio_normalized, 
                (0, min_length - len(audio_normalized))
            )
        
        return audio_normalized
    
    def extract_mfcc_features(self, audio):
        """
        Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        
        Plain English:
        MFCCs capture how humans perceive sound, not raw frequency.
        
        They represent:
        - Vocal tract shape
        - Phonetic content
        - Speaker characteristics
        
        Stress affects vocal tract tension → MFCCs change
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            MFCC feature array
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels
        )
        
        # Statistical aggregation (mean, std, min, max)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        
        # Concatenate statistics
        mfcc_features = np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])
        
        return mfcc_features
    
    def extract_pitch_features(self, audio):
        """
        Extract pitch (fundamental frequency F0) features
        
        Why pitch matters:
        Stress → higher pitch (vocal cord tension)
        
        Features:
        - Mean pitch
        - Pitch variability
        - Pitch range
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Pitch feature array
        """
        # Extract pitch using librosa's piptrack
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            fmin=75,  # Minimum human voice frequency
            fmax=400  # Maximum typical pitch
        )
        
        # Get pitch values where magnitude is highest
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            pitch_values = [0]
        
        # Statistical features
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        pitch_min = np.min(pitch_values)
        pitch_max = np.max(pitch_values)
        pitch_range = pitch_max - pitch_min
        
        pitch_features = np.array([
            pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range
        ])
        
        return pitch_features
    
    def extract_energy_features(self, audio):
        """
        Extract energy/intensity features
        
        Why energy matters:
        Stress → louder or inconsistent volume
        
        Features:
        - RMS energy (loudness)
        - Energy variability
        - Zero crossing rate (voice quality)
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Energy feature array
        """
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_max = np.max(rms)
        
        # Zero crossing rate (voice quality indicator)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        energy_features = np.array([
            rms_mean, rms_std, rms_max, zcr_mean, zcr_std
        ])
        
        return energy_features
    
    def extract_speech_rate_features(self, audio):
        """
        Extract speech rate features
        
        Why speech rate matters:
        Stress → faster or erratic speech
        
        Features:
        - Tempo estimation
        - Onset strength (syllable rate proxy)
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Speech rate feature array
        """
        # Onset strength (proxy for syllables/phonemes)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        
        # Tempo estimation
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)[0]
        
        # Onset statistics
        onset_mean = np.mean(onset_env)
        onset_std = np.std(onset_env)
        
        speech_rate_features = np.array([tempo, onset_mean, onset_std])
        
        return speech_rate_features
    
    def extract_all_features(self, audio):
        """
        Step 2: Complete feature extraction pipeline
        
        Combines all acoustic features:
        - MFCCs: vocal tract characteristics
        - Pitch: vocal cord tension
        - Energy: loudness and voice quality
        - Speech rate: temporal patterns
        
        These features reflect biomechanical changes caused by stress.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Tuple of (feature vector, feature_dict for logging)
        """
        # Extract all feature groups
        mfcc_features = self.extract_mfcc_features(audio)
        pitch_features = self.extract_pitch_features(audio)
        energy_features = self.extract_energy_features(audio)
        speech_rate_features = self.extract_speech_rate_features(audio)
        
        # Concatenate all features
        all_features = np.concatenate([
            mfcc_features,
            pitch_features,
            energy_features,
            speech_rate_features
        ])
        
        # Create feature dictionary for logging/diagnostics
        feature_dict = {
            'mfcc_mean': float(np.mean(mfcc_features[:self.n_mfcc])),
            'mfcc_std': float(np.mean(mfcc_features[self.n_mfcc:2*self.n_mfcc])),
            'pitch_mean': float(pitch_features[0]),
            'pitch_std': float(pitch_features[1]),
            'pitch_range': float(pitch_features[4]),
            'rms_energy': float(energy_features[0]),
            'rms_std': float(energy_features[1]),
            'zcr_mean': float(energy_features[3]),
            'tempo': float(speech_rate_features[0]),
            'onset_mean': float(speech_rate_features[1]),
            'feature_count': len(all_features)
        }
        
        return all_features, feature_dict
    
    def is_audio_silent(self, audio, energy_threshold=0.01):
        """
        Detect if audio is silence or too quiet
        
        Args:
            audio: Audio array
            energy_threshold: Minimum RMS energy threshold
            
        Returns:
            Boolean indicating if audio is silent
        """
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        
        # Check if RMS energy is below threshold
        is_silent = rms_mean < energy_threshold
        
        if is_silent:
            logger.debug(f"Audio detected as silent: RMS={rms_mean:.6f} < threshold={energy_threshold}")
        
        return is_silent
    
    def predict_stress_level(self, features):
        """
        Step 3: ML inference - predict stress level
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with stress predictions
        """
        # Check if model is trained
        if not hasattr(self.model, 'estimators_'):
            raise RuntimeError(
                "Speech stress model is not trained. "
                "Please run 'python train_speech_model.py' first to train the model."
            )
        
        # Check if scaler is fitted
        if not hasattr(self.scaler, 'mean_'):
            raise RuntimeError(
                "Feature scaler is not fitted. "
                "Please run 'python train_speech_model.py' first to train the model."
            )
        
        # Reshape for sklearn
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        stress_probs = self.model.predict_proba(features_scaled)[0]
        stress_level = self.model.predict(features_scaled)[0]
        
        result = {
            'stress_level': self.stress_levels[stress_level],
            'stress_level_index': int(stress_level),
            'probabilities': {
                level: float(prob)
                for level, prob in zip(self.stress_levels, stress_probs)
            }
        }
        
        return result
    
    def calculate_speech_stress_score(self, stress_result):
        """
        Convert stress level to 0-100 score
        
        Args:
            stress_result: Dictionary from predict_stress_level
            
        Returns:
            Speech stress score (0-100)
        """
        # Weighted sum based on probabilities
        weights = {'Low': 0, 'Medium': 50, 'High': 100}
        score = sum(
            stress_result['probabilities'][level] * weights[level]
            for level in self.stress_levels
        )
        return score
    
    def analyze_audio(self, audio_data, sr=None):
        """
        Complete pipeline: Audio → Preprocess → Features → Predict → Score
        
        Args:
            audio_data: Raw audio array
            sr: Sample rate (if None, uses config)
            
        Returns:
            Dictionary with analysis results:
            {
                'stress_level': str,
                'speech_stress_score': float,
                'probabilities': dict,
                'features_summary': dict,
                'speech_available': bool,
                'speech_model_used': bool
            }
        """
        try:
            # Preprocess audio
            audio_preprocessed = self.preprocess_audio(audio_data, sr)
            
            logger.info(f"Audio preprocessed: {len(audio_preprocessed)} samples, duration={len(audio_preprocessed)/self.sample_rate:.2f}s")
            
            # Check if audio is silent
            is_silent = self.is_audio_silent(audio_preprocessed)
            
            if is_silent:
                logger.warning("⚠️  Audio is silent or too quiet - no speech detected")
                return {
                    'stress_level': 'Unknown',
                    'speech_stress_score': 0.0,
                    'probabilities': {},
                    'features_summary': {'rms_energy': 0.0},
                    'speech_available': False,
                    'speech_model_used': False,
                    'note': 'Audio is silent or too quiet'
                }
            
            # Extract features
            features, feature_dict = self.extract_all_features(audio_preprocessed)
            
            # Log feature values for debugging
            logger.info(f"📊 Speech features extracted:")
            logger.info(f"   RMS Energy: {feature_dict['rms_energy']:.4f}")
            logger.info(f"   Pitch: mean={feature_dict['pitch_mean']:.1f}Hz, std={feature_dict['pitch_std']:.1f}, range={feature_dict['pitch_range']:.1f}")
            logger.info(f"   MFCC: mean={feature_dict['mfcc_mean']:.4f}, std={feature_dict['mfcc_std']:.4f}")
            logger.info(f"   Tempo: {feature_dict['tempo']:.1f} BPM")
            logger.info(f"   Total features: {feature_dict['feature_count']}")
            
            # Check if model is trained
            if not self.model_loaded:
                logger.warning("⚠️  Speech model not trained - returning fallback score of 0.0")
                logger.warning("   Train the model with: python train_speech_from_ravdess.py")
                return {
                    'stress_level': 'Unknown',
                    'speech_stress_score': 0.0,
                    'probabilities': {},
                    'features_summary': feature_dict,
                    'speech_available': True,
                    'speech_model_used': False,
                    'note': 'Model not trained - run train_speech_from_ravdess.py'
                }
            
            # Predict stress
            stress_result = self.predict_stress_level(features)
            
            # Calculate score
            stress_score = self.calculate_speech_stress_score(stress_result)
            
            logger.info(f"✓ Speech stress prediction: level={stress_result['stress_level']}, score={stress_score:.1f}")
            logger.info(f"   Probabilities: {stress_result['probabilities']}")
            
            result = {
                'stress_level': stress_result['stress_level'],
                'speech_stress_score': stress_score,
                'probabilities': stress_result['probabilities'],
                'features_summary': feature_dict,
                'speech_available': True,
                'speech_model_used': True,
                'audio_duration': len(audio_preprocessed) / self.sample_rate
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'stress_level': 'Unknown',
                'speech_stress_score': 0.0,
                'probabilities': {},
                'features_summary': {},
                'speech_available': False,
                'speech_model_used': False,
                'error': str(e)
            }
    
    def analyze_audio_file(self, file_path):
        """
        Analyze audio from file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Analysis results dictionary
        """
        try:
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            return self.analyze_audio(audio_data, sr)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {'error': str(e)}
    
    def save_model(self, path):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': {
                'sample_rate': self.sample_rate,
                'n_mfcc': self.n_mfcc,
                'stress_levels': self.stress_levels
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {path}")
    
    def _load_model(self, path):
        """Load pre-trained model and scaler"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        logger.info(f"Model loaded from {path}")


# Standalone testing
if __name__ == "__main__":
    recognizer = SpeechStressRecognizer()
    print(f"Model created with {len(recognizer.stress_levels)} stress levels")
    print(f"Sample rate: {recognizer.sample_rate} Hz")
    print(f"MFCC count: {recognizer.n_mfcc}")
