"""
Configuration file for Worker Stress Analysis System
Contains all system parameters, thresholds, and settings
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///stress_analysis.db')

# Model paths
MODELS_DIR = BASE_DIR / 'models' / 'trained'
FACE_MODEL_PATH = MODELS_DIR / 'facial_emotion_model_v2.keras'  # Using v2 (MobileNetV2 transfer learning)
SPEECH_MODEL_PATH = MODELS_DIR / 'speech_stress_model.pkl'

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_VIDEO_DIR = DATA_DIR / 'raw_videos'
RAW_AUDIO_DIR = DATA_DIR / 'raw_audio'
PROCESSED_DIR = DATA_DIR / 'processed'

# Create directories if they don't exist
for directory in [MODELS_DIR, RAW_VIDEO_DIR, RAW_AUDIO_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Facial emotion recognition settings
FACE_CONFIG = {
    'input_shape': (48, 48, 1),  # Grayscale 48x48
    'emotions': ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    'stress_emotions': ['Angry', 'Fear', 'Sad'],  # Emotions mapped to stress
    'confidence_threshold': 0.5,
    'face_cascade': 'haarcascade_frontalface_default.xml',
    'frame_interval': 1.0,  # Capture 1 frame per second
}

# Speech stress recognition settings
SPEECH_CONFIG = {
    'sample_rate': 16000,
    'audio_duration': 15,  # seconds
    'frame_length': 0.025,  # 25ms
    'frame_stride': 0.010,  # 10ms
    'n_mfcc': 13,
    'n_mels': 40,
    'stress_levels': ['Low', 'Medium', 'High'],
    'features': ['mfcc', 'pitch', 'energy', 'speech_rate'],
}

# Multimodal fusion weights
FUSION_CONFIG = {
    'speech_weight': 0.6,  # 60% - harder to consciously control
    'facial_weight': 0.4,  # 40% - external expression
    'smoothing_window': 5,  # Number of readings to smooth
}

# Stress thresholds and categories
STRESS_THRESHOLDS = {
    'low': (0, 33),
    'medium': (33, 66),
    'high': (66, 100),
}

# Business rules
RULES_CONFIG = {
    'high_stress_duration_threshold': 300,  # seconds (5 minutes)
    'alert_cooldown': 1800,  # seconds (30 minutes) between alerts
    'session_timeout': 3600,  # seconds (1 hour)
    'trend_analysis_days': 7,
}

# Real-time processing settings
REALTIME_CONFIG = {
    'target_latency': 5,  # seconds
    'buffer_size': 1024,
    'max_queue_size': 100,
    'processing_interval': 2,  # seconds between analyses
}

# Dashboard settings
DASHBOARD_CONFIG = {
    'refresh_interval': 5000,  # milliseconds
    'chart_history_points': 50,
    'anonymization_threshold': 3,  # Minimum team size for aggregation
}

# Privacy and ethics settings
PRIVACY_CONFIG = {
    'store_raw_video': False,  # Never store raw video
    'store_raw_audio': False,  # Never store raw audio
    'anonymize_admin_data': True,
    'data_retention_days': 90,
    'require_consent': True,
}

# Flask settings
FLASK_CONFIG = {
    'host': '127.0.0.1',
    'port': 5000,
    'debug': False,
    'secret_key': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
}

# API endpoints
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'
