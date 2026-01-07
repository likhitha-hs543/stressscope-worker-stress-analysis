"""
Initialize modules package
"""

__version__ = "1.0.0"
__author__ = "Worker Stress Analysis Team"

# Make key classes easily importable
from modules.facial_recognition import FacialEmotionRecognizer
from modules.speech_recognition import SpeechStressRecognizer
from modules.multimodal_fusion import MultimodalFusionEngine
from modules.rules_engine import StressRulesEngine

__all__ = [
    'FacialEmotionRecognizer',
    'SpeechStressRecognizer',
    'MultimodalFusionEngine',
    'StressRulesEngine',
]
