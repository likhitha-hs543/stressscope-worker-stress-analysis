"""
System Validation Script
Tests the entire stress detection pipeline end-to-end
"""

import numpy as np
import cv2
from pathlib import Path
import logging

from modules.facial_recognition import FacialEmotionRecognizer
from modules.speech_recognition import SpeechStressRecognizer
from modules.multimodal_fusion import MultimodalFusionEngine
from modules.rules_engine import StressRulesEngine
from config import FACE_MODEL_PATH, SPEECH_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemValidator:
    """Validates entire stress detection system"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def validate_models_exist(self):
        """Check if trained models exist"""
        logger.info("Checking model files...")
        
        facial_exists = FACE_MODEL_PATH.exists()
        speech_exists = SPEECH_MODEL_PATH.exists()
        
        if not facial_exists:
            self.issues.append(f"❌ Facial model not found: {FACE_MODEL_PATH}")
        else:
            logger.info(f"✅ Facial model found: {FACE_MODEL_PATH}")
        
        if not speech_exists:
            self.issues.append(f"❌ Speech model not found: {SPEECH_MODEL_PATH}")
        else:
            logger.info(f"✅ Speech model found: {SPEECH_MODEL_PATH}")
        
        return facial_exists and speech_exists
    
    def validate_model_loading(self):
        """Test model loading"""
        logger.info("Testing model loading...")
        
        try:
            facial = FacialEmotionRecognizer(model_path=str(FACE_MODEL_PATH))
            logger.info("✅ Facial model loaded successfully")
        except Exception as e:
            self.issues.append(f"❌ Failed to load facial model: {e}")
            return False
        
        try:
            speech = SpeechStressRecognizer(model_path=str(SPEECH_MODEL_PATH))
            logger.info("✅ Speech model loaded successfully")
        except Exception as e:
            self.issues.append(f"❌ Failed to load speech model: {e}")
            return False
        
        return True
    
    def validate_facial_prediction(self):
        """Test facial prediction with synthetic data"""
        logger.info("Testing facial prediction...")
        
        try:
            recognizer = FacialEmotionRecognizer(model_path=str(FACE_MODEL_PATH))
            
            # Create synthetic image (random data)
            test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            
            # Predict
            result = recognizer.analyze_frame(test_image)
            
            # Validate output structure
            required_keys = ['face_detected', 'emotion_probs', 'dominant_emotion', 'facial_stress_score']
            for key in required_keys:
                if key not in result:
                    self.issues.append(f"❌ Missing key in facial result: {key}")
                    return False
            
            logger.info(f"✅ Facial prediction works - Score: {result['facial_stress_score']:.1f}")
            return True
            
        except Exception as e:
            self.issues.append(f"❌ Facial prediction failed: {e}")
            return False
    
    def validate_speech_prediction(self):
        """Test speech prediction with synthetic data"""
        logger.info("Testing speech prediction...")
        
        try:
            recognizer = SpeechStressRecognizer(model_path=str(SPEECH_MODEL_PATH))
            
            # Create synthetic audio (1 second)
            test_audio = np.random.randn(16000).astype(np.float32)
            
            # Predict
            result = recognizer.analyze_audio(test_audio, 16000)
            
            # Validate output structure
            required_keys = ['stress_level', 'speech_stress_score', 'probabilities']
            for key in required_keys:
                if key not in result:
                    self.issues.append(f"❌ Missing key in speech result: {key}")
                    return False
            
            logger.info(f"✅ Speech prediction works - Score: {result['speech_stress_score']:.1f}")
            return True
            
        except Exception as e:
            self.issues.append(f"❌ Speech prediction failed: {e}")
            return False
    
    def validate_fusion(self):
        """Test multimodal fusion"""
        logger.info("Testing multimodal fusion...")
        
        try:
            fusion_engine = MultimodalFusionEngine()
            
            # Mock results
            facial_result = {'facial_stress_score': 65, 'face_detected': True, 'confidence': 0.8}
            speech_result = {'speech_stress_score': 70, 'probabilities': {'High': 0.7}}
            
            # Fuse
            result = fusion_engine.analyze_multimodal(facial_result, speech_result)
            
            # Validate
            required_keys = ['fused_score', 'smoothed_score', 'stress_category', 'confidence']
            for key in required_keys:
                if key not in result:
                    self.issues.append(f"❌ Missing key in fusion result: {key}")
                    return False
            
            logger.info(f"✅ Fusion works - Score: {result['fused_score']:.1f}, Category: {result['stress_category']}")
            return True
            
        except Exception as e:
            self.issues.append(f"❌ Fusion failed: {e}")
            return False
    
    def validate_rules_engine(self):
        """Test rules engine"""
        logger.info("Testing rules engine...")
        
        try:
            rules_engine = StressRulesEngine()
            
            # Test alert logic
            alert = rules_engine.should_trigger_alert('test_session', 85)
            
            # Get recommendation
            rec = rules_engine.get_recommendation(85, 'High')
            
            logger.info(f"✅ Rules engine works - Alert: {alert['should_alert']}")
            logger.info(f"   Recommendation: {rec}")
            return True
            
        except Exception as e:
            self.issues.append(f"❌ Rules engine failed: {e}")
            return False
    
    def validate_end_to_end(self):
        """Test complete pipeline"""
        logger.info("Testing end-to-end pipeline...")
        
        try:
            # Load all components
            facial = FacialEmotionRecognizer(model_path=str(FACE_MODEL_PATH))
            speech = SpeechStressRecognizer(model_path=str(SPEECH_MODEL_PATH))
            fusion = MultimodalFusionEngine()
            rules = StressRulesEngine()
            
            # Synthetic inputs
            test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            test_audio = np.random.randn(16000).astype(np.float32)
            
            # Process
            facial_result = facial.analyze_frame(test_image)
            speech_result = speech.analyze_audio(test_audio, 16000)
            fusion_result = fusion.analyze_multimodal(facial_result, speech_result)
            alert_result = rules.should_trigger_alert('test_session', fusion_result['smoothed_score'])
            
            logger.info("✅ End-to-end pipeline successful!")
            logger.info(f"   Final stress score: {fusion_result['smoothed_score']:.1f}/100")
            logger.info(f"   Stress category: {fusion_result['stress_category']}")
            
            return True
            
        except Exception as e:
            self.issues.append(f"❌ End-to-end pipeline failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*60)
        print("WORKER STRESS ANALYSIS SYSTEM - VALIDATION")
        print("="*60 + "\n")
        
        tests = [
            ("Model Files", self.validate_models_exist),
            ("Model Loading", self.validate_model_loading),
            ("Facial Prediction", self.validate_facial_prediction),
            ("Speech Prediction", self.validate_speech_prediction),
            ("Multimodal Fusion", self.validate_fusion),
            ("Rules Engine", self.validate_rules_engine),
            ("End-to-End Pipeline", self.validate_end_to_end),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            print("-" * 60)
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Test crashed: {e}")
                failed += 1
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"\n✅ Passed: {passed}/{len(tests)}")
        print(f"❌ Failed: {failed}/{len(tests)}")
        
        if self.issues:
            print("\n🚨 ISSUES FOUND:")
            for issue in self.issues:
                print(f"  {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! System is ready for use.")
            print("\nNext steps:")
            print("  1. Run: python app.py")
            print("  2. Open: http://localhost:5000")
            print("  3. Start monitoring stress in real-time!")
        else:
            print("\n❌ Some tests failed. Please fix issues before deployment.")
            print("\nCommon fixes:")
            print("  - Train models if missing")
            print("  - Check model file paths in config.py")
            print("  - Ensure all dependencies installed")
        
        print("\n" + "="*60 + "\n")
        
        return failed == 0


if __name__ == "__main__":
    validator = SystemValidator()
    success = validator.run_all_tests()
    exit(0 if success else 1)
