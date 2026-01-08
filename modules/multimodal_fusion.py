"""
Multimodal Fusion Engine
Combines facial and speech signals for robust stress assessment

Logic: Speech reflects internal stress better (60% weight)
       Facial emotion reflects external expression (40% weight)
       
This approach reduces single-modality noise and improves reliability.
"""

import numpy as np
from collections import deque
import logging
from config import FUSION_CONFIG, STRESS_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusionEngine:
    """
    Combines face and speech signals using weighted fusion
    
    Why multimodal fusion works:
    - Independent signal sources reduce false positives
    - Speech is harder to consciously control (internal state)
    - Face shows external expression (can be masked)
    - Together = more reliable than either alone
    
    Example scenarios:
    - Calm face + stressed voice → internal stress (trust voice)
    - Stressed face + calm voice → momentary expression (balance both)
    """
    
    def __init__(self):
        """
        Initialize fusion engine with weights and smoothing
        """
        self.speech_weight = FUSION_CONFIG['speech_weight']
        self.facial_weight = FUSION_CONFIG['facial_weight']
        self.smoothing_window = FUSION_CONFIG['smoothing_window']
        
        # Verify weights sum to 1.0
        assert abs(self.speech_weight + self.facial_weight - 1.0) < 0.01, \
            "Weights must sum to 1.0"
        
        # Smoothing buffer for temporal consistency
        self.score_history = deque(maxlen=self.smoothing_window)
        
        logger.info(
            f"Fusion engine initialized: "
            f"Speech={self.speech_weight*100}%, "
            f"Facial={self.facial_weight*100}%"
        )
    
    def fuse_scores(self, facial_score, speech_score):
        """
        Simple weighted fusion of facial and speech stress scores
        
        Formula:
        Final Score = (Speech Weight × Speech Score) + (Facial Weight × Facial Score)
        
        Args:
            facial_score: Facial stress score (0-100)
            speech_score: Speech stress score (0-100)
            
        Returns:
            Fused stress score (0-100)
        """
        # Validate inputs
        facial_score = max(0, min(100, facial_score))
        speech_score = max(0, min(100, speech_score))
        
        # Weighted combination
        fused_score = (
            self.speech_weight * speech_score +
            self.facial_weight * facial_score
        )
        
        return fused_score
    
    def smooth_score(self, current_score):
        """
        Temporal smoothing using moving average
        
        Why smoothing:
        - Reduces frame-to-frame jitter
        - Filters outliers
        - Provides stable trends
        
        Method: Simple moving average over last N readings
        
        Args:
            current_score: Latest fused score
            
        Returns:
            Smoothed score
        """
        self.score_history.append(current_score)
        
        if len(self.score_history) == 0:
            return current_score
        
        smoothed_score = np.mean(self.score_history)
        return smoothed_score
    
    def categorize_stress(self, stress_score):
        """
        Map stress score to category (Low/Medium/High)
        
        Thresholds:
        - Low: 0-33
        - Medium: 33-66
        - High: 66-100
        
        Args:
            stress_score: Stress score (0-100)
            
        Returns:
            Stress category string
        """
        for category, (low, high) in STRESS_THRESHOLDS.items():
            if low <= stress_score < high:
                return category.capitalize()
        
        # Edge case: exactly 100
        if stress_score >= STRESS_THRESHOLDS['high'][0]:
            return 'High'
        
        return 'Low'
    
    def calculate_confidence(self, facial_result, speech_result):
        """
        Calculate confidence in the fusion result
        
        Factors:
        - Agreement between modalities (higher = more confident)
        - Individual confidence scores
        - Data availability (both modalities present)
        
        Args:
            facial_result: Dictionary from facial analysis
            speech_result: Dictionary from speech analysis
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from individual modalities
        facial_confidence = facial_result.get('confidence', 0.0)
        speech_confidence = max(
            speech_result.get('probabilities', {}).values(), 
            default=0.0
        )
        
        # Weighted average
        base_confidence = (
            self.facial_weight * facial_confidence +
            self.speech_weight * speech_confidence
        )
        
        # Agreement bonus: if both modalities agree on stress level
        facial_score = facial_result.get('facial_stress_score', 0)
        speech_score = speech_result.get('speech_stress_score', 0)
        
        # Calculate agreement (inverse of normalized difference)
        difference = abs(facial_score - speech_score) / 100.0
        agreement = 1.0 - difference
        
        # Final confidence combines base confidence and agreement
        final_confidence = 0.7 * base_confidence + 0.3 * agreement
        
        return min(1.0, final_confidence)
    
    def analyze_multimodal(self, facial_result, speech_result, apply_smoothing=True):
        """
        Complete multimodal fusion pipeline
        
        Steps:
        1. Extract scores from both modalities
        2. Fuse using weighted combination
        3. Apply temporal smoothing
        4. Categorize stress level
        5. Calculate confidence
        
        Args:
            facial_result: Dictionary from FacialEmotionRecognizer
            speech_result: Dictionary from SpeechStressRecognizer
            apply_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Comprehensive analysis result:
            {
                'facial_score': float,
                'speech_score': float,
                'fused_score': float,
                'smoothed_score': float,
                'stress_category': str,
                'confidence': float,
                'facial_emotion': str,
                'speech_stress_level': str,
                'system_state': str  # 'NORMAL' or 'DEGRADED'
            }
        """
        # Extract scores
        facial_score = facial_result.get('facial_stress_score', 0.0)
        speech_score = speech_result.get('speech_stress_score', 0.0)
        
        # Check if both modalities available
        has_facial = facial_result.get('face_detected', False)
        has_speech = speech_result.get('speech_available', False)
        
        # Determine system state
        if not has_facial and not has_speech:
            system_state = 'DEGRADED'
            logger.warning("🚨 SYSTEM DEGRADED: Both facial and speech modalities are invalid!")
        else:
            system_state = 'NORMAL'
        
        # Fuse scores (will be 0 if both invalid)
        fused_score = self.fuse_scores(facial_score, speech_score)
        
        # Apply smoothing
        if apply_smoothing:
            smoothed_score = self.smooth_score(fused_score)
        else:
            smoothed_score = fused_score
        
        # Categorize
        stress_category = self.categorize_stress(smoothed_score)
        
        # Calculate confidence
        confidence = self.calculate_confidence(facial_result, speech_result)
        
        # Build result
        result = {
            'facial_score': facial_score,
            'speech_score': speech_score,
            'fused_score': fused_score,
            'smoothed_score': smoothed_score,
            'stress_category': stress_category,
            'confidence': confidence,
            'has_facial': has_facial,
            'has_speech': has_speech,
            'system_state': system_state,
            'facial_emotion': facial_result.get('dominant_emotion', 'Unknown'),
            'speech_stress_level': speech_result.get('stress_level', 'Unknown'),
            'timestamp': None  # Will be set by calling code
        }
        
        return result
    
    def reset_smoothing(self):
        """
        Reset smoothing buffer (e.g., start of new session)
        """
        self.score_history.clear()
        logger.debug("Smoothing buffer reset")
    
    def get_trend(self, lookback=5):
        """
        Calculate stress trend (increasing/decreasing/stable)
        
        Args:
            lookback: Number of recent scores to analyze
            
        Returns:
            Trend string: 'increasing', 'decreasing', 'stable'
        """
        if len(self.score_history) < 2:
            return 'stable'
        
        recent_scores = list(self.score_history)[-lookback:]
        
        if len(recent_scores) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Threshold for significant change
        threshold = 2.0  # points per reading
        
        if slope > threshold:
            return 'increasing'
        elif slope < -threshold:
            return 'decreasing'
        else:
            return 'stable'


# Helper functions for external use

def create_fusion_report(result):
    """
    Create human-readable report from fusion result
    
    Args:
        result: Dictionary from analyze_multimodal
        
    Returns:
        Formatted report string
    """
    report = f"""
Multimodal Stress Analysis Report
=================================
Facial Stress Score:     {result['facial_score']:.1f}/100
Speech Stress Score:     {result['speech_score']:.1f}/100
-----------------------------------
Fused Score:            {result['fused_score']:.1f}/100
Smoothed Score:         {result['smoothed_score']:.1f}/100
Stress Category:        {result['stress_category']}
Confidence:             {result['confidence']:.1%}

Details:
- Facial Emotion:       {result['facial_emotion']}
- Speech Stress Level:  {result['speech_stress_level']}
- Modality Status:      Face={'✓' if result['has_facial'] else '✗'}, Speech={'✓' if result['has_speech'] else '✗'}
"""
    return report


# Standalone testing
if __name__ == "__main__":
    engine = MultimodalFusionEngine()
    
    # Test case 1: High stress (both modalities agree)
    facial = {'facial_stress_score': 75, 'face_detected': True, 'confidence': 0.85}
    speech = {'speech_stress_score': 80, 'stress_level': 'High', 'probabilities': {'High': 0.8}}
    
    result = engine.analyze_multimodal(facial, speech)
    print("Test Case 1: High stress (agreement)")
    print(create_fusion_report(result))
    
    # Test case 2: Internal stress (face calm, voice stressed)
    facial = {'facial_stress_score': 20, 'face_detected': True, 'confidence': 0.9}
    speech = {'speech_stress_score': 70, 'stress_level': 'High', 'probabilities': {'High': 0.7}}
    
    result = engine.analyze_multimodal(facial, speech)
    print("\nTest Case 2: Internal stress (voice dominant)")
    print(create_fusion_report(result))
