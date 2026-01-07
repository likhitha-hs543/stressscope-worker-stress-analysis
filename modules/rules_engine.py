"""
Business Rules Engine
Implements stress threshold detection, alerting, and decision logic

Remember: AI predicts. Rules decide.
"""

import logging
from datetime import datetime, timedelta
from collections import deque
from config import STRESS_THRESHOLDS, RULES_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressRulesEngine:
    """
    Business logic for stress monitoring
    
    Responsibilities:
    - Detect threshold violations
    - Track stress duration
    - Generate alerts
    - Provide recommendations
    
    Not AI - pure business logic.
    """
    
    def __init__(self):
        """Initialize rules engine with configuration"""
        self.high_stress_threshold = STRESS_THRESHOLDS['high'][0]
        self.medium_stress_threshold = STRESS_THRESHOLDS['medium'][0]
        
        self.high_stress_duration_limit = RULES_CONFIG['high_stress_duration_threshold']
        self.alert_cooldown = RULES_CONFIG['alert_cooldown']
        self.session_timeout = RULES_CONFIG['session_timeout']
        
        # State tracking per session
        self.session_states = {}
        
        logger.info("Rules engine initialized")
    
    def _get_session_state(self, session_id):
        """
        Get or create session state
        
        State includes:
        - High stress start time
        - Last alert time
        - Stress history
        """
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                'high_stress_start': None,
                'high_stress_duration': 0,
                'last_alert_time': None,
                'alert_count': 0,
                'stress_history': deque(maxlen=100),
                'session_start': datetime.utcnow()
            }
        return self.session_states[session_id]
    
    def check_high_stress_duration(self, session_id, stress_score, timestamp=None):
        """
        Rule: If stress > threshold for X minutes → flag
        
        Tracks continuous high stress duration
        
        Args:
            session_id: Session identifier
            stress_score: Current stress score
            timestamp: Current timestamp (uses now if None)
            
        Returns:
            Dictionary with duration info and alert flag
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        state = self._get_session_state(session_id)
        
        # Check if currently in high stress
        is_high_stress = stress_score >= self.high_stress_threshold
        
        if is_high_stress:
            # Start tracking if not already
            if state['high_stress_start'] is None:
                state['high_stress_start'] = timestamp
                logger.info(f"Session {session_id}: High stress period started")
            
            # Calculate duration
            duration = (timestamp - state['high_stress_start']).total_seconds()
            state['high_stress_duration'] = duration
            
            # Check if exceeds limit
            should_alert = duration >= self.high_stress_duration_limit
            
            return {
                'is_high_stress': True,
                'duration_seconds': duration,
                'exceeds_limit': should_alert,
                'limit_seconds': self.high_stress_duration_limit
            }
        else:
            # Reset if stress dropped
            if state['high_stress_start'] is not None:
                logger.info(
                    f"Session {session_id}: High stress period ended "
                    f"(duration: {state['high_stress_duration']:.0f}s)"
                )
                state['high_stress_start'] = None
                state['high_stress_duration'] = 0
            
            return {
                'is_high_stress': False,
                'duration_seconds': 0,
                'exceeds_limit': False
            }
    
    def check_alert_cooldown(self, session_id, timestamp=None):
        """
        Rule: Don't spam alerts - cooldown period between alerts
        
        Args:
            session_id: Session identifier
            timestamp: Current timestamp
            
        Returns:
            Boolean - can send alert
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        state = self._get_session_state(session_id)
        
        # No previous alert - ok to send
        if state['last_alert_time'] is None:
            return True
        
        # Check cooldown period
        time_since_alert = (timestamp - state['last_alert_time']).total_seconds()
        can_alert = time_since_alert >= self.alert_cooldown
        
        if not can_alert:
            logger.debug(
                f"Session {session_id}: In alert cooldown "
                f"({time_since_alert:.0f}s / {self.alert_cooldown}s)"
            )
        
        return can_alert
    
    def should_trigger_alert(self, session_id, stress_score, timestamp=None):
        """
        Master alert decision function
        
        Combines multiple rules:
        1. High stress duration exceeded
        2. Alert cooldown respected
        
        Args:
            session_id: Session identifier
            stress_score: Current stress score
            timestamp: Current timestamp
            
        Returns:
            Dictionary with alert decision and details
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Check duration rule
        duration_check = self.check_high_stress_duration(
            session_id, stress_score, timestamp
        )
        
        # Check cooldown rule
        can_alert = self.check_alert_cooldown(session_id, timestamp)
        
        # Decision
        should_alert = (
            duration_check['exceeds_limit'] and 
            can_alert
        )
        
        if should_alert:
            state = self._get_session_state(session_id)
            state['last_alert_time'] = timestamp
            state['alert_count'] += 1
            
            logger.warning(
                f"Session {session_id}: Alert triggered! "
                f"(stress={stress_score:.1f}, duration={duration_check['duration_seconds']:.0f}s)"
            )
        
        return {
            'should_alert': should_alert,
            'reason': 'prolonged_high_stress' if should_alert else None,
            'stress_score': stress_score,
            'duration_seconds': duration_check['duration_seconds'],
            'alert_count': self._get_session_state(session_id)['alert_count']
        }
    
    def get_recommendation(self, stress_score, stress_category, duration_info=None):
        """
        Generate friendly recommendations based on stress level
        
        Not diagnosis - just helpful nudges
        
        Args:
            stress_score: Current stress score
            stress_category: Low/Medium/High
            duration_info: Optional duration information
            
        Returns:
            Recommendation string
        """
        recommendations = {
            'Low': [
                "You're doing great! Keep up the good work.",
                "Stress levels are low. Maintain your current routine.",
                "Looking good! Remember to stay hydrated."
            ],
            'Medium': [
                "Consider taking a short break.",
                "Try some deep breathing exercises.",
                "Stretch for a few minutes to relax.",
                "Step away from your screen briefly."
            ],
            'High': [
                "Take a break - you've earned it.",
                "Consider a short walk to clear your mind.",
                "Try the 4-7-8 breathing technique.",
                "Reach out to someone if you need support.",
                "High stress detected. Please take care of yourself."
            ]
        }
        
        # Select appropriate recommendation
        import random
        rec_list = recommendations.get(stress_category, recommendations['Medium'])
        base_rec = random.choice(rec_list)
        
        # Add duration context if available
        if duration_info and duration_info.get('is_high_stress'):
            duration_min = duration_info['duration_seconds'] / 60
            if duration_min >= 10:
                base_rec += f" (High stress for {duration_min:.0f} minutes)"
        
        return base_rec
    
    def analyze_trend(self, stress_history):
        """
        Rule: Repeated high stress over days → risk indicator
        
        Analyzes stress patterns over time
        
        Args:
            stress_history: List of (timestamp, stress_score) tuples
            
        Returns:
            Trend analysis dictionary
        """
        if len(stress_history) < 10:
            return {
                'trend': 'insufficient_data',
                'risk_level': 'unknown'
            }
        
        # Calculate statistics
        recent_scores = [score for _, score in stress_history[-50:]]
        avg_stress = sum(recent_scores) / len(recent_scores)
        
        # Count high stress occurrences
        high_stress_count = sum(1 for score in recent_scores if score >= self.high_stress_threshold)
        high_stress_ratio = high_stress_count / len(recent_scores)
        
        # Determine risk level
        if high_stress_ratio >= 0.5:
            risk_level = 'high'
            trend = 'consistently_high'
        elif high_stress_ratio >= 0.3:
            risk_level = 'medium'
            trend = 'frequently_elevated'
        else:
            risk_level = 'low'
            trend = 'normal'
        
        return {
            'trend': trend,
            'risk_level': risk_level,
            'avg_stress': avg_stress,
            'high_stress_ratio': high_stress_ratio,
            'sample_size': len(recent_scores)
        }
    
    def check_session_timeout(self, session_id, timestamp=None):
        """
        Check if session has timed out (inactive)
        
        Args:
            session_id: Session identifier
            timestamp: Current timestamp
            
        Returns:
            Boolean - session timed out
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        state = self._get_session_state(session_id)
        session_duration = (timestamp - state['session_start']).total_seconds()
        
        return session_duration >= self.session_timeout
    
    def reset_session(self, session_id):
        """
        Reset session state (e.g., user logout)
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.session_states:
            del self.session_states[session_id]
            logger.info(f"Session {session_id} state reset")
    
    def get_session_summary(self, session_id):
        """
        Get summary of session state
        
        Args:
            session_id: Session identifier
            
        Returns:
            Summary dictionary
        """
        if session_id not in self.session_states:
            return {'exists': False}
        
        state = self.session_states[session_id]
        
        return {
            'exists': True,
            'alert_count': state['alert_count'],
            'high_stress_duration': state['high_stress_duration'],
            'is_currently_high_stress': state['high_stress_start'] is not None,
            'session_duration': (datetime.utcnow() - state['session_start']).total_seconds()
        }


# Standalone testing
if __name__ == "__main__":
    engine = StressRulesEngine()
    
    # Test high stress detection
    session_id = "test_session_1"
    
    print("Test 1: High stress for 6 minutes")
    for i in range(7):
        timestamp = datetime.utcnow() + timedelta(minutes=i)
        result = engine.should_trigger_alert(session_id, 75, timestamp)
        print(f"  Minute {i}: Alert={result['should_alert']}, Duration={result['duration_seconds']:.0f}s")
    
    print("\nTest 2: Recommendations")
    for category in ['Low', 'Medium', 'High']:
        rec = engine.get_recommendation(50, category)
        print(f"  {category}: {rec}")
    
    print("\nTest 3: Session summary")
    summary = engine.get_session_summary(session_id)
    print(f"  {summary}")
