"""
Real-time Processing Pipeline
Handles live webcam and microphone capture with 2-5 second latency

This module provides:
- Webcam frame capture
- Microphone audio capture  
- Asynchronous processing
- Real-time analysis coordination
"""

import cv2
import numpy as np
import pyaudio
import wave
import threading
import queue
import time
import logging
from datetime import datetime

from modules.facial_recognition import FacialEmotionRecognizer
from modules.speech_recognition import SpeechStressRecognizer
from modules.multimodal_fusion import MultimodalFusionEngine
from modules.rules_engine import StressRulesEngine
from config import REALTIME_CONFIG, SPEECH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeProcessor:
    """
    Real-time multimodal stress analysis pipeline
    
    Architecture:
    - Separate threads for video and audio capture
    - Queue-based processing to prevent blocking
    - Periodic analysis (every 2-5 seconds)
    - Fusion of both modalities
    
    Latency target: 2-5 seconds (academically acceptable)
    """
    
    def __init__(self, callback=None):
        """
        Initialize real-time processor
        
        Args:
            callback: Function to call with analysis results
        """
        # Initialize AI modules
        self.facial_recognizer = FacialEmotionRecognizer()
        self.speech_recognizer = SpeechStressRecognizer()
        self.fusion_engine = MultimodalFusionEngine()
        self.rules_engine = StressRulesEngine()
        
        # Processing queues
        self.video_queue = queue.Queue(maxsize=REALTIME_CONFIG['max_queue_size'])
        self.audio_queue = queue.Queue(maxsize=REALTIME_CONFIG['max_queue_size'])
        
        # Control flags
        self.is_running = False
        self.session_id = None
        
        # Callback for results
        self.callback = callback
        
        # Threads
        self.video_thread = None
        self.audio_thread = None
        self.processing_thread = None
        
        # Performance metrics
        self.frame_count = 0
        self.analysis_count = 0
        self.start_time = None
        
        logger.info("Real-time processor initialized")
    
    def start(self, session_id, enable_video=True, enable_audio=True):
        """
        Start real-time processing
        
        Args:
            session_id: Unique session identifier
            enable_video: Enable webcam capture
            enable_audio: Enable microphone capture
        """
        if self.is_running:
            logger.warning("Processor already running")
            return
        
        self.is_running = True
        self.session_id = session_id
        self.start_time = time.time()
        
        logger.info(f"Starting real-time processing (session: {session_id})")
        
        # Start capture threads
        if enable_video:
            self.video_thread = threading.Thread(
                target=self._video_capture_loop,
                daemon=True
            )
            self.video_thread.start()
            logger.info("Video capture started")
        
        if enable_audio:
            self.audio_thread = threading.Thread(
                target=self._audio_capture_loop,
                daemon=True
            )
            self.audio_thread.start()
            logger.info("Audio capture started")
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("Processing loop started")
    
    def stop(self):
        """Stop real-time processing"""
        logger.info("Stopping real-time processing")
        self.is_running = False
        
        # Wait for threads to finish
        if self.video_thread:
            self.video_thread.join(timeout=2)
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        # Print performance stats
        if self.start_time:
            duration = time.time() - self.start_time
            fps = self.frame_count / duration if duration > 0 else 0
            aps = self.analysis_count / duration if duration > 0 else 0
            
            logger.info(
                f"Session ended. Duration: {duration:.1f}s, "
                f"Frames: {self.frame_count} ({fps:.1f} fps), "
                f"Analyses: {self.analysis_count} ({aps:.2f}/s)"
            )
        
        logger.info("Real-time processing stopped")
    
    def _video_capture_loop(self):
        """
        Video capture thread
        
        Captures frames from webcam at regular intervals
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        logger.info("Webcam opened successfully")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Add to queue (non-blocking)
                try:
                    self.video_queue.put_nowait({
                        'frame': frame,
                        'timestamp': datetime.utcnow()
                    })
                    self.frame_count += 1
                except queue.Full:
                    # Queue full - skip frame
                    pass
                
                # Limit frame rate (e.g., 10 fps for processing)
                time.sleep(0.1)
        
        finally:
            cap.release()
            logger.info("Webcam released")
    
    def _audio_capture_loop(self):
        """
        Audio capture thread
        
        Captures audio from microphone in chunks
        """
        CHUNK = REALTIME_CONFIG['buffer_size']
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = SPEECH_CONFIG['sample_rate']
        RECORD_SECONDS = SPEECH_CONFIG['audio_duration']
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            logger.info("Microphone opened successfully")
            
            while self.is_running:
                # Record audio chunk
                frames = []
                for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    if not self.is_running:
                        break
                    data = stream.read(CHUNK)
                    frames.append(data)
                
                # Convert to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
                
                # Add to queue (non-blocking)
                try:
                    self.audio_queue.put_nowait({
                        'audio': audio_data,
                        'timestamp': datetime.utcnow()
                    })
                except queue.Full:
                    pass
        
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("Microphone released")
    
    def _processing_loop(self):
        """
        Main processing thread
        
        Periodically analyzes captured data from both modalities
        """
        last_analysis_time = time.time()
        analysis_interval = REALTIME_CONFIG['processing_interval']
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time for analysis
            if current_time - last_analysis_time < analysis_interval:
                time.sleep(0.1)
                continue
            
            # Get latest data from queues
            video_data = None
            audio_data = None
            
            # Get most recent video frame
            while not self.video_queue.empty():
                try:
                    video_data = self.video_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Get most recent audio chunk
            while not self.audio_queue.empty():
                try:
                    audio_data = self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Perform analysis
            if video_data or audio_data:
                self._analyze_multimodal(video_data, audio_data)
                self.analysis_count += 1
                last_analysis_time = current_time
    
    def _analyze_multimodal(self, video_data, audio_data):
        """
        Perform multimodal analysis
        
        Steps:
        1. Analyze face (if available)
        2. Analyze speech (if available)
        3. Fuse results
        4. Apply business rules
        5. Generate recommendations
        6. Call callback
        
        Args:
            video_data: Dictionary with frame and timestamp
            audio_data: Dictionary with audio and timestamp
        """
        try:
            # Analyze facial emotion
            facial_result = {'facial_stress_score': 0, 'face_detected': False}
            if video_data:
                facial_result = self.facial_recognizer.analyze_frame(
                    video_data['frame']
                )
            
            # Analyze speech stress
            speech_result = {'speech_stress_score': 0}
            if audio_data:
                speech_result = self.speech_recognizer.analyze_audio(
                    audio_data['audio']
                )
            
            # Multimodal fusion
            fusion_result = self.fusion_engine.analyze_multimodal(
                facial_result,
                speech_result
            )
            
            # Add timestamp
            fusion_result['timestamp'] = datetime.utcnow().isoformat()
            
            # Apply business rules
            alert_result = self.rules_engine.should_trigger_alert(
                self.session_id,
                fusion_result['smoothed_score']
            )
            
            # Get recommendation
            recommendation = self.rules_engine.get_recommendation(
                fusion_result['smoothed_score'],
                fusion_result['stress_category']
            )
            
            # Build complete result
            result = {
                **fusion_result,
                'alert': alert_result,
                'recommendation': recommendation,
                'session_id': self.session_id
            }
            
            # Log result
            logger.info(
                f"Analysis: Score={fusion_result['smoothed_score']:.1f}, "
                f"Category={fusion_result['stress_category']}, "
                f"Alert={alert_result['should_alert']}"
            )
            
            # Call callback if provided
            if self.callback:
                self.callback(result)
        
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.start_time:
            return {}
        
        duration = time.time() - self.start_time
        
        return {
            'session_id': self.session_id,
            'duration_seconds': duration,
            'frames_captured': self.frame_count,
            'analyses_performed': self.analysis_count,
            'fps': self.frame_count / duration if duration > 0 else 0,
            'analyses_per_second': self.analysis_count / duration if duration > 0 else 0,
            'is_running': self.is_running
        }


# Command-line interface for testing
if __name__ == "__main__":
    def result_callback(result):
        """Print results to console"""
        print(f"\n{'='*60}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Facial Score: {result['facial_score']:.1f}")
        print(f"Speech Score: {result['speech_score']:.1f}")
        print(f"Fused Score: {result['fused_score']:.1f}")
        print(f"Smoothed Score: {result['smoothed_score']:.1f}")
        print(f"Category: {result['stress_category']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Recommendation: {result['recommendation']}")
        if result['alert']['should_alert']:
            print(f"⚠️  ALERT: {result['alert']['reason']}")
        print(f"{'='*60}")
    
    # Create processor
    processor = RealtimeProcessor(callback=result_callback)
    
    # Start processing
    session_id = f"test_session_{int(time.time())}"
    processor.start(session_id, enable_video=True, enable_audio=False)
    
    # Run for 60 seconds
    try:
        print("Running real-time analysis for 60 seconds...")
        print("Press Ctrl+C to stop early")
        time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Stop processing
    processor.stop()
    
    # Print stats
    stats = processor.get_stats()
    print("\nSession Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
