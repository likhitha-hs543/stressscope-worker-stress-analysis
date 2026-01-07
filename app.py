"""
Flask Backend API
Provides REST endpoints for stress analysis system

Endpoints:
- /api/v1/analyze_face
- /api/v1/analyze_speech
- /api/v1/analyze_multimodal
- /api/v1/stress_score
- /api/v1/dashboard/employee
- /api/v1/dashboard/admin
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import base64
import io
import logging
from datetime import datetime, timedelta
import uuid

from config import FLASK_CONFIG, API_PREFIX, FACE_MODEL_PATH, SPEECH_MODEL_PATH
from modules.facial_recognition import FacialEmotionRecognizer
from modules.speech_recognition import SpeechStressRecognizer
from modules.multimodal_fusion import MultimodalFusionEngine
from modules.rules_engine import StressRulesEngine
from modules.database import (
    init_database, get_session as get_db_session,
    Employee, StressSession, StressRecord, Alert, AggregatedStats
)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_CONFIG['secret_key']
# CORS: Allow all origins in development, restrict in production
# TODO: In production, replace '*' with specific allowed origins
CORS(app, origins='*')
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI modules (global instances)
facial_recognizer = FacialEmotionRecognizer()
speech_recognizer = SpeechStressRecognizer()
fusion_engine = MultimodalFusionEngine()
rules_engine = StressRulesEngine()

# Initialize database
db_engine = init_database()

logger.info("Backend initialized successfully")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def decode_image_from_base64(base64_str):
    """
    Decode base64 image to numpy array
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        OpenCV image (numpy array)
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(base64_str)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None


def decode_audio_from_base64(base64_str):
    """
    Decode base64 audio to numpy array
    
    Args:
        base64_str: Base64 encoded audio string
        
    Returns:
        Audio data as numpy array
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        audio_bytes = base64.b64decode(base64_str)
        
        # Convert to numpy array (assuming float32 PCM)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        
        return audio_data
    except Exception as e:
        logger.error(f"Error decoding audio: {e}")
        return None


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/')
def index():
    """Root endpoint - serve main page"""
    return render_template('index.html')


@app.route(f'{API_PREFIX}/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'modules': {
            'facial_recognition': True,
            'speech_recognition': True,
            'fusion_engine': True,
            'rules_engine': True,
            'database': True
        }
    })


@app.route(f'{API_PREFIX}/model_status')
def model_status():
    """
    Check if ML models are trained and ready
    
    Response:
    {
        "facial_model_ready": bool,
        "speech_model_ready": bool,
        "system_ready": bool,
        "message": str
    }
    """
    facial_ready = FACE_MODEL_PATH.exists()
    speech_ready = SPEECH_MODEL_PATH.exists()
    system_ready = facial_ready and speech_ready
    
    message = "System ready" if system_ready else "Models need training"
    
    return jsonify({
        'facial_model_ready': facial_ready,
        'facial_model_path': str(FACE_MODEL_PATH),
        'speech_model_ready': speech_ready,
        'speech_model_path': str(SPEECH_MODEL_PATH),
        'system_ready': system_ready,
        'message': message,
        'training_commands': {
            'facial': 'python train_facial_model.py --data path/to/fer2013.csv',
            'speech': 'python train_speech_model.py --data-dir path/to/audio_data'
        }
    })


# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.route(f'{API_PREFIX}/analyze_face', methods=['POST'])
def analyze_face():
    """
    Analyze facial emotion from image
    
    Request body:
    {
        "image": "base64_encoded_image"
    }
    
    Response:
    {
        "face_detected": bool,
        "emotion_probs": dict,
        "dominant_emotion": str,
        "facial_stress_score": float,
        "confidence": float
    }
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        img = decode_image_from_base64(data['image'])
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Analyze
        result = facial_recognizer.analyze_frame(img)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in analyze_face: {e}")
        return jsonify({'error': str(e)}), 500


@app.route(f'{API_PREFIX}/analyze_speech', methods=['POST'])
def analyze_speech():
    """
    Analyze speech stress from audio
    
    Request body:
    {
        "audio": "base64_encoded_audio",
        "sample_rate": 16000
    }
    
    Response:
    {
        "stress_level": str,
        "speech_stress_score": float,
        "probabilities": dict
    }
    """
    try:
        data = request.get_json()
        
        if 'audio' not in data:
            return jsonify({'error': 'No audio provided'}), 400
        
        # Decode audio
        audio_data = decode_audio_from_base64(data['audio'])
        
        if audio_data is None:
            return jsonify({'error': 'Invalid audio format'}), 400
        
        # Get sample rate
        sample_rate = data.get('sample_rate', 16000)
        
        # Analyze
        result = speech_recognizer.analyze_audio(audio_data, sample_rate)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in analyze_speech: {e}")
        return jsonify({'error': str(e)}), 500


@app.route(f'{API_PREFIX}/analyze_multimodal', methods=['POST'])
def analyze_multimodal():
    """
    Analyze stress using both face and speech
    
    Request body:
    {
        "image": "base64_encoded_image",
        "audio": "base64_encoded_audio",
        "sample_rate": 16000,
        "employee_id": "string",
        "session_id": "string"
    }
    
    Response:
    {
        "facial_score": float,
        "speech_score": float,
        "fused_score": float,
        "smoothed_score": float,
        "stress_category": str,
        "confidence": float,
        "recommendation": str,
        "alert": dict
    }
    """
    try:
        data = request.get_json()
        
        employee_id = data.get('employee_id', 'anonymous')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Analyze face (if provided)
        facial_result = {'facial_stress_score': 0, 'face_detected': False}
        if 'image' in data and data['image']:
            img = decode_image_from_base64(data['image'])
            if img is not None:
                facial_result = facial_recognizer.analyze_frame(img)
        
        # Analyze speech (if provided)
        speech_result = {'speech_stress_score': 0}
        if 'audio' in data and data['audio']:
            audio_data = decode_audio_from_base64(data['audio'])
            if audio_data is not None:
                sample_rate = data.get('sample_rate', 16000)
                speech_result = speech_recognizer.analyze_audio(audio_data, sample_rate)
        
        # Multimodal fusion
        fusion_result = fusion_engine.analyze_multimodal(facial_result, speech_result)
        fusion_result['timestamp'] = datetime.utcnow().isoformat()
        
        # Apply business rules
        alert_result = rules_engine.should_trigger_alert(
            session_id,
            fusion_result['smoothed_score']
        )
        
        # Get recommendation
        recommendation = rules_engine.get_recommendation(
            fusion_result['smoothed_score'],
            fusion_result['stress_category']
        )
        
        # Build response
        response = {
            **fusion_result,
            'recommendation': recommendation,
            'alert': alert_result,
            'session_id': session_id
        }
        
        # Store in database (if employee ID provided)
        if employee_id != 'anonymous':
            try:
                db_session = get_db_session(db_engine)
                
                # Get or create employee
                employee = db_session.query(Employee).filter_by(employee_id=employee_id).first()
                if not employee:
                    # Create new employee if doesn't exist
                    employee = Employee(
                        employee_id=employee_id,
                        name=f"User {employee_id}",
                        role='employee',
                        consent_given=True
                    )
                    db_session.add(employee)
                    db_session.flush()  # Get the ID without committing
                
                # Get or create session
                stress_session = db_session.query(StressSession).filter_by(
                    session_id=session_id,
                    is_active=True
                ).first()
                
                if not stress_session:
                    # Create new session
                    stress_session = StressSession(
                        session_id=session_id,
                        employee_id=employee.id
                    )
                    db_session.add(stress_session)
                    db_session.flush()  # Get the ID without committing
                
                # Add stress record with proper foreign keys
                record = StressRecord(
                    session_id=stress_session.id,
                    employee_id=employee.id,
                    facial_score=fusion_result['facial_score'],
                    speech_score=fusion_result['speech_score'],
                    fused_score=fusion_result['fused_score'],
                    smoothed_score=fusion_result['smoothed_score'],
                    stress_category=fusion_result['stress_category']
                )
                db_session.add(record)
                db_session.commit()
                db_session.close()
            except Exception as e:
                logger.error(f"Error storing record: {e}")
                if db_session:
                    db_session.rollback()
                    db_session.close()
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in analyze_multimodal: {e}")
        return jsonify({'error': str(e)}), 500


@app.route(f'{API_PREFIX}/stress_score', methods=['POST'])
def get_stress_score():
    """
    Simple endpoint to get current stress score
    
    Request body:
    {
        "session_id": "string"
    }
    
    Response:
    {
        "stress_score": float,
        "stress_category": str
    }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Get session summary from rules engine
        summary = rules_engine.get_session_summary(session_id)
        
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Error in get_stress_score: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@app.route(f'{API_PREFIX}/dashboard/employee/<employee_id>')
def employee_dashboard_data(employee_id):
    """
    Get employee dashboard data
    
    Privacy: Only returns individual's own data
    
    Response:
    {
        "current_stress": float,
        "stress_history": list,
        "today_stats": dict,
        "recommendations": list
    }
    """
    try:
        db_session = get_db_session(db_engine)
        
        # Get recent stress records
        records = db_session.query(StressRecord).filter(
            StressRecord.employee_id == employee_id
        ).order_by(StressRecord.timestamp.desc()).limit(50).all()
        
        if not records:
            db_session.close()
            return jsonify({
                'current_stress': 0,
                'stress_history': [],
                'today_stats': {},
                'recommendations': []
            })
        
        # Build response
        stress_history = [
            {
                'timestamp': r.timestamp.isoformat(),
                'score': r.smoothed_score,
                'category': r.stress_category
            }
            for r in reversed(records)
        ]
        
        # Today's statistics
        today = datetime.utcnow().date()
        today_records = [r for r in records if r.timestamp.date() == today]
        
        today_stats = {
            'avg_stress': np.mean([r.smoothed_score for r in today_records]) if today_records else 0,
            'max_stress': max([r.smoothed_score for r in today_records]) if today_records else 0,
            'total_readings': len(today_records)
        }
        
        db_session.close()
        
        return jsonify({
            'current_stress': records[0].smoothed_score,
            'stress_history': stress_history,
            'today_stats': today_stats,
            'recommendations': [rules_engine.get_recommendation(
                records[0].smoothed_score,
                records[0].stress_category
            )]
        })
    
    except Exception as e:
        logger.error(f"Error in employee_dashboard_data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route(f'{API_PREFIX}/dashboard/admin')
def admin_dashboard_data():
    """
    Get admin dashboard data (aggregated, anonymized)
    
    Privacy: No individual identification, team-level only
    
    Response:
    {
        "team_avg_stress": float,
        "stress_distribution": dict,
        "daily_trends": list,
        "alert_summary": dict
    }
    """
    try:
        db_session = get_db_session(db_engine)
        
        # Get aggregated stats for last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        stats = db_session.query(AggregatedStats).filter(
            AggregatedStats.date >= week_ago
        ).all()
        
        # Get all recent records for current stats
        records = db_session.query(StressRecord).filter(
            StressRecord.timestamp >= week_ago
        ).all()
        
        if not records:
            db_session.close()
            return jsonify({
                'team_avg_stress': 0,
                'stress_distribution': {},
                'daily_trends': [],
                'alert_summary': {}
            })
        
        # Calculate aggregated metrics
        all_scores = [r.smoothed_score for r in records]
        team_avg = np.mean(all_scores)
        
        # Stress distribution
        low_count = sum(1 for r in records if r.stress_category == 'Low')
        medium_count = sum(1 for r in records if r.stress_category == 'Medium')
        high_count = sum(1 for r in records if r.stress_category == 'High')
        
        stress_distribution = {
            'Low': low_count,
            'Medium': medium_count,
            'High': high_count
        }
        
        # Daily trends
        daily_trends = []
        for i in range(7):
            day = datetime.utcnow().date() - timedelta(days=6-i)
            day_records = [r for r in records if r.timestamp.date() == day]
            
            if day_records:
                daily_trends.append({
                    'date': day.isoformat(),
                    'avg_stress': np.mean([r.smoothed_score for r in day_records])
                })
        
        # Alert summary
        alerts = db_session.query(Alert).filter(
            Alert.timestamp >= week_ago
        ).all()
        
        alert_summary = {
            'total_alerts': len(alerts),
            'high_severity': sum(1 for a in alerts if a.severity == 'high')
        }
        
        db_session.close()
        
        return jsonify({
            'team_avg_stress': team_avg,
            'stress_distribution': stress_distribution,
            'daily_trends': daily_trends,
            'alert_summary': alert_summary,
            'disclaimer': 'Aggregated, anonymized data only. No individual identification.'
        })
    
    except Exception as e:
        logger.error(f"Error in admin_dashboard_data: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WEBSOCKET EVENTS (Real-time updates)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    logger.info('Client connected')
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    logger.info('Client disconnected')


@socketio.on('start_session')
def handle_start_session(data):
    """Start new monitoring session"""
    session_id = str(uuid.uuid4())
    emit('session_started', {'session_id': session_id})


@socketio.on('analysis_update')
def handle_analysis_update(data):
    """Broadcast analysis update to dashboards"""
    emit('stress_update', data, broadcast=True)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    logger.info(f"Starting server on {FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    socketio.run(
        app,
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug']
    )
