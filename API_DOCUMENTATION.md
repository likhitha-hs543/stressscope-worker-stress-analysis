# Worker Stress Analysis System - API Reference

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication
Currently no authentication required (add JWT/OAuth for production).

---

## Endpoints

### 1. Health Check

#### `GET /health`

Check if the system is running and all modules are initialized.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-06T12:00:00.000Z",
  "modules": {
    "facial_recognition": true,
    "speech_recognition": true,
    "fusion_engine": true,
    "rules_engine": true,
    "database": true
  }
}
```

---

### 2. Facial Analysis

#### `POST /analyze_face`

Analyze facial emotion from a single image frame.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..." // Base64 encoded image
}
```

**Response:**
```json
{
  "face_detected": true,
  "emotion_probs": {
    "Angry": 0.15,
    "Fear": 0.10,
    "Happy": 0.30,
    "Neutral": 0.25,
    "Sad": 0.10,
    "Surprise": 0.10
  },
  "dominant_emotion": "Happy",
  "facial_stress_score": 35.0,
  "confidence": 0.85,
  "face_location": [100, 150, 200, 200]
}
```

**Error Response:**
```json
{
  "error": "No image provided"
}
```

---

### 3. Speech Analysis

#### `POST /analyze_speech`

Analyze stress level from audio segment.

**Request Body:**
```json
{
  "audio": "base64_encoded_audio_data",
  "sample_rate": 16000  // Optional, defaults to 16000
}
```

**Response:**
```json
{
  "stress_level": "Medium",
  "speech_stress_score": 55.0,
  "probabilities": {
    "Low": 0.20,
    "Medium": 0.55,
    "High": 0.25
  },
  "features": {
    "feature_count": 75,
    "audio_duration": 10.5
  }
}
```

**Error Response:**
```json
{
  "error": "Invalid audio format",
  "stress_level": "Unknown"
}
```

---

### 4. Multimodal Analysis

#### `POST /analyze_multimodal`

Perform complete analysis using both face and speech (recommended).

**Request Body:**
```json
{
  "image": "base64_encoded_image",      // Optional
  "audio": "base64_encoded_audio",      // Optional
  "sample_rate": 16000,                 // Optional
  "employee_id": "EMP001",              // Optional
  "session_id": "session_123456"        // Optional
}
```

**Response:**
```json
{
  "facial_score": 35.0,
  "speech_score": 55.0,
  "fused_score": 47.0,
  "smoothed_score": 45.5,
  "stress_category": "Medium",
  "confidence": 0.78,
  "has_facial": true,
  "has_speech": true,
  "facial_emotion": "Happy",
  "speech_stress_level": "Medium",
  "recommendation": "Consider taking a short break.",
  "alert": {
    "should_alert": false,
    "reason": null,
    "stress_score": 45.5,
    "duration_seconds": 0,
    "alert_count": 0
  },
  "session_id": "session_123456",
  "timestamp": "2026-01-06T12:00:00.000Z"
}
```

---

### 5. Current Stress Score

#### `POST /stress_score`

Get current stress score and session summary.

**Request Body:**
```json
{
  "session_id": "session_123456"
}
```

**Response:**
```json
{
  "exists": true,
  "alert_count": 2,
  "high_stress_duration": 120.5,
  "is_currently_high_stress": false,
  "session_duration": 1800.0
}
```

---

### 6. Employee Dashboard Data

#### `GET /dashboard/employee/<employee_id>`

Get personalized dashboard data for an employee.

**URL Parameters:**
- `employee_id` (string): Employee identifier

**Response:**
```json
{
  "current_stress": 42.0,
  "stress_history": [
    {
      "timestamp": "2026-01-06T12:00:00",
      "score": 38.0,
      "category": "Medium"
    },
    {
      "timestamp": "2026-01-06T12:05:00",
      "score": 42.0,
      "category": "Medium"
    }
  ],
  "today_stats": {
    "avg_stress": 40.0,
    "max_stress": 55.0,
    "total_readings": 50
  },
  "recommendations": [
    "Consider taking a short break."
  ]
}
```

---

### 7. Admin Dashboard Data

#### `GET /dashboard/admin`

Get aggregated, anonymized team-level analytics.

**Response:**
```json
{
  "team_avg_stress": 38.5,
  "stress_distribution": {
    "Low": 120,
    "Medium": 180,
    "High": 50
  },
  "daily_trends": [
    {
      "date": "2026-01-01",
      "avg_stress": 35.0
    },
    {
      "date": "2026-01-02",
      "avg_stress": 38.0
    }
  ],
  "alert_summary": {
    "total_alerts": 12,
    "high_severity": 3
  },
  "disclaimer": "Aggregated, anonymized data only. No individual identification."
}
```

---

## WebSocket Events

### Connection

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected to server');
});
```

### Start Session

**Emit:**
```javascript
socket.emit('start_session', {
  employee_id: 'EMP001'
});
```

**Receive:**
```javascript
socket.on('session_started', (data) => {
  console.log('Session ID:', data.session_id);
});
```

### Real-time Updates

**Emit:**
```javascript
socket.emit('analysis_update', {
  session_id: 'session_123',
  stress_score: 45.5,
  category: 'Medium'
});
```

**Receive:**
```javascript
socket.on('stress_update', (data) => {
  console.log('Stress update:', data);
});
```

---

## Data Models

### Stress Record

```json
{
  "id": 1,
  "session_id": 1,
  "employee_id": 1,
  "timestamp": "2026-01-06T12:00:00",
  "facial_score": 35.0,
  "facial_emotion": "Happy",
  "facial_confidence": 0.85,
  "face_detected": true,
  "speech_score": 55.0,
  "speech_stress_level": "Medium",
  "speech_confidence": 0.70,
  "fused_score": 47.0,
  "smoothed_score": 45.5,
  "stress_category": "Medium",
  "overall_confidence": 0.78,
  "modality_used": "both",
  "alert_triggered": false
}
```

### Alert

```json
{
  "id": 1,
  "employee_id": 1,
  "session_id": 1,
  "timestamp": "2026-01-06T12:00:00",
  "alert_type": "prolonged_high_stress",
  "severity": "high",
  "stress_score": 75.0,
  "duration_seconds": 360,
  "message": "High stress detected for 6 minutes",
  "resolved": false,
  "resolved_at": null
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (missing parameters) |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Rate Limiting

Currently no rate limiting (add in production).

**Recommended for production:**
- 100 requests per minute per client
- Use `flask-limiter` package

---

## Example Usage

### Python

```python
import requests
import base64

# Read image
with open('face.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Analyze
response = requests.post(
    'http://localhost:5000/api/v1/analyze_face',
    json={'image': f'data:image/jpeg;base64,{image_data}'}
)

result = response.json()
print(f"Stress Score: {result['facial_stress_score']}")
```

### JavaScript

```javascript
async function analyzeFace(imageData) {
  const response = await fetch('http://localhost:5000/api/v1/analyze_face', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  });
  
  const result = await response.json();
  console.log('Stress Score:', result.facial_stress_score);
}
```

### cURL

```bash
curl -X POST http://localhost:5000/api/v1/health

curl -X POST http://localhost:5000/api/v1/analyze_face \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

---

## Configuration

Environment variables (optional):

```bash
export DATABASE_URL="sqlite:///stress_analysis.db"
export SECRET_KEY="your-secret-key-here"
export DEBUG=False
```

---

## Security Considerations

**For Production:**

1. **Add Authentication:** JWT tokens or OAuth2
2. **HTTPS Only:** Use SSL/TLS certificates
3. **Input Validation:** Sanitize all inputs
4. **Rate Limiting:** Prevent abuse
5. **CORS:** Configure allowed origins
6. **Data Encryption:** Encrypt sensitive data at rest

---

## Support

For issues or questions:
- Check logs: `logs/app.log`
- Enable debug mode: `DEBUG=True` in config
- Review README.md for troubleshooting
