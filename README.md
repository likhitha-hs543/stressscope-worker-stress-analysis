# 🧠 Worker Stress Analysis System

**Real-time stress monitoring through facial and speech emotion recognition**

A comprehensive, privacy-first multimodal AI system for workplace stress detection and analysis. Built with academic rigor and industrial practicality.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [API Documentation](#api-documentation)
- [Training Models](#training-models)
- [Privacy & Ethics](#privacy--ethics)
- [Evaluation & Validation](#evaluation--validation)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Overview

### What This System Does

This system is **NOT** a mental illness diagnostic tool. It is a **stress indicator detection system** that:

- Measures physical manifestations of stress through facial expressions and voice patterns
- Provides self-awareness tools for employees
- Offers aggregated, anonymized insights for organizational wellness
- Focuses on **trend analysis** rather than instant judgment

### Mental Model

```
Human → Signals (Face + Voice) → AI Analysis → Insights → Decision Support
```

**Key Principle:** Speech reflects internal stress better (60% weight), facial emotion reflects external expression (40% weight).

---

## 🏗️ System Architecture

### Signal Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   Webcam     │              │  Microphone  │            │
│  │   (Video)    │              │   (Audio)    │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼───────────────────────────────┼──────────────────┘
          │                               │
┌─────────▼────────────────┐    ┌────────▼─────────────────┐
│   PREPROCESSING          │    │   PREPROCESSING          │
│  - Face detection        │    │  - Noise removal         │
│  - Crop & resize         │    │  - Silence trimming      │
│  - Normalize             │    │  - Normalization         │
└─────────┬────────────────┘    └────────┬─────────────────┘
          │                               │
┌─────────▼────────────────┐    ┌────────▼─────────────────┐
│   FEATURE EXTRACTION     │    │   FEATURE EXTRACTION     │
│  - CNN (automatic)       │    │  - MFCCs                 │
│  - Emotion patterns      │    │  - Pitch (F0)            │
│                          │    │  - Energy (RMS)          │
│                          │    │  - Speech rate           │
└─────────┬────────────────┘    └────────┬─────────────────┘
          │                               │
┌─────────▼────────────────┐    ┌────────▼─────────────────┐
│   EMOTION RECOGNITION    │    │   STRESS RECOGNITION     │
│  CNN Model               │    │  Random Forest / SVM     │
│  Output: Emotion probs   │    │  Output: Stress level    │
└─────────┬────────────────┘    └────────┬─────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                  ┌───────▼───────┐
                  │ FUSION ENGINE │
                  │  60% Speech   │
                  │  40% Facial   │
                  └───────┬───────┘
                          │
                  ┌───────▼───────┐
                  │ RULES ENGINE  │
                  │ - Thresholds  │
                  │ - Alerts      │
                  │ - Trends      │
                  └───────┬───────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
┌─────────▼────────────┐        ┌────────▼─────────────┐
│  EMPLOYEE DASHBOARD  │        │   ADMIN DASHBOARD    │
│  - Personal stress   │        │  - Team aggregates   │
│  - Recommendations   │        │  - Anonymized data   │
│  - Timeline          │        │  - Trends & alerts   │
└──────────────────────┘        └──────────────────────┘
```

---

## ✨ Key Features

### 🎭 Facial Emotion Recognition
- CNN-based emotion classification (6 emotions)
- Haar Cascade face detection
- Real-time frame processing (1 fps)
- Stress mapping: Angry + Fear + Sad → Stress indicators

### 🎤 Speech Stress Recognition
- MFCC-based acoustic feature extraction
- Pitch, energy, and speech rate analysis
- Random Forest / SVM classification
- 3-level stress detection (Low/Medium/High)

### 🔀 Multimodal Fusion
- Weighted combination (60% speech, 40% face)
- Temporal smoothing (moving average)
- Confidence scoring
- Agreement-based reliability

### 📊 Dual Dashboard System

**Employee Dashboard:**
- Personal stress monitoring
- Real-time feedback
- Stress timeline visualization
- Actionable recommendations

**Admin Dashboard:**
- Team-level aggregated metrics
- Privacy-preserving analytics
- Weekly trends
- Alert summaries (no individual identification)

### ⚖️ Business Rules Engine
- High stress duration tracking
- Alert cooldown management
- Trend analysis
- Recommendation generation

### 💾 Data Management
- SQLite database
- Session tracking
- No raw video/audio storage
- 90-day data retention

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Webcam (for facial analysis)
- Microphone (for speech analysis)
- 4GB+ RAM recommended

### Step 1: Clone Repository

```bash
cd "d:\PROJECTS\Working\worker stress analysis"
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with PyAudio:
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# Linux
sudo apt-get install portaudio19-dev python3-pyaudio

# Mac
brew install portaudio
pip install pyaudio
```

### Step 4: Initialize Database

```bash
python -c "from modules.database import init_database; init_database()"
```

---

## 🎯 Quick Start

### Option 1: Web Application (Recommended)

```bash
python app.py
```

Open browser and navigate to: `http://localhost:5000`

### Option 2: Real-time CLI

```bash
python modules/realtime_processor.py
```

### Option 3: API Only

```bash
# Start Flask server
python app.py

# In another terminal, test API
curl -X POST http://localhost:5000/api/v1/health
```

---

## 📁 Project Structure

```
worker stress analysis/
│
├── modules/                        # Core AI modules
│   ├── facial_recognition.py      # CNN-based face emotion
│   ├── speech_recognition.py      # MFCC-based speech stress
│   ├── multimodal_fusion.py       # Fusion engine (60-40 weighting)
│   ├── rules_engine.py             # Business logic & alerts
│   ├── database.py                 # SQLAlchemy models
│   └── realtime_processor.py       # Live capture & processing
│
├── templates/                      # Frontend
│   └── index.html                  # Dual dashboard UI
│
├── models/                         # Model storage
│   └── trained/                    # Saved models (.h5, .pkl)
│
├── data/                           # Data storage
│   ├── processed/                  # Processed features
│   ├── raw_videos/                 # (Not used - privacy)
│   └── raw_audio/                  # (Not used - privacy)
│
├── app.py                          # Flask backend API
├── config.py                       # Configuration file
├── requirements.txt                # Python dependencies
│
├── train_facial_model.py           # Facial model training
├── train_speech_model.py           # Speech model training
│
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 🔬 Technical Details

### Facial Emotion Recognition

**Architecture:**
- 4 convolutional blocks
- Batch normalization
- Dropout regularization
- Dense classification head

**Input:** 48×48 grayscale images

**Output:** 6 emotion probabilities (Angry, Fear, Happy, Neutral, Sad, Surprise)

**Training Dataset:** FER2013 (or custom)

**Accuracy Target:** 60-70% (benchmark for FER2013)

### Speech Stress Recognition

**Features Extracted:**
- **MFCCs:** 13 coefficients (vocal tract characteristics)
- **Pitch (F0):** Fundamental frequency (vocal cord tension)
- **Energy (RMS):** Loudness and voice quality
- **Speech Rate:** Temporal patterns

**Classifier:** Random Forest (100 trees, depth=10)

**Alternative:** SVM with RBF kernel

**Accuracy Target:** 70-80%

### Multimodal Fusion

**Formula:**
```
Fused Score = 0.6 × Speech Score + 0.4 × Facial Score
```

**Justification:**
- Speech is harder to consciously control
- Reflects internal physiological state
- Face can mask emotions (social conditioning)

**Smoothing:** 5-point moving average for temporal stability

### Stress Thresholds

| Category | Score Range | Interpretation |
|----------|-------------|----------------|
| **Low** | 0-33 | Minimal stress indicators |
| **Medium** | 33-66 | Moderate stress indicators |
| **High** | 66-100 | Significant stress indicators |

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api/v1
```

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-06T12:00:00",
  "modules": {
    "facial_recognition": true,
    "speech_recognition": true,
    "fusion_engine": true,
    "rules_engine": true,
    "database": true
  }
}
```

#### `POST /analyze_face`
Analyze facial emotion from image.

**Request:**
```json
{
  "image": "base64_encoded_image"
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
  "confidence": 0.85
}
```

#### `POST /analyze_speech`
Analyze speech stress from audio.

**Request:**
```json
{
  "audio": "base64_encoded_audio",
  "sample_rate": 16000
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
  }
}
```

#### `POST /analyze_multimodal`
Perform complete multimodal analysis.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "audio": "base64_encoded_audio",
  "sample_rate": 16000,
  "employee_id": "EMP001",
  "session_id": "session_123"
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
  "recommendation": "Consider taking a short break.",
  "alert": {
    "should_alert": false,
    "reason": null
  }
}
```

#### `GET /dashboard/employee/<employee_id>`
Get employee dashboard data.

**Response:**
```json
{
  "current_stress": 42.0,
  "stress_history": [
    {"timestamp": "2026-01-06T12:00:00", "score": 38.0, "category": "Medium"},
    {"timestamp": "2026-01-06T12:05:00", "score": 42.0, "category": "Medium"}
  ],
  "today_stats": {
    "avg_stress": 40.0,
    "max_stress": 55.0,
    "total_readings": 50
  },
  "recommendations": ["Consider taking a short break."]
}
```

#### `GET /dashboard/admin`
Get admin dashboard data (aggregated).

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
    {"date": "2026-01-01", "avg_stress": 35.0},
    {"date": "2026-01-02", "avg_stress": 38.0}
  ],
  "alert_summary": {
    "total_alerts": 12,
    "high_severity": 3
  },
  "disclaimer": "Aggregated, anonymized data only."
}
```

---

## 🎓 Training Models

### Training Facial Model

```bash
# With FER2013 dataset
python train_facial_model.py --data path/to/fer2013.csv --epochs 50 --batch-size 64

# With custom data
python train_facial_model.py --epochs 30
```

**Expected Output:**
- `training_history.png` - Accuracy/loss curves
- `confusion_matrix.png` - Confusion matrix visualization
- `models/trained/facial_emotion_model.h5` - Trained model

### Training Speech Model

```bash
# With audio dataset
python train_speech_model.py --data-dir path/to/audio_data --model rf --n-estimators 100

# With SVM
python train_speech_model.py --model svm
```

**Dataset Structure:**
```
audio_data/
  low/
    sample1.wav
    sample2.wav
  medium/
    sample1.wav
  high/
    sample1.wav
```

**Expected Output:**
- `speech_confusion_matrix.png` - Confusion matrix
- `feature_importance.png` - Feature importance (RF only)
- `models/trained/speech_stress_model.pkl` - Trained model

---

## 🔒 Privacy & Ethics

### What We DO

✅ Detect stress indicators (physical signals)  
✅ Store aggregated metrics only  
✅ Anonymize admin-level data  
✅ Provide self-awareness tools  
✅ Respect data retention limits (90 days)  

### What We DON'T DO

❌ Diagnose mental illness  
❌ Store raw video/audio  
❌ Allow individual surveillance by admins  
❌ Make employment decisions  
❌ Track without consent  

### Privacy Principles

1. **Consent Required:** Users must opt-in
2. **Transparency:** Clear about what's measured
3. **Aggregation:** Admin sees team-level data only
4. **No Micromanagement:** Individual monitoring only for self
5. **Data Minimization:** Store derived metrics, not raw data

---

## 📈 Evaluation & Validation

### Model Accuracy

**Facial Emotion Recognition:**
- Typical accuracy on FER2013: 60-65%
- Challenge: In-the-wild conditions, occlusions

**Speech Stress Recognition:**
- Expected accuracy: 70-80%
- Challenge: Background noise, speaker variability

**Multimodal Fusion:**
- Improved reliability through independent signals
- Reduces single-modality false positives

### Validation Approach

1. **Accuracy metrics:** Precision, recall, F1-score
2. **Confusion matrices:** Per-class performance
3. **Cross-validation:** 5-fold CV for robustness
4. **Real-world testing:** Pilot with volunteers

### Key Statement for Academic Defense

> "The system focuses on **continuous stress trends** rather than single-frame predictions, improving reliability over time. It is designed for **trend analysis**, not instant judgment."

---

## 🚢 Deployment

### Local Development

```bash
python app.py
# Access at http://localhost:5000
```

### Production Deployment (Option 1: Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Production Deployment (Option 2: Docker)

```dockerfile
# Dockerfile (create this)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t stress-analysis .
docker run -p 5000:5000 stress-analysis
```

### Cloud Deployment

**AWS EC2 / Azure VM:**
1. Launch instance (Ubuntu 20.04, t2.medium)
2. Install Python and dependencies
3. Clone repository
4. Run with gunicorn
5. Configure nginx as reverse proxy

**Heroku:**
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create stress-analysis-app
git push heroku main
```

---

## 🐛 Troubleshooting

### Issue: Webcam not detected

**Solution:**
```python
# Test webcam
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())  # Should be True
```

### Issue: PyAudio installation fails

**Solution (Windows):**
```bash
pip install pipwin
pipwin install pyaudio
```

**Solution (Linux):**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### Issue: Low facial recognition accuracy

**Possible causes:**
- Poor lighting
- Face not centered
- Occlusions (glasses, mask)

**Solution:** Ensure good lighting, face camera directly

### Issue: High latency (>5 seconds)

**Solution:**
- Reduce frame processing rate
- Use smaller models
- Enable GPU acceleration (TensorFlow GPU)

---

## 📚 References

### Datasets

- **FER2013:** Facial Expression Recognition Challenge  
- **RAVDESS:** Ryerson Audio-Visual Database of Emotional Speech  
- **SAVEE:** Surrey Audio-Visual Expressed Emotion

### Research Papers

1. Goodfellow et al. (2013) - "Challenges in Representation Learning: Facial Expression Recognition Challenge"
2. Livingstone & Russo (2018) - "The Ryerson Audio-Visual Database of Emotional Speech"
3. Busso et al. (2008) - "IEMOCAP: Interactive emotional dyadic motion capture database"

### Technologies Used

- **TensorFlow/Keras:** Deep learning framework
- **OpenCV:** Computer vision
- **Librosa:** Audio processing
- **Scikit-learn:** Machine learning
- **Flask:** Web framework
- **Chart.js:** Visualization

---

## 🤝 Contributing

This is an academic/demonstration project. Contributions welcome for:
- Model improvements
- Additional features
- Bug fixes
- Documentation

---

## 📄 License

MIT License - Free to use for educational and research purposes.

**Disclaimer:** This system is for research and self-awareness purposes only. It is not a medical device and should not be used for clinical diagnosis or employment decisions.

---

## 👨‍💻 Author

Built with academic rigor and practical focus for worker wellness monitoring.

**Academic Context:**  
This project demonstrates understanding of:
- Multimodal machine learning
- Real-time signal processing
- Privacy-preserving analytics
- System design and deployment
- Human-computer interaction

---

## 🎓 Academic Defense Points

### "How does it work?"

> "The system measures physical signals: facial muscle tension via CNN-based emotion recognition, and voice production changes via MFCC-based acoustic analysis. These are independent sensors that, when fused with 60-40 weighting (speech-dominant), provide robust stress indicators."

### "How accurate is it?"

> "Facial emotion recognition achieves ~60-65% on FER2013, speech stress ~70-80%. More importantly, the system focuses on **trend analysis** over time, not single-frame accuracy, which significantly improves reliability."

### "Privacy concerns?"

> "The system is privacy-first: raw video/audio are never stored, only derived metrics. Admin dashboards show aggregated team-level data only—no individual identification. Employees control their own data."

### "Why 60-40 weighting?"

> "Speech reflects internal physiological state (vocal cord tension, breathing patterns) which is harder to consciously control. Facial expressions can be socially masked. This weighting reflects biomechanical reality."

### "Is this a medical tool?"

> "No. This detects stress **indicators** for self-awareness and organizational wellness trends. It is explicitly not a diagnostic tool and makes no clinical claims."

---

**🎯 Remember: This system measures signals → interprets patterns → suggests actions. It's about self-awareness, not surveillance.**
