# 📘 User Guide - Worker Stress Analysis System

**Complete setup, installation, and usage guide**

---

## 🚀 Quick Setup (30 Minutes)

### Prerequisites
- Python 3.8+
- Webcam (for facial analysis)
- Microphone (for speech analysis)
- 4GB+ RAM recommended

---

## ⚡ Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 (Windows): PyAudio Fix

PyAudio requires special installation on Windows:

```bash
pip install pipwin
pipwin install pyaudio
```

**Alternative (if pipwin fails):**
Download unofficial binary from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

```bash
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Mac:**
```bash
brew install portaudio
pip install pyaudio
```

### Step 3: Environment Setup

```bash
# Copy example environment file
copy .env.example .env

# Generate secure SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# Edit .env and paste the generated key
```

### Step 4: Initialize Database

```bash
python setup.py
```

Or manually:
```bash
python -c "from modules.database import init_database; init_database()"
```

---

## 🎯 Running the Application

### Web Application (Recommended)

```bash
python app.py
```

Visit: **http://localhost:5000**

### Check System Status

```bash
curl http://localhost:5000/api/v1/model_status
```

Expected response:
```json
{
  "system_ready": true,
  "facial_model_ready": true,
  "speech_model_ready": true,
  "message": "All models loaded successfully"
}
```

---

## 🎓 Training Models

### Facial Emotion Model (MobileNetV2 Transfer Learning)

**1. Download FER2013 Dataset**
- Source: https://www.kaggle.com/datasets/msambare/fer2013

**2. Train the model**

```bash
python train_facial_mobilenetv2.py --data-dir "data/face/FER 2013" --phase1-epochs 15 --phase2-epochs 10
```

**Expected:**
- Training time: ~1.5 hours (CPU), ~30 min (GPU)
- Accuracy: **50-60%** (7-class emotions)
- Output: `models/trained/facial_emotion_model_v2.keras`

### Speech Stress Model (Ensemble on RAVDESS)

**1. Download RAVDESS Dataset**
- Source: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio

**2. Organize the data**

```bash
python prepare_ravdess_data.py --input-dir "data/speech/SER/Ravdess" --output-dir "data/speech/SER/prepared"
```

**3. Train the model**

```bash
python train_speech_from_ravdess.py --data-dir "data/speech/SER/prepared"
```

**Expected:**
- Training time: 30-60 minutes
- Accuracy: **70-80%** (3-class stress levels)
- Output: `models/trained/speech_stress_model.pkl`

---

## ✅ Verification

### Test Models Loaded

```bash
python validate_system.py
```

Should show:
- ✅ Facial model found
- ✅ Speech model found  
- ✅ Fusion engine operational
- ✅ 7/7 tests passed

### Test API

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Model status
curl http://localhost:5000/api/v1/model_status
```

---

## 🎪 Using the Dashboards

### Employee Dashboard
1. Open http://localhost:5000
2. Click "Start Monitoring"
3. Grant webcam/microphone permissions
4. View real-time stress score
5. Receive personalized recommendations

### Admin Dashboard
1. Click "Admin Dashboard" tab
2. View team-level aggregated metrics
3. See anonymized stress trends
4. Review alert summaries

**Note:** Admin cannot see individual employee data - privacy-preserved

---

## 🐛 Troubleshooting

### Webcam Not Detected

```python
# Test webcam
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())  # Should be True
```

### Database Errors

```bash
# Reset database
rm stress_analysis.db
python setup.py
```

### Models Not Loading

```bash
# Check model paths
python -c "from config import FACE_MODEL_PATH, SPEECH_MODEL_PATH; print(f'Facial: {FACE_MODEL_PATH}'); print(f'Speech: {SPEECH_MODEL_PATH}')"

# Verify files exist
dir models\trained
```

### High Latency (>5 seconds)

**Solutions:**
- Reduce frame processing rate in config.py
- Use smaller models (reduce complexity)
- Enable GPU acceleration:
  ```bash
  pip install tensorflow-gpu
  ```

### CORS Errors

Update `app.py` line ~38 to allow your frontend origin:
```python
CORS(app, resources={r"/api/*": {"origins": "http://your-frontend-domain.com"}})
```

---

## 📊 Expected Performance

With proper training:
- **Facial Emotion**: 50-60% (v2 with transfer learning)
- **Speech Stress**: 70-80% (ensemble on quality dataset)
- **Multimodal Fusion**: ~60% (weighted combination)

---

## 🔒 Privacy Features

**What's Stored:**
- ✅ Aggregated stress scores
- ✅ Timestamps
- ✅ Session metadata

**What's NOT Stored:**
- ❌ Raw video frames
- ❌ Raw audio files
- ❌ Facial images
- ❌ Individual identifiable data (in admin view)

**Data Retention:** 90 days (configurable in `.env`)

---

## 🚢 Production Deployment

### Security Checklist
1. **Change SECRET_KEY** in `.env` to strong random value
2. **Set DEBUG=False** in `.env`
3. **Configure CORS** to restrict origins (not `*`)
4. **Use PostgreSQL** instead of SQLite
5. **Enable HTTPS** (use nginx + Let's Encrypt)

### Deploy with Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Deploy with Docker

```dockerfile
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

---

## 💡 Tips

- 🎥 **Better webcam** = better face detection
- 🎤 **External microphone** recommended for speech
- 💾 **More training data** = better accuracy
- ⚡ **GPU** significantly speeds up facial inference
- 🔒 **Privacy-first** design - no raw media stored

---

## ✨ You're Ready!

Start the application:
```bash
python app.py
```

Visit: **http://localhost:5000**

Monitor stress in real-time! 🧠
