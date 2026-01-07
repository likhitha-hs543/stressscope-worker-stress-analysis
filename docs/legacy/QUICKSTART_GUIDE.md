# 🚀 Quick Setup Guide

This guide will get your Worker Stress Analysis System running in **30 minutes** (excluding model training).

---

## ⚡ Quick Start (5 minutes)

### 1. **Install Dependencies**

```bash
# Basic installation
pip install -r requirements.txt
```

**Windows users - PyAudio Fix:**
```bash
# PyAudio requires special installation on Windows
pip install pipwin
pipwin install pyaudio
```

### 2. **Create Environment File**

```bash
# Copy the example file
copy .env.example .env

# Edit .env and change SECRET_KEY (IMPORTANT!)
# Generate a secure key with:
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. **Initialize Database**

```bash
python setup.py
```

Or manually:
```bash
python -c "from modules.database import init_database; init_database()"
```

### 4. **Check System Status**

```bash
python app.py
```

Then visit: `http://localhost:5000/api/v1/model_status`

You should see:
```json
{
  "system_ready": false,
  "facial_model_ready": false,
  "speech_model_ready": false,
  "message": "Models need training"
}
```

---

## 🎯 Model Training (2-4 hours)

### Option A: Use Demo Models (Quick Testing)

For **testing only**, the system will create untrained model architectures. These won't give accurate results but let you test the UI.

### Option B: Train Real Models (Production)

#### **Facial Emotion Model**

1. **Download FER2013 Dataset**
   - Get from: https://www.kaggle.com/datasets/msambare/fer2013
   - Or use your own labeled facial emotion dataset

2. **Train the model**
   ```bash
   python train_facial_model.py --data path/to/fer2013.csv --epochs 50
   ```

   Expected training time: 1-2 hours (CPU), 20-30 minutes (GPU)

#### **Speech Stress Model**

1. **Prepare Audio Data**
   
   Organize your audio files like this:
   ```
   speech_data/
     ├── low/
     │   ├── sample1.wav
     │   ├── sample2.wav
     ├── medium/
     │   ├── sample1.wav
     ├── high/
     │   ├── sample1.wav
   ```

2. **Train the model**
   ```bash
   python train_speech_model.py --data-dir path/to/speech_data --model rf
   ```

   Expected training time: 30 minutes - 2 hours (depends on dataset size)

---

## ✅ Verify Installation

### Check Models

```bash
# Check if models exist
python -c "from config import FACE_MODEL_PATH, SPEECH_MODEL_PATH; print(f'Facial: {FACE_MODEL_PATH.exists()}'); print(f'Speech: {SPEECH_MODEL_PATH.exists()}')"
```

### Test API

```bash
# Start server
python app.py

# In another terminal, test health
curl http://localhost:5000/api/v1/health

# Check model status
curl http://localhost:5000/api/v1/model_status
```

### Test Frontend

1. Open browser: `http://localhost:5000`
2. Click "Start Monitoring" on Employee Dashboard
3. Grant webcam permission
4. You should see live video feed

---

## 🎪 Demo Mode (Without Training)

Want to see the UI without training models? Run:

```bash
python app.py
```

The system will work with the following limitations:
- ⚠️ Facial analysis creates untrained model (random predictions)
- ⚠️ Speech analysis will throw errors (need trained model)
- ✅ UI, database, and real-time processing all work
- ✅ Can test webcam capture and frontend features

---

## 📚 Next Steps

### Production Deployment

1. **Change SECRET_KEY** in `.env` to a strong random value
2. **Train both models** with real datasets
3. **Configure CORS** in `app.py` line 38-40 (restrict origins)
4. **Use PostgreSQL** instead of SQLite (update DATABASE_URL in `.env`)
5. **Set DEBUG=False** in `.env`
6. **Deploy with gunicorn**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Testing Real-Time Processing

```bash
# Test real-time processor (standalone)
python modules/realtime_processor.py
```

### Adding Authentication

See `API_DOCUMENTATION.md` for implementing JWT authentication.

---

## 🐛 Troubleshooting

### PyAudio Won't Install (Windows)
```bash
pip install pipwin
pipwin install pyaudio
```

### Database Errors
```bash
# Reset database
rm stress_analysis.db
python -c "from modules.database import init_database; init_database()"
```

### Models Not Loading
Check file paths:
```bash
python -c "from config import MODELS_DIR; print(MODELS_DIR); import os; print(os.listdir(MODELS_DIR))"
```

### CORS Errors in Browser
Update `app.py` line 38-40 to allow your frontend origin.

---

## 📊 Expected Accuracies

With proper training:
- **Facial Emotion Recognition**: 60-70% accuracy (FER2013 is challenging)
- **Speech Stress Recognition**: 70-80% accuracy (depends on dataset quality)
- **Multimodal Fusion**: Better reliability than either alone

---

## 💡 Tips

- 🎥 **Webcam Quality**: Higher resolution = better face detection
- 🎤 **Microphone**: External mic recommended for speech analysis
- 💾 **Dataset Size**: More data = better model performance
- ⚡ **GPU**: Significantly speeds up training (optional)
- 🔒 **Privacy**: Raw video/audio are NEVER stored (configurable in `.env`)

---

## ✨ You're Ready!

Once models are trained, your system is production-ready. Start with:

```bash
python app.py
```

Visit `http://localhost:5000` and monitor stress in real-time! 🧠
