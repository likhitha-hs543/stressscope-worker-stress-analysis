# 🧠 StressScope - Worker Stress Analysis System

**Real-time multimodal AI system for workplace stress detection**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 What It Does

StressScope combines **facial emotion recognition** and **speech stress analysis** to provide real-time, privacy-preserving workplace stress detection.

**Key Features:**
- 🎭 Facial emotion recognition (MobileNetV2, 50.17% accuracy)
- 🎤 Speech stress detection (Ensemble, 69.91% accuracy)  
- 🔀 Multimodal fusion (~60% combined accuracy)
- 📊 Dual dashboards (employee self-awareness + admin analytics)
- 🔒 Privacy-first design (no raw media storage)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python setup.py

# Run application
python app.py
```

Visit: **http://localhost:5000**

📘 **Full Setup Guide:** See [docs/User_Guide.md](docs/User_Guide.md)

---

## 📊 System Performance

| Component | Accuracy | Method |
|-----------|----------|--------|
| **Facial v2** | 50.17% | MobileNetV2 (transfer learning) |
| **Speech v1** | 69.91% | Ensemble (Random Forest + GB) |
| **Fusion** | ~60% | Weighted (60% speech, 40% facial) |

**Improvement:** Facial v1 (13.21%) → v2 (50.17%) = **+36.96pp** through transfer learning

---

## 🏗️ Architecture

```
Webcam + Microphone
        ↓
Facial (MobileNetV2) + Speech (Ensemble)
        ↓
Multimodal Fusion (60/40)
        ↓
Rules Engine (alerts + recommendations)
        ↓
Dashboard (employee + admin)
```

🏛️ **Technical Details:** See [docs/Design_and_Implementation.md](docs/Design_and_Implementation.md)

---

## 📁 Project Structure

```
worker stress analysis/
├── app.py                          # Flask API backend
├── config.py                       # Configuration
├── requirements.txt                # Dependencies
├── modules/                        # Core AI modules
│   ├── facial_recognition.py      # MobileNetV2 facial analysis
│   ├── speech_recognition.py      # Ensemble speech analysis
│   ├── multimodal_fusion.py       # Fusion engine
│   └── rules_engine.py             # Business logic
├── templates/                      # Frontend dashboard
├── models/trained/                 # Saved models (.keras, .pkl)
├── train_facial_mobilenetv2.py    # Facial training script
├── train_speech_from_ravdess.py   # Speech training script
└── docs/                           # Documentation
    ├── User_Guide.md               # Setup & usage
    └── Design_and_Implementation.md# Architecture & methodology
```

---

## 🎓 Training Models

### Facial Model (Transfer Learning)
```bash
python train_facial_mobilenetv2.py --data-dir "data/face/FER 2013" --phase1-epochs 15 --phase2-epochs 10
```

### Speech Model (Ensemble)
```bash
python prepare_ravdess_data.py --input-dir "data/speech/SER/Ravdess"
python train_speech_from_ravdess.py --data-dir "data/speech/SER/prepared"
```

📊 **Training Details:** See [TRAINING_STATUS.md](TRAINING_STATUS.md)

---

##Privacy & Ethics

**What's Stored:**
- ✅ Aggregated stress scores + timestamps
- ✅ Session metadata

**What's NOT Stored:**
- ❌ Raw video/audio
- ❌ Facial images
- ❌ Individual PII (in admin view)

**Purpose:** Self-awareness tool, not surveillance or diagnosis

---

## 📚 Documentation

- 📘 **[User Guide](docs/User_Guide.md)** - Installation, setup, troubleshooting
- 🏛️ **[Design & Implementation](docs/Design_and_Implementation.md)** - Architecture, methodology, viva defense
- 📊 **[Training Status](TRAINING_STATUS.md)** - Model versions, performance comparison
- 📡 **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference

---

## 🤝 Academic Context

**This project demonstrates:**
- Complete ML pipeline (baseline → improvement → deployment)
- Transfer learning applied correctly
- Systematic iteration (+36.96pp improvement)
- Privacy-preserving multimodal AI
- Production-ready system design

**Suitable for:** Final year projects, ML coursework, research demonstrations

---

## 📄 License

MIT License - Free for educational and research purposes.

**Disclaimer:** This system is for research and self-awareness only. Not a medical device.

---

## 👨‍💻 Author

Built to demonstrate professional ML engineering and multimodal system design.

**GitHub:** [likhitha-hs543/stressscope-worker-stress-analysis](https://github.com/likhitha-hs543/stressscope-worker-stress-analysis)

---

**Status:** ✅ Production Ready | 📊 ~60% Multimodal Accuracy | 🔒 Privacy-First Design
