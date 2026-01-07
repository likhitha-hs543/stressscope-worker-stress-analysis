# Worker Stress Analysis System - Architecture Overview

## System Components Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                                 │
│  ┌───────────────────────────────┐    ┌─────────────────────────────────┐  │
│  │    EMPLOYEE DASHBOARD         │    │      ADMIN DASHBOARD            │  │
│  │  - Real-time monitoring       │    │  - Team aggregates              │  │
│  │  - Personal stress score      │    │  - Weekly trends                │  │
│  │  - Recommendations            │    │  - Anonymized analytics         │  │
│  │  - Stress timeline chart      │    │  - Alert summaries              │  │
│  └───────────────────────────────┘    └─────────────────────────────────┘  │
│                                  │                                           │
│                          ┌───────▼───────┐                                   │
│                          │ Templates/     │                                   │
│                          │ index.html     │                                   │
│                          └───────┬───────┘                                   │
└─────────────────────────────────┼─────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────────┐
│                         API LAYER (Flask Backend)                          │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  REST API Endpoints (app.py)                                       │   │
│  │  - GET  /api/v1/health                                             │   │
│  │  - POST /api/v1/analyze_face          (image → emotion)            │   │
│  │  - POST /api/v1/analyze_speech        (audio → stress)             │   │
│  │  - POST /api/v1/analyze_multimodal    (both → fused)               │   │
│  │  - GET  /api/v1/dashboard/employee    (personal data)              │   │
│  │  - GET  /api/v1/dashboard/admin       (aggregated data)            │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────────────┐
│                         PROCESSING LAYER                                   │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │  Real-time Processor (realtime_processor.py)                     │     │
│  │  - Video capture thread        → Queue → Face analysis           │     │
│  │  - Audio capture thread        → Queue → Speech analysis         │     │
│  │  - Processing thread           → Fusion → Results                │     │
│  │  Target latency: 2-5 seconds                                     │     │
│  └──────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │
            ┌─────────────────────┴──────────────────────┐
            │                                            │
┌───────────▼────────────────┐              ┌───────────▼─────────────────┐
│  FACIAL RECOGNITION        │              │  SPEECH RECOGNITION         │
│  (facial_recognition.py)   │              │  (speech_recognition.py)    │
│                            │              │                             │
│  ┌──────────────────────┐  │              │  ┌───────────────────────┐ │
│  │ 1. Face Detection    │  │              │  │ 1. Audio Preprocess   │ │
│  │    - Haar Cascade    │  │              │  │    - Noise removal    │ │
│  │    - Crop face       │  │              │  │    - Normalization    │ │
│  └──────────────────────┘  │              │  └───────────────────────┘ │
│                            │              │                             │
│  ┌──────────────────────┐  │              │  ┌───────────────────────┐ │
│  │ 2. Preprocessing     │  │              │  │ 2. Feature Extract    │ │
│  │    - Resize 48x48    │  │              │  │    - MFCCs (13)       │ │
│  │    - Grayscale       │  │              │  │    - Pitch (F0)       │ │
│  │    - Normalize [0,1] │  │              │  │    - Energy (RMS)     │ │
│  └──────────────────────┘  │              │  │    - Speech rate      │ │
│                            │              │  └───────────────────────┘ │
│  ┌──────────────────────┐  │              │                             │
│  │ 3. CNN Model         │  │              │  ┌───────────────────────┐ │
│  │    - 4 Conv blocks   │  │              │  │ 3. ML Model           │ │
│  │    - BatchNorm       │  │              │  │    - Random Forest    │ │
│  │    - Dropout         │  │              │  │    - 100 trees        │ │
│  │    - Dense layers    │  │              │  │    - Depth=10         │ │
│  └──────────────────────┘  │              │  └───────────────────────┘ │
│                            │              │                             │
│  Output: Emotion probs     │              │  Output: Stress level       │
│  {Angry, Fear, Happy...}   │              │  {Low, Medium, High}        │
│                            │              │                             │
│  Facial Stress Score:      │              │  Speech Stress Score:       │
│  Sum(Angry+Fear+Sad) * 100 │              │  Weighted prob * 100        │
└───────────┬────────────────┘              └───────────┬─────────────────┘
            │                                            │
            └─────────────────────┬──────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  MULTIMODAL FUSION        │
                    │  (multimodal_fusion.py)   │
                    │                           │
                    │  Formula:                 │
                    │  Score = 0.6×Speech +     │
                    │          0.4×Facial       │
                    │                           │
                    │  Smoothing:               │
                    │  5-point moving average   │
                    │                           │
                    │  Categorization:          │
                    │  Low:    0-33             │
                    │  Medium: 33-66            │
                    │  High:   66-100           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  BUSINESS RULES ENGINE    │
                    │  (rules_engine.py)        │
                    │                           │
                    │  Rules:                   │
                    │  - High stress > 5min     │
                    │    → Trigger alert        │
                    │  - Alert cooldown 30min   │
                    │  - Trend analysis         │
                    │  - Recommendations        │
                    │                           │
                    │  Decisions:               │
                    │  - should_alert?          │
                    │  - recommendation text    │
                    │  - risk_level             │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  DATABASE LAYER           │
                    │  (database.py)            │
                    │                           │
                    │  SQLite (SQLAlchemy)      │
                    │                           │
                    │  Tables:                  │
                    │  - employees              │
                    │  - sessions               │
                    │  - stress_records         │
                    │  - alerts                 │
                    │  - aggregated_stats       │
                    │                           │
                    │  Privacy:                 │
                    │  - No raw video/audio     │
                    │  - 90-day retention       │
                    │  - Anonymized aggregation │
                    └───────────────────────────┘
```

## Data Flow

```
┌──────────┐
│  Webcam  │────┐
└──────────┘    │
                ├─→ [Frame Buffer] ──→ Face Detection ──→ CNN ──┐
┌──────────┐    │                                                │
│   Mic    │────┘                                                │
└──────────┘                                                     │
                └─→ [Audio Buffer] ──→ MFCC Extract ──→ RF ──┘  │
                                                                 │
                                                                 ▼
                                                        ┌────────────────┐
                                                        │ Fusion Engine  │
                                                        │ 60% + 40%      │
                                                        └────────┬───────┘
                                                                 │
                                                                 ▼
                                                        ┌────────────────┐
                                                        │ Rules Engine   │
                                                        │ + Thresholds   │
                                                        └────────┬───────┘
                                                                 │
                                         ┌───────────────────────┴──────────────┐
                                         │                                       │
                                         ▼                                       ▼
                                  ┌─────────────┐                        ┌─────────────┐
                                  │ Employee UI │                        │  Database   │
                                  │ (Real-time) │                        │  (Storage)  │
                                  └─────────────┘                        └─────────────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │  Admin UI   │
                                  │ (Aggregated)│
                                  └─────────────┘
```

## Module Dependencies

```
app.py (Flask Backend)
  ├── config.py (Configuration)
  ├── modules/
  │   ├── facial_recognition.py
  │   │   └── tensorflow, keras, opencv
  │   ├── speech_recognition.py
  │   │   └── librosa, sklearn, numpy
  │   ├── multimodal_fusion.py
  │   │   ├── facial_recognition
  │   │   └── speech_recognition
  │   ├── rules_engine.py
  │   │   └── config
  │   ├── realtime_processor.py
  │   │   ├── facial_recognition
  │   │   ├── speech_recognition
  │   │   ├── multimodal_fusion
  │   │   └── rules_engine
  │   └── database.py
  │       └── sqlalchemy
  └── templates/
      └── index.html (Frontend)
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   FACIAL MODEL TRAINING                     │
│  (train_facial_model.py)                                    │
│                                                              │
│  Dataset: FER2013 (or custom)                               │
│  ┌────────────┐                                             │
│  │  fer2013   │──→ Load & Parse ──→ Preprocess ──┐         │
│  │  .csv      │                                    │         │
│  └────────────┘                                    │         │
│                                                     ▼         │
│                                            ┌────────────────┐│
│                                            │  CNN Training  ││
│                                            │  - Augment     ││
│                                            │  - Epochs: 50  ││
│                                            │  - Callbacks   ││
│                                            └────────┬───────┘│
│                                                     │         │
│                                                     ▼         │
│                                            ┌────────────────┐│
│                                            │   Evaluate     ││
│                                            │  - Accuracy    ││
│                                            │  - Conf Matrix ││
│                                            └────────┬───────┘│
│                                                     │         │
│                                                     ▼         │
│                                            ┌────────────────┐│
│                                            │  Save Model    ││
│                                            │  .h5 file      ││
│                                            └────────────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   SPEECH MODEL TRAINING                     │
│  (train_speech_model.py)                                    │
│                                                              │
│  Dataset: Audio files (organized by stress level)          │
│  ┌────────────┐                                             │
│  │ audio_data/│──→ Load WAV ──→ Extract Features ──┐       │
│  │  low/      │                 - MFCCs             │       │
│  │  medium/   │                 - Pitch             │       │
│  │  high/     │                 - Energy            │       │
│  └────────────┘                                     │       │
│                                                     ▼       │
│                                            ┌────────────────┐│
│                                            │ RF Training    ││
│                                            │ - 100 trees    ││
│                                            │ - CV: 5-fold   ││
│                                            └────────┬───────┘│
│                                                     │         │
│                                                     ▼         │
│                                            ┌────────────────┐│
│                                            │   Evaluate     ││
│                                            │  - Accuracy    ││
│                                            │  - Feature Imp ││
│                                            └────────┬───────┘│
│                                                     │         │
│                                                     ▼         │
│                                            ┌────────────────┐│
│                                            │  Save Model    ││
│                                            │  .pkl file     ││
│                                            └────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Privacy & Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRIVACY LAYERS                          │
│                                                                  │
│  INPUT                    PROCESSING              OUTPUT        │
│  ┌────────┐              ┌─────────┐            ┌─────────┐    │
│  │ Video  │──────────→   │ Extract │──────→     │ Metrics │    │
│  │ Stream │   (discard)  │ Features│  (keep)    │  Only   │    │
│  └────────┘              └─────────┘            └─────────┘    │
│                                                                  │
│  ┌────────┐              ┌─────────┐            ┌─────────┐    │
│  │ Audio  │──────────→   │ Extract │──────→     │ Metrics │    │
│  │ Stream │   (discard)  │ Features│  (keep)    │  Only   │    │
│  └────────┘              └─────────┘            └─────────┘    │
│                                                                  │
│  EMPLOYEE VIEW:           ADMIN VIEW:                           │
│  - Personal data          - Aggregated only                     │
│  - Full history           - No individual IDs                   │
│  - Recommendations        - Team-level trends                   │
│                           - Anonymization threshold: 3+         │
│                                                                  │
│  DATA RETENTION:          STORAGE:                              │
│  - 90 days maximum        - SQLite (local)                      │
│  - Auto-cleanup           - Encrypted at rest (optional)        │
│  - Consent required       - No raw video/audio                  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Specifications

```
┌────────────────────────────────────────────────────────────┐
│                    LATENCY TARGETS                          │
│  ┌──────────────────────────┬──────────────────────────┐   │
│  │ Component                │ Target Latency           │   │
│  ├──────────────────────────┼──────────────────────────┤   │
│  │ Face Detection           │ < 100ms                  │   │
│  │ Facial Inference (CNN)   │ < 50ms                   │   │
│  │ Speech Feature Extract   │ ~ 500ms                  │   │
│  │ Speech Inference (RF)    │ < 10ms                   │   │
│  │ Multimodal Fusion        │ < 5ms                    │   │
│  │ Database Write           │ < 100ms                  │   │
│  ├──────────────────────────┼──────────────────────────┤   │
│  │ TOTAL END-TO-END         │ 2-5 seconds              │   │
│  └──────────────────────────┴──────────────────────────┘   │
│                                                              │
│  Note: Academically acceptable real-time performance        │
│  (Not instant, but near real-time)                          │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                    ACCURACY TARGETS                         │
│  ┌──────────────────────────┬──────────────────────────┐   │
│  │ Model                    │ Expected Accuracy        │   │
│  ├──────────────────────────┼──────────────────────────┤   │
│  │ Facial Emotion (FER2013) │ 60-65%                   │   │
│  │ Speech Stress (Custom)   │ 70-80%                   │   │
│  │ Multimodal Fusion        │ Improved reliability     │   │
│  └──────────────────────────┴──────────────────────────┘   │
│                                                              │
│  Focus: Trend analysis over time, not single predictions    │
└────────────────────────────────────────────────────────────┘
```

## Deployment Architecture Options

```
┌─────────────────────────────────────────────────────────────┐
│  OPTION 1: Localhost (Development)                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Browser ← http://localhost:5000 ← Flask (app.py)  │     │
│  │                                   ↓                 │     │
│  │                            SQLite (local file)      │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  OPTION 2: Server + Nginx (Production)                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Browser ← HTTPS ← Nginx ← Gunicorn ← Flask        │     │
│  │                                      ↓              │     │
│  │                               PostgreSQL/MySQL      │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  OPTION 3: Docker Container                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │  Docker Container                        │      │     │
│  │  │  - Python 3.9                            │      │     │
│  │  │  - All dependencies                      │      │     │
│  │  │  - Flask app                             │      │     │
│  │  │  - SQLite (volume mount)                 │      │     │
│  │  └──────────────────────────────────────────┘      │     │
│  │           ↑                                         │     │
│  │  Port mapping: 5000:5000                           │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

**This architecture supports academic requirements while being production-ready.**
