# 🧹 Project Cleanup - Archived Scripts

This folder contains **older or redundant** training scripts that have been replaced by better implementations.

## Why These Were Archived

### Facial Training Scripts (Replaced by `train_facial_mobilenetv2.py`)
- **`train_facial_from_images.py`** - Early implementation, no transfer learning
- **`train_facial_model.py`** - Basic training pipeline, replaced by optimized version
- **`train_facial_optimized.py`** - Improved but still scratch training (13% accuracy)

**Current Best:** `train_facial_mobilenetv2.py` (50.17% accuracy using transfer learning)

### Speech Training Scripts (Replaced by `train_speech_from_ravdess.py`)
- **`train_speech_model.py`** - Generic speech training
- **`train_speech_optimized.py`** - Improved but not dataset-specific

**Current Best:** `train_speech_from_ravdess.py` (69.91% accuracy with ensemble)

### Utilities
- **`generate_demo_data.py`** - Demo data generator (not needed with real datasets)

---

## Current Production Scripts

Use these for all training:

```bash
# Facial emotion training (MobileNetV2 transfer learning)
python train_facial_mobilenetv2.py --data-dir "data/face/FER 2013" --phase1-epochs 15 --phase2-epochs 10

# Speech stress training (Ensemble with RAVDESS)
python prepare_ravdess_data.py  # First organize data
python train_speech_from_ravdess.py --data-dir "data/speech/SER/prepared"
```

---

**These archived scripts are kept for reference but should NOT be used for training.**
