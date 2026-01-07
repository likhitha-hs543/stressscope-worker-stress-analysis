# 📊 Data Directory

This directory contains **training datasets** for the Worker Stress Analysis System.

---

## ⚠️ Datasets Not Included

Due to size limitations and licensing, **datasets are not committed to this repository**.

You must download them separately.

---

## 📁 Required Datasets

### 1. FER2013 (Facial Emotions)

**Location:** `face/FER 2013/`

**Download From:**
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- Alternative: https://www.kaggle.com/datasets/ashishpatel26/fer2013

**Expected Structure:**
```
face/FER 2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

**Details:**
- Format: 48×48 grayscale images
- Total: ~35,000 images
- Classes: 7 emotions

---

### 2. RAVDESS (Speech Emotions)

**Location:** `speech/SER/Ravdess/`

**Download From:**
- Zenodo: https://zenodo.org/record/1188976
- Kaggle: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio

**Preprocessing Required:**
```bash
python prepare_ravdess_data.py --input-dir "data/speech/SER/Ravdess" --output-dir "data/speech/SER/prepared"
```

**Expected Structure (After Preprocessing):**
```
speech/SER/prepared/
├── low/        # Calm, neutral emotions
├── medium/     # Moderate emotions
└── high/       # Angry, fear, sad
```

**Details:**
- Format: 16kHz WAV audio files
- Total: 1,440 files
- Duration: 3-4 seconds each
- Speakers: 24 professional actors

---

## 🔒 .gitignore Configuration

The following are **automatically ignored** (will not be committed):

```
data/raw_videos/
data/raw_audio/
*.wav
*.mp4
*.avi
```

This ensures:
- Privacy (no raw media)
- Repository stays lightweight
- Compliance with licensing

---

## ✅ Verification

After downloading datasets, verify structure:

```bash
# Check facial dataset
dir data\face\FER 2013\train
dir data\face\FER 2013\test

# Check speech dataset (after preparation)
dir data\speech\SER\prepared
```

Each should show 7 (facial) or 3 (speech) subdirectories.

---

## 💡 Tips

- **Facial:** Keep images as-is (48×48 grayscale)
- **Speech:** Run preprocessing script before training
- **Storage:** ~2GB total for both datasets
- **License:** Check dataset sources for usage terms
