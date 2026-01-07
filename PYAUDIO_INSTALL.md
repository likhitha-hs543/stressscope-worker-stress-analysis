# PyAudio Installation Guide for Windows

PyAudio requires the PortAudio library, which can be challenging to install on Windows.

## Quick Installation (Recommended)

### Method 1: Using pipwin (Easiest)

```bash
# Install pipwin (a tool for installing pre-compiled Windows binaries)
pip install pipwin

# Install PyAudio using pipwin
pipwin install pyaudio
```

### Method 2: Using Unofficial Windows Binaries

1. **Download the wheel file** from Christoph Gohlke's repository:
   - Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Download the appropriate `.whl` file for your Python version:
     - `cp39` = Python 3.9
     - `cp310` = Python 3.10
     - `cp311` = Python 3.11
     - `win_amd64` = 64-bit Windows
     - `win32` = 32-bit Windows
   
   Example: `PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl` for Python 3.11 (64-bit)

2. **Install the wheel**:
   ```bash
   pip install path\to\downloaded\PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl
   ```

### Method 3: Build from Source (Advanced)

Only if the above methods fail:

1. **Install Microsoft C++ Build Tools**:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

2. **Install PortAudio**:
   ```bash
   # Using conda (if you have Anaconda)
   conda install portaudio
   
   # Or download pre-built binaries
   ```

3. **Build PyAudio**:
   ```bash
   pip install pyaudio
   ```

## Verification

Test if PyAudio is installed correctly:

```bash
python -c "import pyaudio; print('PyAudio version:', pyaudio.__version__)"
```

Expected output:
```
PyAudio version: 0.2.14
```

## Troubleshooting

### Error: "Microsoft Visual C++ 14.0 is required"
- Use Method 1 (pipwin) or Method 2 (wheel file)
- These methods bypass the need for compilation

### Error: "No module named 'pyaudio'"
- Ensure you're using the correct Python environment
- Try: `pip list | findstr pyaudio`

### Error: "DLL load failed"
- Install Microsoft Visual C++ Redistributable:
  - https://aka.ms/vs/17/release/vc_redist.x64.exe

### Still not working?
- Check Python version: `python --version`
- Match the wheel file to your exact Python version
- Try Method 1 (pipwin) - it's the most reliable

## Alternative: Skip Audio Analysis

If you can't install PyAudio, you can still use the system with **facial analysis only**:

1. In `app.py`, comment out audio imports:
   ```python
   # from modules.speech_recognition import SpeechStressRecognizer
   ```

2. Modify analysis to use facial-only mode
   - The system will gracefully handle missing speech analysis

## Linux/Mac Users

PyAudio is much easier to install on Unix systems:

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

---

## Need Help?

- Check Python version compatibility
- Ensure you're using 64-bit Python if on 64-bit Windows
- Try creating a fresh virtual environment
- Contact support with your exact error message
