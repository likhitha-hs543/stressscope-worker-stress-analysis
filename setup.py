"""
Setup Script for Worker Stress Analysis System
Automates initial setup and configuration
"""

import os
import sys
from pathlib import Path
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def create_directories():
    """Create required directories"""
    print("Creating directories...")
    
    directories = [
        'models/trained',
        'data/processed',
        'data/raw_videos',
        'data/raw_audio',
        'logs',
        'templates',
        'static/css',
        'static/js',
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    print("This may take several minutes...\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("\n✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        print("   Try manually: pip install -r requirements.txt")
        return False

def initialize_database():
    """Initialize SQLite database"""
    print("Initializing database...")
    
    try:
        from modules.database import init_database
        engine = init_database()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def download_haar_cascade():
    """Download Haar Cascade file if not present"""
    print("Checking Haar Cascade classifier...")
    
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if os.path.exists(cascade_path):
            print("✅ Haar Cascade found")
            return True
        else:
            print("❌ Haar Cascade not found")
            print("   OpenCV should include it by default")
            return False
    except Exception as e:
        print(f"❌ Error checking Haar Cascade: {e}")
        return False

def create_env_file():
    """Create .env file with default configuration"""
    print("Creating .env file...")
    
    env_content = """# Worker Stress Analysis System Configuration

# Database
DATABASE_URL=sqlite:///stress_analysis.db

# Flask
SECRET_KEY=dev-secret-key-change-in-production
DEBUG=True
HOST=0.0.0.0
PORT=5000

# Privacy Settings
STORE_RAW_VIDEO=False
STORE_RAW_AUDIO=False
DATA_RETENTION_DAYS=90

# Model Settings
FACIAL_MODEL_PATH=models/trained/facial_emotion_model.h5
SPEECH_MODEL_PATH=models/trained/speech_stress_model.pkl
"""
    
    env_path = Path('.env')
    
    if env_path.exists():
        print("  ℹ️  .env file already exists (not overwriting)")
        return True
    
    try:
        env_path.write_text(env_content)
        print("✅ .env file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_imports():
    """Test if critical modules can be imported"""
    print("Testing imports...")
    
    modules = [
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('sklearn', 'Scikit-learn'),
        ('librosa', 'Librosa'),
        ('flask', 'Flask'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]
    
    failed = []
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name}")
            failed.append(display_name)
    
    if failed:
        print(f"\n⚠️  Failed imports: {', '.join(failed)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def create_test_employee():
    """Create test employee in database"""
    print("Creating test employee...")
    
    try:
        from modules.database import get_session, Employee
        
        db_session = get_session()
        
        # Check if test employee exists
        existing = db_session.query(Employee).filter_by(employee_id='test_001').first()
        
        if existing:
            print("  ℹ️  Test employee already exists")
        else:
            # Create test employee
            test_employee = Employee(
                employee_id='test_001',
                name='Test User',
                role='employee',
                department='Testing',
                consent_given=True
            )
            db_session.add(test_employee)
            db_session.commit()
            print("✅ Test employee created (ID: test_001)")
        
        db_session.close()
        return True
    except Exception as e:
        print(f"❌ Failed to create test employee: {e}")
        return False

def print_next_steps():
    """Print next steps for user"""
    print_header("🎉 Setup Complete!")
    
    print("Next steps:\n")
    
    print("1️⃣  Start the application:")
    print("   python app.py")
    print()
    
    print("2️⃣  Open your browser:")
    print("   http://localhost:5000")
    print()
    
    print("3️⃣  Train models (optional):")
    print("   python train_facial_model.py")
    print("   python train_speech_model.py")
    print()
    
    print("4️⃣  Test real-time processing:")
    print("   python modules/realtime_processor.py")
    print()
    
    print("📚 Documentation:")
    print("   - README.md - Comprehensive guide")
    print("   - API_DOCUMENTATION.md - API reference")
    print()
    
    print("⚠️  Notes:")
    print("   - Grant webcam/microphone permissions when prompted")
    print("   - Models may need training for production use")
    print("   - See README.md for detailed setup and usage")
    print()

def main():
    """Main setup routine"""
    print_header("Worker Stress Analysis System - Setup")
    
    print("This script will set up your environment.\n")
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Directories", create_directories),
        ("Environment File", create_env_file),
        ("Dependencies", install_dependencies),
        ("Imports Test", test_imports),
        ("Database", initialize_database),
        ("Haar Cascade", download_haar_cascade),
        ("Test Employee", create_test_employee),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print_header(f"Step: {step_name}")
        
        try:
            success = step_func()
            if not success:
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print_header("Setup Summary")
    
    if failed_steps:
        print("⚠️  Setup completed with warnings\n")
        print("Failed steps:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nYou may need to fix these manually.")
        print("Check README.md for troubleshooting.\n")
    else:
        print("✅ All steps completed successfully!\n")
    
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)
