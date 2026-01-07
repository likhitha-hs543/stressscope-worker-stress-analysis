"""
Generate Demo/Test Data
Use ONLY for testing workflow - not production!
Creates small synthetic datasets to test training pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_demo_fer2013():
    """Generate small demo FER2013-like dataset"""
    logger.info("Generating demo FER2013 dataset...")
    
    # Small dataset for quick testing
    n_samples = 300  # 50 per emotion
    emotions = 6  # Angry, Disgust, Fear, Happy, Sad, Surprise
    
    data = []
    
    for usage, count in [('Training', 200), ('PublicTest', 50), ('PrivateTest', 50)]:
        for _ in range(count):
            # Random 48x48 grayscale image
            pixels = np.random.randint(0, 256, 48 * 48)
            pixels_str = ' '.join(map(str, pixels))
            emotion = np.random.randint(0, emotions)
            
            data.append({
                'emotion': emotion,
                'pixels': pixels_str,
                'Usage': usage
            })
    
    df = pd.DataFrame(data)
    
    # Save
    output_dir = Path('data/demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'fer2013_demo.csv'
    
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Demo FER2013 saved to {output_path} ({len(df)} samples)")
    
    return output_path


def generate_demo_speech_data():
    """Generate small demo speech dataset"""
    logger.info("Generating demo speech dataset...")
    
    output_dir = Path('data/demo/speech_demo')
    sample_rate = 16000
    duration = 3  # 3 seconds per sample
    
    # Create directories
    for level in ['low', 'medium', 'high']:
        (output_dir / level).mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    samples_per_class = 20
    
    for level in ['low', 'medium', 'high']:
        for i in range(samples_per_class):
            # Generate random audio (white noise with different characteristics)
            if level == 'low':
                audio = np.random.normal(0, 0.1, sample_rate * duration)
            elif level == 'medium':
                audio = np.random.normal(0, 0.2, sample_rate * duration)
            else:  # high
                audio = np.random.normal(0, 0.3, sample_rate * duration)
            
            # Add some simple frequency variation
            t = np.linspace(0, duration, sample_rate * duration)
            if level == 'high':
                audio += 0.1 * np.sin(2 * np.pi * 300 * t)  # Higher pitch
            else:
                audio += 0.1 * np.sin(2 * np.pi * 150 * t)  # Lower pitch
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save
            filename = output_dir / level / f'sample_{i:03d}.wav'
            sf.write(filename, audio.astype(np.float32), sample_rate)
        
        logger.info(f"✅ Generated {samples_per_class} samples for '{level}' stress")
    
    total_samples = samples_per_class * 3
    logger.info(f"✅ Demo speech dataset saved to {output_dir} ({total_samples} samples)")
    
    return output_dir


def main():
    """Generate all demo data"""
    print("="*60)
    print("DEMO DATA GENERATION")
    print("="*60)
    print("\n⚠️  WARNING: This data is SYNTHETIC and RANDOM!")
    print("   - DO NOT use for production")
    print("   - Only for testing workflow")
    print("   - Accuracy will be ~30-40% (random guessing)")
    print("\n" + "="*60 + "\n")
    
    # Generate
    fer_path = generate_demo_fer2013()
    speech_dir = generate_demo_speech_data()
    
    print("\n" + "="*60)
    print("✅ DEMO DATA GENERATED")
    print("="*60)
    print(f"\nFacial: {fer_path}")
    print(f"Speech: {speech_dir}")
    print("\nYou can now train models with:")
    print(f"  python train_facial_optimized.py --data {fer_path} --epochs 10")
    print(f"  python train_speech_optimized.py --data-dir {speech_dir} --method ensemble")
    print("\n⚠️  Remember: Get real datasets for production!")
    print("="*60)


if __name__ == "__main__":
    main()
