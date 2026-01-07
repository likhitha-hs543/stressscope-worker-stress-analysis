"""
Prepare RAVDESS Dataset for Stress Training
Organizes RAVDESS audio files into low/medium/high stress categories
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract emotion
    
    Format: 03-01-XX-01-01-01-24.wav
    Position 3 (XX) = Emotion:
    01 = neutral, 02 = calm
    03 = happy, 04 = sad
    05 = angry, 06 = fearful
    07 = disgust, 08 = surprised
    """
    parts = filename.stem.split('-')
    if len(parts) < 3:
        return None
    
    emotion_code = int(parts[2])
    
    # Map to stress levels
    stress_mapping = {
        1: 'low',    # neutral
        2: 'low',    # calm
        3: 'low',    # happy
        4: 'medium', # sad
        5: 'high',   # angry
        6: 'high',   # fearful
        7: 'medium', # disgust
        8: 'medium'  # surprised
    }
    
    return stress_mapping.get(emotion_code)


def organize_ravdess_data(input_dir, output_dir):
    """Organize RAVDESS files into stress categories"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for level in ['low', 'medium', 'high']:
        (output_path / level).mkdir(parents=True, exist_ok=True)
    
    # Process all WAV files
    wav_files = list(input_path.rglob('*.wav'))
    logger.info(f"Found {len(wav_files)} WAV files in {input_dir}")
    
    counts = {'low': 0, 'medium': 0, 'high': 0}
    skipped = 0
    
    for wav_file in wav_files:
        stress_level = parse_ravdess_filename(wav_file)
        
        if stress_level:
            # Copy file to appropriate directory
            dest = output_path / stress_level / wav_file.name
            shutil.copy2(wav_file, dest)
            counts[stress_level] += 1
        else:
            skipped += 1
    
    # Print summary
    print("\n" + "="*60)
    print("RAVDESS DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\n📁 Output directory: {output_path}")
    print(f"\n📊 File distribution:")
    print(f"  Low stress:    {counts['low']} files")
    print(f"  Medium stress: {counts['medium']} files")
    print(f"  High stress:   {counts['high']} files")
    print(f"  Skipped:       {skipped} files")
    print(f"\n✅ Total organized: {sum(counts.values())} files")
    print("="*60 + "\n")
    
    return counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare RAVDESS data for stress training')
    parser.add_argument('--input-dir', type=str,
                       default='data/speech/SER/Ravdess',
                       help='Input RAVDESS directory')
    parser.add_argument('--output-dir', type=str,
                       default='data/speech/SER/prepared',
                       help='Output directory for organized files')
    
    args = parser.parse_args()
    
    counts = organize_ravdess_data(args.input_dir, args.output_dir)
    
    print("Next step:")
    print(f"  python train_speech_from_ravdess.py --data-dir \"{args.output_dir}\"")
