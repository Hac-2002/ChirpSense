import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import soundfile as sf
from pathlib import Path

# Configuration variables
INPUT_PATH = os.path.join("data", "raw", "train_audio")
OUTPUT_BASE_PATH = os.path.join("data", "processed", "zcr")

# Audio processing parameters
SAMPLE_RATE = 22050
FIXED_LENGTH = 30
FRAME_LENGTH = 2048
HOP_LENGTH = 512
BATCH_SIZE = 5

def create_zcr(audio_path, npy_path, img_path, filename, fixed_length=FIXED_LENGTH, sr=SAMPLE_RATE):
    """
    Create and save zero crossing rate with fixed dimensions for 30 seconds
    
    Parameters:
    - audio_path: path to audio file
    - npy_path: path to save numpy arrays
    - img_path: path to save visualization images
    - filename: name of the file
    - fixed_length: duration in seconds to process (default 30 seconds)
    - sr: sample rate (default 22050 Hz)
    """
    try:
        # Calculate fixed number of samples for the desired duration
        fixed_samples = int(fixed_length * sr)
        
        # Load and process audio
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Handle audio length
        if len(y) < fixed_samples:
            pad_length = fixed_samples - len(y)
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            y = np.pad(y, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            start_idx = (len(y) - fixed_samples) // 2
            y = y[start_idx:start_idx + fixed_samples]
        
        # Calculate Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            center=True
        )
        
        # Normalize the ZCR
        zcr_normalized = (zcr - zcr.min()) / (zcr.max() - zcr.min())
        
        # Save the raw ZCR data
        npy_save_path = Path(npy_path) / os.path.dirname(filename)
        npy_save_path.mkdir(parents=True, exist_ok=True)
        np.save(npy_save_path / f"{Path(filename).name}.npy", zcr_normalized)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Create time array for x-axis
        times = librosa.times_like(zcr, sr=sr, hop_length=HOP_LENGTH)
        
        # Plot ZCR
        plt.plot(times, zcr[0], label='Zero Crossing Rate', color='blue', linewidth=1)
        plt.fill_between(times, zcr[0], alpha=0.5, color='blue')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Zero Crossing Rate')
        plt.title(f'Zero Crossing Rate - Duration: {len(y)/sr:.2f}s')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save visualization
        img_save_path = Path(img_path) / os.path.dirname(filename)
        img_save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(img_save_path / f"{Path(filename).name}.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Clear memory
        del y, zcr, zcr_normalized
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        plt.close('all')
        gc.collect()

def process_dataset(input_path, output_base_path, batch_size=BATCH_SIZE):
    """
    Process all audio files in the dataset with batched processing
    """
    # Create separate directories for numpy arrays and images
    npy_path = Path(output_base_path) / 'numpy_features'
    img_path = Path(output_base_path) / 'visualizations'
    
    npy_path.mkdir(parents=True, exist_ok=True)
    img_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Numpy arrays will be saved to: {npy_path}")
    print(f"Visualizations will be saved to: {img_path}")
    
    # Get all MP3 files recursively
    audio_files = list(Path(input_path).rglob("*.mp3"))
    print(f"Found {len(audio_files)} audio files")
    
    # Calculate and display expected dimensions
    n_frames = int(np.ceil(FIXED_LENGTH * SAMPLE_RATE / HOP_LENGTH))
    print(f"\nExpected ZCR dimensions for each file:")
    print(f"Time steps: {n_frames}")
    print(f"Final shape: (1, {n_frames})")
    
    # Track audio lengths for analysis
    audio_lengths = []
    
    # Process files in batches
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(audio_files)//batch_size + 1}")
        
        for audio_file in tqdm(batch):
            rel_path = audio_file.relative_to(input_path)
            filename = rel_path.with_suffix('')
            
            try:
                with sf.SoundFile(audio_file) as f:
                    duration = len(f) / f.samplerate
                    audio_lengths.append(duration)
            except Exception as e:
                print(f"Could not get duration for {audio_file}: {str(e)}")
            
            create_zcr(audio_file, npy_path, img_path, filename)
            plt.close('all')
            gc.collect()
        
        gc.collect()
    
    if audio_lengths:
        audio_lengths = np.array(audio_lengths)
        print("\nAudio Length Statistics:")
        print(f"Mean duration: {np.mean(audio_lengths):.2f} seconds")
        print(f"Median duration: {np.median(audio_lengths):.2f} seconds")
        print(f"Min duration: {np.min(audio_lengths):.2f} seconds")
        print(f"Max duration: {np.max(audio_lengths):.2f} seconds")
        print(f"Files < {FIXED_LENGTH}s: {np.sum(audio_lengths < FIXED_LENGTH)}")
        print(f"Files > {FIXED_LENGTH}s: {np.sum(audio_lengths > FIXED_LENGTH)}")

if __name__ == "__main__":
    # Disable librosa cache
    os.environ['LIBROSA_CACHE_DIR'] = ''
    
    # Configure matplotlib to use Agg backend
    plt.switch_backend('Agg')
    
    try:
        process_dataset(INPUT_PATH, OUTPUT_BASE_PATH, batch_size=BATCH_SIZE)
    except MemoryError:
        print("\nMemory error occurred. Retrying with single file processing...")
        process_dataset(INPUT_PATH, OUTPUT_BASE_PATH, batch_size=1)