import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import soundfile as sf

# Configuration variables for paths
# These should be modified by users according to their setup
INPUT_PATH = os.path.join("data", "raw", "train_audio")  # Path to input audio files
OUTPUT_BASE_PATH = os.path.join("data", "processed", "mel_spectrograms")  # Path for output files

def create_mel_spectrogram(audio_path, npy_path, img_path, filename, fixed_length=30, sr=22050):
    """
    Create and save mel spectrogram with fixed dimensions for 30 seconds
    
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
            # For shorter audio, center it in the 30-second window
            pad_length = fixed_samples - len(y)
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            y = np.pad(y, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            # If audio is longer, take the middle 30 seconds
            start_idx = (len(y) - fixed_samples) // 2
            y = y[start_idx:start_idx + fixed_samples]
        
        # Create mel spectrogram with fixed parameters
        mel_spect = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            power=2.0,
            fmin=20,
            fmax=sr/2
        )
        
        # Convert to log scale
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Normalize the spectrogram
        mel_spect_db = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min())
        
        # Save the raw spectrogram data for CNN training
        # Create subdirectory structure in numpy folder
        os.makedirs(os.path.join(npy_path, os.path.dirname(filename)), exist_ok=True)
        save_filename_npy = os.path.join(npy_path, f"{filename}.npy")
        np.save(save_filename_npy, mel_spect_db)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(
            mel_spect_db,
            sr=sr,
            hop_length=512,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        
        # Add duration information to the title
        original_duration = len(y) / sr
        plt.title(f'Mel Spectrogram - Duration: {original_duration:.2f}s')
        
        # Save visualization
        # Create subdirectory structure in images folder
        os.makedirs(os.path.join(img_path, os.path.dirname(filename)), exist_ok=True)
        save_filename_png = os.path.join(img_path, f"{filename}.png")
        plt.savefig(save_filename_png, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Clear memory
        del y, mel_spect, mel_spect_db
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        plt.close('all')
        gc.collect()

def process_dataset(input_path, output_base_path, batch_size=5):
    """
    Process all audio files in the dataset with batched processing
    """
    # Create separate directories for numpy arrays and images
    npy_path = os.path.join(output_base_path, 'numpy_features')
    img_path = os.path.join(output_base_path, 'visualizations')
    
    os.makedirs(npy_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    
    print(f"Numpy arrays will be saved to: {npy_path}")
    print(f"Visualizations will be saved to: {img_path}")
    
    # Get all MP3 files recursively
    audio_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Calculate and display expected dimensions
    sr = 22050
    fixed_length = 30
    n_mels = 128
    hop_length = 512
    n_time_steps = int(np.ceil(fixed_length * sr / hop_length))
    print(f"\nExpected spectrogram dimensions for each file:")
    print(f"Mel bands (frequency dimension): {n_mels}")
    print(f"Time steps: {n_time_steps}")
    print(f"Final shape: ({n_mels}, {n_time_steps})")
    
    # Track audio lengths for analysis
    audio_lengths = []
    
    # Process files in batches
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(audio_files)//batch_size + 1}")
        
        for audio_file in tqdm(batch):
            # Create output filename (preserve folder structure)
            rel_path = os.path.relpath(audio_file, input_path)
            filename = os.path.splitext(rel_path)[0]
            
            # Get audio length before processing
            try:
                with sf.SoundFile(audio_file) as f:
                    duration = len(f) / f.samplerate
                    audio_lengths.append(duration)
            except Exception as e:
                print(f"Could not get duration for {audio_file}: {str(e)}")
            
            create_mel_spectrogram(audio_file, npy_path, img_path, filename)
            
            # Force garbage collection after each file
            plt.close('all')
            gc.collect()
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Print audio length statistics
    audio_lengths = np.array(audio_lengths)
    print("\nAudio Length Statistics:")
    print(f"Mean duration: {np.mean(audio_lengths):.2f} seconds")
    print(f"Median duration: {np.median(audio_lengths):.2f} seconds")
    print(f"Min duration: {np.min(audio_lengths):.2f} seconds")
    print(f"Max duration: {np.max(audio_lengths):.2f} seconds")
    print(f"Number of files shorter than 30s: {np.sum(audio_lengths < 30)}")
    print(f"Number of files longer than 30s: {np.sum(audio_lengths > 30)}")

if __name__ == "__main__":
    # Disable librosa cache
    os.environ['LIBROSA_CACHE_DIR'] = ''
    
    # Configure matplotlib to use Agg backend
    plt.switch_backend('Agg')
    
    try:
        process_dataset(INPUT_PATH, OUTPUT_BASE_PATH, batch_size=5)
    except MemoryError:
        print("\nMemory error occurred. Retrying with single file processing...")
        process_dataset(INPUT_PATH, OUTPUT_BASE_PATH, batch_size=1)
