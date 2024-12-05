import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def log_message(message):
    """Helper function to log messages."""
    print(f"[INFO] {message}")

def analyze_audio(file_path):
    """Analyze the given audio file and extract properties."""
    try:
        log_message(f"Loading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=None, mono=False)  # Load audio without resampling
        duration = librosa.get_duration(y=audio, sr=sr)
        num_channels = 1 if len(audio.shape) == 1 else audio.shape[0]

        log_message(f"Sample Rate: {sr} Hz")
        log_message(f"Duration: {duration:.2f} seconds")
        log_message(f"Number of Channels: {num_channels}")

        # Calculate energy
        if num_channels > 1:
            audio = np.mean(audio, axis=0)  # Average channels for stereo
        energy = np.square(audio)

        log_message("Plotting energy distribution...")
        times = np.arange(len(energy)) / sr
        plt.figure(figsize=(12, 6))
        plt.plot(times, energy, label="Energy")
        plt.title("Energy Distribution of Audio File")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.legend()
        plt.savefig("audio_energy_curve.png")
        log_message("Energy curve saved as 'audio_energy_curve.png'.")

    except Exception as e:
        log_message(f"Error analyzing audio: {e}")

if __name__ == "__main__":
    audio_file_path = "audio\Frasier1.wav"  # Example path
    analyze_audio(audio_file_path)
