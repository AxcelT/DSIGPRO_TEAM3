from pyannote.audio import Pipeline
from pyannote.audio import Model
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch

# Replace with your verified Hugging Face access token
ACCESS_TOKEN = "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" #this is a placeholder btw not the actual token

def log_message(message):
    """Helper function to log messages."""
    print(f"[INFO] {message}")

# Step 1: Data Preprocessing
def preprocess_audio(file_path):
    """Convert audio to mono and resample to 16kHz."""
    log_message(f"Loading and preprocessing audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=None)  # Load audio with original sampling rate
    log_message(f"Original sampling rate: {sr}")
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    log_message(f"Resampled audio to 16kHz and converted to mono.")
    return audio_resampled, 16000

# Step 2: Initialize Models
log_message("Loading segmentation model: pyannote/segmentation-3.0")
segmentation_model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=ACCESS_TOKEN
)
log_message("Segmentation model loaded successfully.")

log_message("Initializing speaker diarization pipeline: pyannote/speaker-diarization-3.1")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=ACCESS_TOKEN
)
log_message("Pipeline initialized successfully.")

# Check for GPU and move the pipeline to GPU if available
if torch.cuda.is_available():
    log_message("GPU detected. Moving pipeline to GPU.")
    pipeline.to(torch.device("cuda"))
else:
    log_message("No GPU detected. Using CPU.")

# Step 3: Speaker Diarization and Segment Filtering
def detect_short_segments(file_path, max_duration=2.0):
    """Identify short speech segments (< max_duration)."""
    log_message("Preprocessing audio for diarization...")
    audio, sr = preprocess_audio(file_path)
    
    log_message("Running diarization pipeline...")
    diarization = pipeline({"waveform": torch.tensor(audio).unsqueeze(0), "sample_rate": sr})

    short_segments = []
    log_message("Analyzing diarization segments...")
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        duration = segment.end - segment.start
        log_message(f"Segment detected - Start: {segment.start:.2f}s, End: {segment.end:.2f}s, Duration: {duration:.2f}s, Speaker: {speaker}")
        if duration <= max_duration:
            short_segments.append((segment.start, segment.end, speaker))
            log_message(f"Short segment added - Start: {segment.start:.2f}s, End: {segment.end:.2f}s, Speaker: {speaker}")
        else:
            log_message(f"Segment ignored (duration > {max_duration}s) - Duration: {duration:.2f}s, Speaker: {speaker}")
    return short_segments

# Step 4: Feature Extraction
def extract_features(audio, sr, segment_start, segment_end):
    """Extract features from an audio segment."""
    start_sample = int(segment_start * sr)
    end_sample = int(segment_end * sr)
    segment = audio[start_sample:end_sample]

    log_message("Extracting features from segment...")
    features = {
        "duration": segment_end - segment_start,
        "rms": np.sqrt(np.mean(segment**2)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(segment)),
    }
    log_message(f"Extracted features: {features}")
    return features

# Step 5: Classification
def classify_segment(features):
    """Classify segment based on heuristic thresholds."""
    if features["rms"] > 0.05 and features["zero_crossing_rate"] > 0.1:
        if features["duration"] < 1.0:
            return "Exclamation"
        else:
            return "Response"
    elif features["rms"] < 0.01:
        return "Noise"
    else:
        return "Unknown"

# Main Function
if __name__ == "__main__":
    audio_file = r"audio\Hardware Implementation.wav"
    log_message("Starting short speech segment detection...")

    # Detect short segments
    short_segments = detect_short_segments(audio_file)
    log_message(f"Detected {len(short_segments)} short segments.")

    if short_segments:
        log_message("Processing short segments...")
        audio, sr = preprocess_audio(audio_file)
        for start, end, speaker in short_segments:
            log_message(f"Processing segment - Start: {start:.2f}s, End: {end:.2f}s, Speaker: {speaker}")
            features = extract_features(audio, sr, start, end)
            classification = classify_segment(features)
            log_message(f"Segment classification: {classification}")
            print(f"[RESULT] Speaker {speaker} from {start:.2f}s to {end:.2f}s classified as {classification}.")
    else:
        log_message("No short segments detected.")

    log_message("Script execution completed.")
