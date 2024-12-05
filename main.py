import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from pyannote.audio import Pipeline, Model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Replace with your verified Hugging Face access token
ACCESS_TOKEN = "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"

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

# Step 3: Read CSV Segments
def read_csv_segments(csv_file):
    """Read segments from the provided CSV file."""
    short_segments = []
    all_segments = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            start_time, end_time, speaker, transcript = row
            start_time, end_time = float(start_time), float(end_time)
            duration = end_time - start_time
            segment = {"start": start_time, "end": end_time, "speaker": speaker, "duration": duration, "transcript": transcript}
            all_segments.append(segment)
            if duration < 2.0:  # Short segment threshold
                short_segments.append(segment)
    return short_segments, all_segments

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

# Step 6: Evaluation Metrics
def evaluate_metrics(ground_truth, predictions):
    """Compute precision, recall, and F1 score."""
    y_true = [segment["speaker"] for segment in ground_truth]
    y_pred = [prediction["speaker"] for prediction in predictions]
    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")
    return precision, recall, f1

# Main Function
if __name__ == "__main__":
    csv_file = "Frasier_02x01.csv"
    audio_file = r"audio\Hardware Implementation.wav"
    
    # Step 1: Read Ground Truth Short Segments
    log_message("Reading CSV data...")
    short_segments_gt, all_segments_gt = read_csv_segments(csv_file)
    log_message(f"Loaded {len(short_segments_gt)} ground truth short segments.")

    # Step 2: Preprocess Audio
    log_message("Preprocessing audio...")
    audio, sr = preprocess_audio(audio_file)

    # Step 3: Detect Short Segments
    log_message("Detecting short segments using diarization pipeline...")
    diarization = pipeline({"waveform": torch.tensor(audio).unsqueeze(0), "sample_rate": sr})
    detected_segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if segment.end - segment.start < 2.0:
            detected_segments.append({"start": segment.start, "end": segment.end, "speaker": speaker})

    # Step 4: Evaluate Results
    precision, recall, f1 = evaluate_metrics(short_segments_gt, detected_segments)
    log_message(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    # Step 5: Visualize Results
    short_durations_gt = [s["duration"] for s in short_segments_gt]
    short_durations_detected = [s["end"] - s["start"] for s in detected_segments]
    all_durations = [s["duration"] for s in all_segments_gt]

    plt.hist(short_durations_gt, bins=20, alpha=0.5, label="Ground Truth Short Segments")
    plt.hist(short_durations_detected, bins=20, alpha=0.5, label="Detected Short Segments")
    plt.hist(all_durations, bins=20, alpha=0.5, label="All Segments")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.title("Segment Duration Comparison")
    plt.show()

    log_message("Script execution completed.")
