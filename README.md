# Audio Segmentation and Subtitling Tool

This repository implements an audio segmentation tool aimed at detecting and extracting short speech segments from audio files. The tool is aligned with the pipeline described in the paper **"Look, Listen, and Recognise: Character-Aware Audio-Visual Subtitling"** and can be extended to perform speaker diarization, subtitle generation, and character-aware audio-visual processing.

## Features

- **Audio Preprocessing**: Resamples and converts audio files to mono format with a target sampling rate (default: 16 kHz).
- **Speech Segmentation**: Detects short speech segments using energy thresholds and customizable hyperparameters.
- **Minimum Segment Duration**: Ensures that all extracted segments are at least 1 second long for meaningful processing.
- **Visualization**: Plots the RMS energy curve with detected speech segments, providing insights into audio dynamics.
- **Customizable Parameters**:
  - Energy threshold (`min_energy`)
  - Maximum segment duration (`max_duration`)
  - Minimum silence duration (`min_silence_duration`)
  - Minimum gap between segments (`min_gap`)
- **File Management**:
  - Outputs are saved in organized folders:
    - `segments`: Contains extracted speech segments as `.wav` files.
    - `plots`: Contains visualized energy curves as `.png` files.

## Requirements

- Python 3.8 or later
- Libraries:
  - `librosa`
  - `numpy`
  - `matplotlib`
  - `soundfile`

Install the dependencies using:
```
pip install librosa numpy matplotlib soundfile
```

## How to Use

1. Clone the repository:
```
git clone https://github.com/AxcelT/DSIGPRO_TEAM3.git
cd audio-segmentation
```

2. Place your input audio file in the `audio/` directory. For example:
```
audio/
  Frasier1.wav
```

3. Run the script:
```
python DSP_main.py
```

4. Outputs(Change the Output Directory made it match mine):
   - Extracted speech segments will be saved in `outputs/segments/`.
   - Energy curve plots will be saved in `outputs/plots/`.

## Configuration

You can adjust the segmentation parameters directly in the script:
- `energy_threshold`: Adjust the sensitivity of speech detection.
- `max_duration`: Maximum duration (in seconds) of a valid speech segment.
- `min_silence_duration`: Minimum silence duration to split segments.
- `min_gap`: Minimum gap between consecutive segments.

Example:
```
short_segments = detect_short_segments(audio, sr, max_duration=1.5, min_energy=0.02, min_gap=0.1)
```

## Visualization

The tool generates an energy curve plot to visualize the detected speech segments along the timeline. The plots are saved in the `plots/` directory and include 1-second interval x-axis ticks for better clarity.
