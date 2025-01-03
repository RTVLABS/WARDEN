# Face Analyzer Documentation

## Overview
Face Analyzer is a real-time face analysis system that uses computer vision and deep learning to detect and analyze faces from webcam input. It provides information about age, gender, and ethnicity with smoothed results for better accuracy.

## Features
- Real-time face detection using YOLOv8
- Age estimation
- Gender detection
- Ethnicity/race detection
- Result smoothing with configurable buffers
- Configurable update intervals
- Adjustable confidence thresholds
- Console-based output with clean formatting

## Requirements
- Python 3.9+
- OpenCV
- Ultralytics YOLO
- DeepFace
- NumPy

## Installation
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python face_analyzer.py
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|--------|---------|-------------|
| `--verbose` | `-v` | False | Enable verbose output |
| `--buffer-size` | `-b` | 15 | Size of rolling buffer for result smoothing |
| `--update-interval` | `-u` | 1.0 | Time between updates (seconds) |
| `--min-votes` | `-m` | 5 | Minimum votes needed before showing results |
| `--display-results` | `-d` | 8 | Number of results to show in history |
| `--confidence` | `-c` | 0.6 | Confidence threshold (0.0-1.0) |
| `--width` | | 640 | Camera width resolution |
| `--height` | | 480 | Camera height resolution |
| `--fps` | | 30 | Camera FPS |

### Example Commands

1. Run with default settings:
```bash
python face_analyzer.py
```

2. Enable verbose mode with higher confidence:
```bash
python face_analyzer.py -v -c 0.8
```

3. Faster updates with smaller buffer:
```bash
python face_analyzer.py -u 0.5 -b 10
```

4. Higher resolution with more history:
```bash
python face_analyzer.py --width 1280 --height 720 -d 12
```

5. Maximum accuracy settings:
```bash
python face_analyzer.py -c 0.8 -b 20 -m 10 -u 1.5
```

## Output Format
The program displays results in a clean, tabulated format:
```
🔍 Real-time Face Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Active Analysis - Last Updated: HH:MM:SS

TIME     │ AGE  │ GENDER │ ETHNICITY  │ CONF
------------------------------------------------
14:25:33 │  25  │ Man    │ white      │ 0.92
14:25:34 │  26  │ Man    │ white      │ 0.94
```

## Configuration Details

### Buffer Size (`-b, --buffer-size`)
- Controls the number of samples used for smoothing results
- Larger values = more stable but slower to change
- Smaller values = more responsive but may fluctuate
- Default: 15

### Update Interval (`-u, --update-interval`)
- Time between result updates in seconds
- Lower values = more frequent updates
- Higher values = more stable output
- Default: 1.0 seconds

### Minimum Votes (`-m, --min-votes`)
- Required samples before showing results
- Higher values = more accurate but longer initial delay
- Lower values = faster initial results but may be less accurate
- Default: 5

### Confidence Threshold (`-c, --confidence`)
- Minimum confidence score for detection
- Range: 0.0 to 1.0
- Higher values = more accurate but may miss detections
- Lower values = more detections but may include false positives
- Default: 0.6

### Display Results (`-d, --display-results`)
- Number of historical results to show
- Affects console output scrolling
- Default: 8

## Performance Tips

1. **For Better Accuracy:**
   - Increase buffer size (`-b 20`)
   - Increase minimum votes (`-m 10`)
   - Increase confidence threshold (`-c 0.8`)
   - Increase update interval (`-u 1.5`)

2. **For Faster Response:**
   - Decrease buffer size (`-b 8`)
   - Decrease minimum votes (`-m 3`)
   - Decrease update interval (`-u 0.3`)
   - Lower resolution (`--width 320 --height 240`)

3. **For Better Performance:**
   - Lower resolution
   - Decrease FPS
   - Increase update interval

## Troubleshooting

### Common Issues

1. **Webcam Not Found**
   - Error: "Could not open webcam"
   - Solution: Check webcam connection and permissions

2. **Slow Performance**
   - Reduce resolution
   - Increase update interval
   - Decrease buffer size

3. **Unstable Results**
   - Increase buffer size
   - Increase minimum votes
   - Increase update interval

4. **No Detection**
   - Decrease confidence threshold
   - Check lighting conditions
   - Ensure face is clearly visible

## Notes
- The system requires good lighting for optimal performance
- Face should be clearly visible to the camera
- Processing speed depends on hardware capabilities
- Results are smoothed over time for stability 