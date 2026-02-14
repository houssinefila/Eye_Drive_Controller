# Eye-Controlled Car Project

A computer vision project that detects whether a person's eyes are open or closed using MediaPipe Face Mesh, and controls a virtual car based on eye state.

## Description

This project uses real-time face and eye detection to monitor if a person is looking at the camera with their eyes open. The system is designed to:
- Move the car forward when eyes are open
- Stop the car when eyes are closed for more than 2 seconds
- Ignore normal blinking (less than 2 seconds)

## Features

- Real-time eye detection using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) calculation for open/closed detection
- 2-second threshold to distinguish between blinking and actual eye closure
- Visual feedback with color-coded status indicators
- Live EAR value display for debugging

## Requirements
```
opencv-python
mediapipe
numpy
```

## Installation

1. Clone this repository or download the script

2. Install required packages:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

1. Run the script:
```bash
python main.py
```

2. The webcam will open and start detecting your face

3. Status indicators:
   - **Green** - Eyes open, car moving forward
   - **Orange** - Eyes blinking (< 2s), car still moving
   - **Red** - Eyes closed for > 2s, car stopped

4. Press 'q' to quit

## How It Works

### Eye Aspect Ratio (EAR)

The system calculates the Eye Aspect Ratio using 6 landmark points around each eye:
```
EAR = (vertical_distance_1 + vertical_distance_2) / (2 × horizontal_distance)
```

- **EAR > 0.25**: Eyes are open
- **EAR < 0.25**: Eyes are closed

### Timer Logic

- When eyes close, a timer starts
- If eyes remain closed for < 2 seconds: Treated as blinking (car continues)
- If eyes remain closed for ≥ 2 seconds: Car stops with warning
- When eyes open, timer resets

## Configuration

You can adjust these parameters in the code:
```python
CLOSED_THRESHOLD = 2.0  # Seconds before car stops
ear < 0.25              # EAR threshold for eye closure
```

## Camera Selection

The script uses camera index 1:
```python
cap = cv2.VideoCapture(1)
```

Change to `0` for default webcam:
```python
cap = cv2.VideoCapture(0)
```

## Troubleshooting

**Camera not opening:**
- Try changing camera index from 1 to 0
- Check if camera is being used by another application

**Face not detected:**
- Ensure proper lighting
- Face the camera directly
- Adjust `min_detection_confidence` value

**False eye closure detection:**
- Adjust the EAR threshold (currently 0.25)
- Modify the 2-second timer threshold

## Technical Details

### Eye Landmarks Used

- **Left Eye**: [33, 160, 158, 133, 153, 144]
- **Right Eye**: [362, 385, 387, 263, 373, 380]

### MediaPipe Configuration
```python
max_num_faces=1
refine_landmarks=True
min_detection_confidence=0.5
min_tracking_confidence=0.5
```

## Future Enhancements

- Add gaze direction detection
- Implement actual motor control for physical car
- Add drowsiness detection alerts
- Support for multiple faces
- Calibration mode for personalized thresholds

## License

MIT License

## Author

Fila houssine 

## Acknowledgments

- MediaPipe by Google
- OpenCV community
