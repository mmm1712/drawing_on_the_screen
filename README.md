# Air Draw – Gesture-Controlled Drawing Overlay

Air Draw is a real-time gesture-controlled drawing application that allows users to draw on the screen using hand movements captured by a webcam. The system tracks finger position and pinch gestures to enable touch-free drawing with adjustable brush size, color selection, canvas clearing, and image export.

This project focuses on real-time computer vision, signal filtering, multi-threaded UI design, and custom rendering.

## Features

- Real-time hand tracking using MediaPipe
- Pinch gesture detection with hysteresis for stable interaction
- Smooth cursor movement using a One Euro adaptive low-pass filter
- Custom transparent overlay UI rendered with PyQt6
- Adjustable brush size and color selection
- Stroke interpolation for smooth continuous lines
- Save drawings as PNG images
- Non-blocking UI using a dedicated worker thread

## How It Works

### Hand Tracking

The webcam feed is processed using MediaPipe Hands. The index finger tip (landmark 8) is tracked and mapped to screen coordinates.

### Signal Smoothing

A One Euro Filter is applied independently to X and Y coordinates to reduce jitter while maintaining low latency.

### Gesture Detection

A pinch gesture is detected based on the distance between the thumb and index finger. Separate activation and release thresholds prevent accidental toggling. A short grace period avoids unintended stroke breaks.

### Drawing Logic

When a pinch is detected, a stroke begins or continues. Points are interpolated between frames to ensure smooth drawing even during fast hand movement.

### UI and Rendering

A fullscreen transparent overlay renders strokes and a glass-style toolbar. The toolbar allows brush size control, color selection, canvas clearing, and saving drawings.

## Architecture

- **HandWorker (QThread)**  
  Handles webcam capture, MediaPipe processing, and gesture detection off the UI thread.

- **Overlay (QWidget)**  
  Manages drawing state, interaction logic, and custom rendering.

- **One Euro Filter**  
  Adaptive signal filter used to smooth cursor motion while preserving responsiveness.

This separation ensures responsive performance and clean maintainable code.

## Technologies Used

- Python 3
- OpenCV
- MediaPipe
- PyQt6
