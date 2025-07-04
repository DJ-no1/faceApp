# MediaPipe Feature Selector (PyQt5)

---

## Overview

MediaPipe Feature Selector is a user-friendly, real-time computer vision app. It lets you turn on hand tracking, face mesh, pose estimation, AR overlays, and more, all from a simple PyQt5 interface. You can use your webcam or screen as input, and see live overlays and stats instantly.

---

## Table of Contents

1. How It Works
2. System Diagram
3. Technology Breakdown
4. Features
5. File Structure
6. Installation
7. Customization
8. License

---

## How It Works

1. **Start the app**: The main window opens.
2. **Choose input**: Select your webcam or screen capture.
3. **Enable features**: Use checkboxes to turn on hand, face, pose, AR, and more.
4. **Click Start**: The video feed appears with overlays and live stats.
5. **Interact**: Try hand gestures, facial expressions, or AR accessories.
6. **Stop or close**: End the session anytime.

---

## System Diagram

Below is a clear, step-by-step block diagram showing the data and control flow in the app. Each block represents a major component, and arrows show how data moves through the system.

```
+-------------------+        +---------------------+
|   Input Devices   |        |    User Controls    |
| (Camera/Screen)   |<------>|      (PyQt5)        |
+--------+----------+        +----------+----------+
         |                              |
         v                              |
+--------+----------+        +----------v----------+
|  Frame Acquisition|        |   Settings/Config   |
| (OpenCV/PyAutoGUI)|        |   (PyQt5 Widgets)   |
+--------+----------+        +----------+----------+
         |                              |
         v                              |
+-------------------+        +---------------------+
|   Frame Buffer    |<-------+   Feature Toggles   |
|   (RAM/NumPy)     |        |   (PyQt5 Signals)   |
+--------+----------+        +----------+----------+
         |                              |
         v                              |
+-------------------+        +---------------------+
|   MediaPipe Core  |<-------+   Overlay Engine    |
| (Hand/Face/Pose/  |        | (Pillow/OpenCV)     |
|  Segmentation)    |        +---------------------+
+--------+----------+
         |
         v
+-------------------+
|   Output Display  |
| (PyQt5 QLabel,    |
|  Output Labels)   |
+-------------------+
```

![Basic flow ](ImgDir/basicflow.png)

**How to read this:**

- **Input Devices** (camera/screen) send frames to **Frame Acquisition** (OpenCV/PyAutoGUI).
- Frames are stored in the **Frame Buffer** (NumPy arrays in RAM).
- **User Controls** (PyQt5) let you change settings and features at any time.
- **MediaPipe Core** processes frames for hand, face, pose, and segmentation.
- **Overlay Engine** (Pillow/OpenCV) draws emojis, AR, and other effects.
- The final result is shown in the **Output Display** (video and stats in the GUI).

**Note:** User controls can affect any stage, and overlays are applied after MediaPipe processing for best results.

---

## Technology Breakdown

- **Python**: The main programming language.
- **sys**: Handles app startup and exit & screen captureing freature .
- **OpenCV (cv2)**: Captures video, processes frames, and handles image conversion. Stores each image/frame as a NumPy array (matrix of pixel values).
- **NumPy**: Fast array operations for image processing.
- **MediaPipe**: process those frames (provided as NumPy arrays from OpenCV) for advanced computer vision tasks like hand tracking, face mesh, pose estimation, and segmentation.Detects hands, faces, poses, and segments people in real time.
- **PyQt5**: Builds the graphical user interface and handles user input.
- **Pillow (PIL)**: Draws emojis and AR overlays on video frames.

---

## Features

- Hand gesture recognition (thumbs up, peace, call me, etc.)
- Face mesh with style and emoji overlays (sunglasses, crown, etc.)
- Eye state detection (open/closed)
- Finger and face counting
- Pose and holistic tracking
- Selfie segmentation (background removal)
- Screen capture mode
- Modern, dark-themed interface

---

## File Structure

- `faceApp.py` : Main application file
- `ar_assets/` : AR overlay images (glasses, moustache)
- `seguiemj.ttf` : Emoji font for overlays
- `MediaPipeApp_Documentation_NEW.md` : Documentation

---

## Installation

**Requirements:** Python 3.x

Install dependencies:

```bash
pip install pyqt5 opencv-python mediapipe pillow numpy pyautogui
```

---

## Customization

- Add new gestures or emojis: Edit `recognize_hand_gesture` and emoji lists in `faceApp.py`.
- Change the look: Update the `setStyleSheet` section in the main window class.
- Adjust detection: Tweak thresholds in `eye_state` and gesture logic.
- Add new AR overlays: Place PNGs in `ar_assets/` and update overlay code.

---

## License

For educational and demonstration use only.

---

## Internal Mechanism: OpenCV & MediaPipe (Detailed)

Below is a clear, step-by-step diagram and explanation of how OpenCV and MediaPipe work together inside the app, including their internal mechanisms and how user controls affect the flow.

```
+-------------------+         +---------------------+
|   Camera/Screen   |         |   User Controls     |
| (cv2.VideoCapture |         |   (PyQt5 Widgets)   |
|  or screenshot)   |         +----------+----------+
+--------+----------+                    |
         |                               |
         v                               |
+-------------------+                    |
| OpenCV:           |<-------------------+
|  - Frame Capture  |   User toggles     |
|  - Color Convert  |   (checkboxes,     |
|  - Flip/Resize    |    dropdowns)      |
|  - image into arr |                    |
+--------+----------+                    |
         |                               |
         v                               |
+-------------------+                    |
| Frame Buffer      |                    |
| (NumPy Array)     |                    |
+--------+----------+                    |
         |                               |
         v                               |
+-------------------+                    |
| MediaPipe:        |<-------------------+
|  - Model Select   |   User toggles     |
|  - Hand/Face/Pose |   (which models    |
|  - Segmentation   |    to run)         |
|  - Inference      |                    |
+--------+----------+                    |
         |                               |
         v                               |
+-------------------+                    |
| Landmark Output   |                    |
| (Hand, Face, etc.)|                    |
+--------+----------+                    |
         |                               |
         v                               |
+-------------------+                    |
| Overlay Engine    |<-------------------+
| (Pillow/OpenCV)   |   User toggles     |
|  - Draw Emoji     |   (emoji, AR,      |
|  - AR Graphics    |    mesh style)     |
+--------+----------+                    |
         |                               |
         v                               |
+-------------------+                    |
| PyQt5 GUI         |                    |
| (QLabel, Labels)  |                    |
+-------------------+                    |
```

### How OpenCV Works Internally

- **Frame Capture**: Uses `cv2.VideoCapture` to grab frames from the camera (or `pyautogui.screenshot()` for screen).
- **Color Conversion**: Converts frames from BGR (OpenCV default) to RGB (needed for MediaPipe).
- **Flip/Resize**: Flips frames horizontally for a mirror effect; resizes if needed for display.
- **Frame Buffer**: Stores the current frame as a NumPy array for fast access and processing.

### How MediaPipe Works Internally

- **Model Selection**: Based on user toggles, only the required models are loaded (e.g., Hands, FaceMesh, Pose, SelfieSegmentation).
- **Inference**: Each model processes the frame and returns landmarks or masks:
  - **Hands**: 21 hand landmarks per detected hand.
  - **FaceMesh**: 468 face landmarks, plus contours and irises.
  - **Pose**: 33 body landmarks.
  - **Segmentation**: Person mask for background removal.
- **Landmark Output**: Landmarks are used for gesture recognition, AR overlays, and stats.

### How User Controls Affect the Flow

- User toggles (checkboxes, dropdowns) are checked every frame.
- They determine:
  - Which MediaPipe models are active (for efficiency).
  - Which overlays are drawn (emoji, AR, mesh style).
  - What stats are shown (gesture, eye state, finger count, face count).
- Changes are applied instantly, without restarting the app.

### Overlay & Output

- **Overlay Engine**: Uses Pillow (PIL) and OpenCV to draw emojis, AR graphics, and mesh overlays based on landmarks and user choices.
- **PyQt5 GUI**: Displays the final frame and updates output labels in real time.

### Main Functions Involved

- `cv2.VideoCapture()`, `cv2.cvtColor()`, `cv2.flip()`: OpenCV frame handling.
- `mp_hands.Hands()`, `mp_face_mesh.FaceMesh()`, etc.: MediaPipe model instantiation.
- `process()`: Runs inference on the current frame.
- `draw_emoji_on_frame()`, `overlay_moustache()`: Overlay logic.
- PyQt5 signals/slots: Update GUI with new frames and stats.

---

For a graphical diagram, use draw.io, PowerPoint, or Mermaid.
