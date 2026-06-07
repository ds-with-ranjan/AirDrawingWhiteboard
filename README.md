# ✋ Air Drawing Whiteboard

> Draw in the air using your webcam and hand gestures — no touch required.

A real-time computer vision project built with Python, OpenCV, and MediaPipe that turns your index finger into a virtual paintbrush.

---

## 📸 Screenshots

| Drawing Mode | Selection Mode | Eraser Mode |
|---|---|---|
| *Index finger up → draw freely* | *Index + middle → hover toolbar* | *Three fingers → erase* |

> Add your own screenshots in the `assets/` folder and link them here!

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-time hand tracking** | 21-landmark MediaPipe model at 30 FPS |
| **Air drawing** | Smooth, jitter-reduced strokes |
| **Gesture controls** | 4 gestures mapped to distinct modes |
| **Color palette** | Red, Green, Blue, Yellow, Black |
| **Brush sizes** | Small (4px), Medium (10px), Large (22px) |
| **Eraser** | Three-finger gesture, 40px radius |
| **Clear canvas** | One click / press `C` |
| **Shape detection** | Recognises Circle, Rectangle, Square, Triangle, Line |
| **Undo** | Up to 20 steps; toolbar button or `Z` key |
| **Save as PNG** | Toolbar button or `S` key; timestamped filename |
| **FPS counter** | Live performance readout |
| **Full-screen mode** | Toggle with `F` |
| **Keyboard shortcuts** | Full keyboard control |

---

## 🤚 Gesture Guide

| Gesture | Mode | What happens |
|---|---|---|
| ☝️ Index finger up | **Draw** | Finger tip leaves a coloured stroke |
| ✌️ Index + middle up | **Select** | Hover over toolbar to change settings |
| 🤟 Index + middle + ring up | **Erase** | Erase ink under cursor |
| ✊ Closed fist | **Pause** | No drawing; stroke is ended cleanly |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `C` | Clear canvas |
| `Z` | Undo last stroke |
| `S` | Save drawing as PNG |
| `F` | Toggle full-screen |

---

## 🗂 Project Structure

```
AirDrawingWhiteboard/
│
├── main.py              # Entry point — camera loop, HUD, keyboard input
├── hand_tracker.py      # MediaPipe wrapper, gesture detection
├── drawing_utils.py     # Canvas, strokes, toolbar, undo, save
├── shape_detector.py    # Contour-based shape recognition
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── assets/              # Screenshots, demo GIFs (add your own)
```

---

## 🛠 Installation

### Prerequisites

- Python **3.11+** (3.10 works too)
- A working **webcam**
- Git (optional, for cloning)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AirDrawingWhiteboard.git
cd AirDrawingWhiteboard
```

Or download and unzip the project.

### Step 2 — Create a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** On some systems `mediapipe` requires `pip install mediapipe --upgrade`.

---

## ▶️ Running the Project

```bash
python main.py
```

The webcam window opens in about 2–3 seconds.  
Position your hand in front of the camera and raise your **index finger** to start drawing!

### Changing the webcam

Edit the top of `main.py`:

```python
WEBCAM_INDEX = 0   # 0 = default, 1 = second camera, etc.
```

---

## ⚙️ Configuration

All tunable parameters live near the top of their respective files:

| File | Variable | Default | Effect |
|---|---|---|---|
| `main.py` | `FRAME_W / FRAME_H` | 1280 × 720 | Capture resolution |
| `main.py` | `FLIP_HORIZONTAL` | `True` | Mirror feed |
| `main.py` | `SHAPE_DETECT_INTERVAL` | `6` | Shape check every N frames |
| `hand_tracker.py` | `detection_confidence` | `0.75` | Detection sensitivity |
| `hand_tracker.py` | `tracking_confidence` | `0.75` | Tracking stability |
| `drawing_utils.py` | `ERASER_SIZE` | `40` | Eraser radius (px) |
| `shape_detector.py` | `min_contour_area` | `1500` | Min ink area for shape detection |

---

## 🧠 How It Works

### Hand Tracking (`hand_tracker.py`)
MediaPipe Hands processes each BGR frame and returns 21 3D landmarks per hand.  
We convert normalised coordinates to pixel coordinates and compare y-positions of **fingertips vs. PIP joints** to decide which fingers are extended.

### Gesture Detection
```
Index up only          → "draw"
Index + Middle up      → "select"
Index + Middle + Ring  → "erase"
All fingers down       → "fist" (pause)
```

### Drawing (`drawing_utils.py`)
- A **BGRA canvas** (same size as the frame) stores all ink.
- Strokes are drawn with `cv2.line` + round end-caps via `cv2.circle`.
- A **5-sample rolling average** smooths noisy landmark coordinates.
- Alpha-compositing merges the canvas over the live camera frame.
- Each draw event snapshots the canvas into a `deque` for **undo**.

### Shape Detection (`shape_detector.py`)
1. Extract the alpha channel as an ink mask.
2. Find contours with `cv2.findContours`.
3. Compute **circularity** (`4π·area / perimeter²`).
4. Use `cv2.approxPolyDP` to count vertices.
5. A shape is announced only after it appears stable for 8 consecutive frames (prevents flicker).

---

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open webcam` | Change `WEBCAM_INDEX` in `main.py` (try 1, 2, …) |
| Very low FPS | Lower `FRAME_W / FRAME_H`; close other apps using the camera |
| Landmarks jitter a lot | Improve lighting; increase `detection_confidence` |
| Wrong hand detected | The tracker defaults to the first hand found; ensure only one hand is visible |
| `ModuleNotFoundError: mediapipe` | Run `pip install mediapipe --upgrade` |
| Window opens then crashes (macOS) | Run inside a virtual environment; avoid Homebrew Python |
| Drawing feels laggy | Reduce `FRAME_W` to 640; lower smoothing buffer size in `drawing_utils.py` |

---

## 🚀 Future Improvements

- [ ] Multi-hand support (second hand for selection while first draws)
- [ ] Text annotation mode
- [ ] More shapes: ellipse, arrow, star
- [ ] Opacity / brush pressure simulation via hand distance from camera
- [ ] Load background images / photos and draw on top
- [ ] Export to SVG
- [ ] Virtual classroom / presentation mode with screen sharing
- [ ] On-screen colour picker with HSV slider

---

## 🤝 Contributing

Pull requests are welcome!  
Please open an issue first to discuss major changes.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) — hand landmark detection
- [OpenCV](https://opencv.org/) — image processing & display
- [NumPy](https://numpy.org/) — fast array operations
