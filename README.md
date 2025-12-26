# Whole-Hand AR HUD

An Augmented Reality (AR) Heads-Up Display (HUD) controlled by your hand gestures! This project uses MediaPipe to create a futuristic, Iron Man-style interface that tracks your hand and displays dynamic elements.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.30-orange.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## ✨ Features

- **Futuristic AR Interface**: A dynamic HUD that tracks your hand position and rotation
- **6-Point Tracking**: Uses Wrist, Thumb, Index, Middle, Ring, and Pinky tips to define the interface
- **Adaptive UI**:
  - **Outer Circle**: Encloses all key fingertips dynamically
  - **Inner Rings**: Rotate and react to hand orientation
  - **Arc Segments**: Animated elements that follow hand rotation
  - **Fingertip Markers**: Visual indicators for each key finger
- **Real-time Performance**: Optimized for smooth tracking using MediaPipe's latest Tasks API
- **Automatic Model Download**: Necessary AI models download automatically on first run

## 🎮 How to Use

1. **Run the application**
2. **Show your hand** to the webcam
3. **Move your hand**: The HUD follows your hand center
4. **Rotate your hand**: The interface elements rotate to match your hand's orientation
5. **Open/Close hand**: The HUD expands and contracts based on your finger positions

**Controls:**

- **Q** or **ESC**: Quit the application

## 🔧 Requirements

- **Python 3.10+** (tested with Python 3.13)
- **Webcam**
- **Operating System**: macOS, Linux, or Windows

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/JAYP752/Whole-hand-AR-Hud.git
cd Whole-hand-AR-Hud
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python3 main.py
```

## 📦 Dependencies

- **opencv-python** - Computer vision and rendering
- **mediapipe** - Hand landmark detection (Tasks API)
- **numpy** - Mathematical operations

## 🤖 About the Model

This project uses the `hand_landmarker.task` model (7.5MB) from Google MediaPipe.

- **Auto-download**: The application automatically downloads this file on first run.
- **Manual download** (optional):
  ```bash
  curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  ```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to report bugs or suggest new AR features.

---

**Built with MediaPipe & OpenCV 🚀**
