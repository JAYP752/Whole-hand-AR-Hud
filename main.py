import cv2
import math
import numpy as np
import urllib.request
import os

# MediaPipe new API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ---------------- Download Model ----------------
def download_model():
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading hand landmark model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded!")
    return model_path


# ---------------- Setup ----------------
cap = cv2.VideoCapture(0)

model_path = download_model()
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
landmarker = vision.HandLandmarker.create_from_options(options)

frame_timestamp = 0

# ---------------- Key Landmarks ----------------
# 0: Wrist, 4: Thumb tip, 8: Index tip, 12: Middle tip, 16: Ring tip, 20: Pinky tip
KEY_LANDMARKS = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}

LANDMARK_COLORS = {
    "wrist": (0, 255, 255),
    "thumb_tip": (255, 0, 255),
    "index_tip": (255, 100, 100),
    "middle_tip": (100, 255, 100),
    "ring_tip": (100, 100, 255),
    "pinky_tip": (255, 255, 100),
}


# ---------------- Helper Functions ----------------


def get_key_landmark_positions(landmarks, w, h):
    """Get pixel positions of the 6 key landmarks"""
    positions = {}
    for name, idx in KEY_LANDMARKS.items():
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        positions[name] = (x, y)
    return positions


def find_enclosing_circle(key_positions):
    """
    Find the minimum enclosing circle that passes through all 6 key points.
    Uses OpenCV's minEnclosingCircle for accurate calculation.
    """
    # Convert positions to numpy array
    points = np.array(list(key_positions.values()), dtype=np.float32)

    # Find minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(points)

    return (int(cx), int(cy)), radius


def get_hand_rotation(landmarks):
    """Get rotation angle of the hand"""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    return math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)


def draw_hand_skeleton(frame, landmarks, w, h):
    """Draw hand skeleton"""
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    for start, end in connections:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)


def draw_hud(frame, center, radius, rotation, key_positions):
    """
    Draw AR HUD where:
    - OUTER CIRCLE passes exactly through the 6 key landmarks
    - All other elements are INSIDE this circle
    """
    cx, cy = center
    r = int(radius)

    # ===== MAIN OUTER CIRCLE - PASSES THROUGH FINGERTIPS =====
    # This circle touches all 6 key landmarks (wrist + 5 fingertips)
    cv2.circle(frame, center, r, (255, 200, 0), 3)

    # ===== ALL INNER ELEMENTS (inside the outer circle) =====

    # Inner rings (percentages of outer radius)
    cv2.circle(frame, center, int(r * 0.80), (0, 255, 255), 2)  # 80%
    cv2.circle(frame, center, int(r * 0.60), (255, 150, 0), 1)  # 60%
    cv2.circle(frame, center, int(r * 0.40), (0, 200, 255), 1)  # 40%

    # Radial lines (from 40% to 80% - all inside)
    for i in range(0, 360, 30):
        a = math.radians(i) + rotation
        x_inner = int(cx + r * 0.40 * math.cos(a))
        y_inner = int(cy + r * 0.40 * math.sin(a))
        x_outer = int(cx + r * 0.80 * math.cos(a))
        y_outer = int(cy + r * 0.80 * math.sin(a))
        cv2.line(frame, (x_inner, y_inner), (x_outer, y_outer), (0, 200, 255), 1)

    # Orbiting nodes (at 85% - inside outer circle)
    for i in range(0, 360, 60):
        a = math.radians(i) - rotation * 2
        x = int(cx + r * 0.85 * math.cos(a))
        y = int(cy + r * 0.85 * math.sin(a))
        cv2.circle(frame, (x, y), 5, (0, 255, 150), -1)
        cv2.circle(frame, (x, y), 8, (0, 255, 150), 1)

    # Rotating arc segments (at 75% - inside)
    arc_radius = int(r * 0.75)
    for i in range(6):
        start_angle = int(math.degrees(rotation) + i * 60)
        end_angle = start_angle + 40
        cv2.ellipse(
            frame,
            center,
            (arc_radius, arc_radius),
            0,
            start_angle,
            end_angle,
            (255, 255, 0),
            2,
        )

    # ===== MARKERS ON THE 6 KEY LANDMARKS (on the outer circle) =====
    for name, pos in key_positions.items():
        px, py = pos
        color = LANDMARK_COLORS[name]

        # Circle marker on fingertip
        cv2.circle(frame, (px, py), 14, color, 2)
        cv2.circle(frame, (px, py), 6, color, -1)

        # Line from center to fingertip
        cv2.line(frame, center, (px, py), color, 1)

    # ===== CENTER CROSSHAIR =====
    size = int(r * 0.15)
    cv2.line(frame, (cx - size, cy), (cx + size, cy), (255, 255, 255), 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), (255, 255, 255), 2)
    cv2.circle(frame, center, 8, (0, 255, 255), -1)
    cv2.circle(frame, center, 12, (255, 255, 255), 2)


# ---------------- Main Loop ----------------
print("\n" + "=" * 50)
print("  WHOLE-HAND AR HUD")
print("=" * 50)
print("\nOuter circle passes through:")
print("  0  - Wrist")
print("  4  - Thumb tip")
print("  8  - Index finger tip")
print("  12 - Middle finger tip")
print("  16 - Ring finger tip")
print("  20 - Pinky tip")
print("\nPress 'Q' to quit")
print("=" * 50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp += 33
    results = landmarker.detect_for_video(mp_image, frame_timestamp)

    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        landmarks = results.hand_landmarks[0]

        # Draw skeleton
        draw_hand_skeleton(frame, landmarks, w, h)

        # Get 6 key landmark positions
        key_positions = get_key_landmark_positions(landmarks, w, h)

        # Find circle that passes through all 6 points
        center, radius = find_enclosing_circle(key_positions)

        # Get rotation for animation
        rotation = get_hand_rotation(landmarks)

        # Draw HUD
        draw_hud(frame, center, radius, rotation, key_positions)
    else:
        cv2.putText(
            frame,
            "Show your hand",
            (w // 2 - 80, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Whole-Hand AR HUD", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("\nGoodbye!")
