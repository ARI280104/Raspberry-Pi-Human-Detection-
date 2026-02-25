#!/usr/bin/env python3
"""
=============================================================
  Human Skeleton Detection — Raspberry Pi 4 (Bookworm)
  Uses: OpenCV + MoveNet TFLite (17 keypoints / skeleton)
  Camera: rpicam-vid  |  No MediaPipe needed
=============================================================
"""

import cv2
import numpy as np
import subprocess
import threading
import time
import sys
import os

# ─────────────────────────────────────────────────────────────
#  MoveNet gives 17 keypoints (skeleton points):
#
#   0  = Nose
#   1  = Left Eye       2  = Right Eye
#   3  = Left Ear       4  = Right Ear
#   5  = Left Shoulder  6  = Right Shoulder
#   7  = Left Elbow     8  = Right Elbow
#   9  = Left Wrist     10 = Right Wrist
#  11  = Left Hip       12 = Right Hip
#  13  = Left Knee      14 = Right Knee
#  15  = Left Ankle     16 = Right Ankle
# ─────────────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    "Nose",
    "L.Eye", "R.Eye",
    "L.Ear", "R.Ear",
    "L.Shoulder", "R.Shoulder",
    "L.Elbow", "R.Elbow",
    "L.Wrist", "R.Wrist",
    "L.Hip", "R.Hip",
    "L.Knee", "R.Knee",
    "L.Ankle", "R.Ankle",
]

# Skeleton connections: pairs of keypoint indices to draw as bones
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]

# Colors
COLOR_JOINT      = (0, 255, 0)      # Green  – joints
COLOR_BONE       = (255, 100, 0)    # Orange – bones
COLOR_LABEL      = (255, 255, 0)    # Yellow – labels
COLOR_DETECTED   = (0, 255, 0)      # Green  – status text
COLOR_NONE       = (0, 120, 255)    # Orange – no detection
COLOR_HUD        = (255, 255, 255)  # White  – HUD text

# Camera / model settings
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
FPS           = 30
MODEL_INPUT   = 192        # MoveNet Lightning input size (192x192)
CONF_THRESH   = 0.25       # Only draw keypoints above this confidence
MODEL_FILE    = "movenet_lightning.tflite"
MODEL_URL     = (
    "https://tfhub.dev/google/lite-model/"
    "movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
)

WINDOW_NAME = "Skeleton Detection (MoveNet 17-pt) — Q to Quit"


# ─────────────────────────────────────────────────────────────
#  Download model if not present
# ─────────────────────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_FILE):
        print(f"[OK] Model found: {MODEL_FILE}")
        return

    print("[INFO] MoveNet TFLite model not found. Downloading...")
    print("       This only happens once (~3 MB). Please wait...")

    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[OK] Model downloaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Could not download model automatically: {e}")
        print("\n  Please download it manually:")
        print(f"  URL : {MODEL_URL}")
        print(f"  Save as: {MODEL_FILE}")
        print(f"  Place it in the same folder as this script.\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  Camera reader (rpicam-vid → YUV420 pipe)
# ─────────────────────────────────────────────────────────────
class CameraReader:
    def __init__(self):
        self.frame   = None
        self.running = False
        self._lock   = threading.Lock()
        self._proc   = None
        self._thread = None

    def start(self):
        cmd = [
            "rpicam-vid",
            "--width",     str(FRAME_WIDTH),
            "--height",    str(FRAME_HEIGHT),
            "--framerate", str(FPS),
            "--codec",     "yuv420",
            "--timeout",   "0",
            "-o",          "-",
            "--nopreview",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError:
            print("[ERROR] 'rpicam-vid' not found!")
            print("  Fix: sudo apt install libcamera-apps -y")
            sys.exit(1)

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[INFO] Camera warming up...")
        time.sleep(2.0)
        print("[OK] Camera ready.")

    def _loop(self):
        size = (FRAME_WIDTH * FRAME_HEIGHT * 3) // 2   # YUV420 bytes
        while self.running:
            raw = self._proc.stdout.read(size)
            if len(raw) < size:
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (FRAME_HEIGHT * 3 // 2, FRAME_WIDTH)
            )
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            with self._lock:
                self.frame = bgr

    def read(self):
        with self._lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait()


# ─────────────────────────────────────────────────────────────
#  MoveNet inference
# ─────────────────────────────────────────────────────────────
def load_model():
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=MODEL_FILE)
    except ImportError:
        try:
            interpreter = cv2.dnn.readNetFromTFLite(MODEL_FILE)
            return None   # signal to use DNN path (not used here)
        except Exception:
            pass
        print("[ERROR] tflite_runtime not installed.")
        print("  Fix: pip3 install tflite-runtime --break-system-packages")
        sys.exit(1)

    interpreter.allocate_tensors()
    return interpreter


def run_movenet(interpreter, frame_bgr):
    """
    Run MoveNet on a BGR frame.
    Returns array of shape (17, 3): [y_norm, x_norm, confidence]
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess
    img = cv2.resize(frame_bgr, (MODEL_INPUT, MODEL_INPUT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.int32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_details[0]['index'])
    # Shape: (1, 1, 17, 3)
    return keypoints[0][0]   # (17, 3)


# ─────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────
def draw_hud(frame, text, pos, color=COLOR_HUD):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame, (x - 2, y - h - 4), (x + w + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(frame, keypoints):
    """
    Draw 17-point skeleton on frame.
    keypoints: array (17, 3) with [y_norm, x_norm, confidence]
    Returns number of visible keypoints.
    """
    h, w = frame.shape[:2]
    visible = 0

    # Convert normalised coords → pixel coords
    pts = []
    for ky, kx, kc in keypoints:
        px = int(kx * w)
        py = int(ky * h)
        pts.append((px, py, float(kc)))

    # Draw bones (connections)
    for (i, j) in SKELETON_CONNECTIONS:
        px1, py1, c1 = pts[i]
        px2, py2, c2 = pts[j]
        if c1 > CONF_THRESH and c2 > CONF_THRESH:
            cv2.line(frame, (px1, py1), (px2, py2), COLOR_BONE, 2, cv2.LINE_AA)

    # Draw joints
    for idx, (px, py, conf) in enumerate(pts):
        if conf > CONF_THRESH:
            visible += 1
            cv2.circle(frame, (px, py), 6, COLOR_JOINT, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)   # outline
            # Label
            label = f"{idx}:{KEYPOINT_NAMES[idx]}"
            cv2.putText(
                frame, label,
                (px + 7, py - 5),
                cv2.FONT_HERSHEY_PLAIN, 0.75, COLOR_LABEL, 1, cv2.LINE_AA
            )

    return visible


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Human Skeleton Detection — Raspberry Pi 4")
    print("  MoveNet Lightning  |  17 keypoints")
    print("=" * 55)

    download_model()

    print("[INFO] Loading MoveNet TFLite model...")
    interpreter = load_model()
    print("[OK] Model loaded.")

    camera = CameraReader()
    camera.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    fps_count   = 0
    fps_display = 0.0
    fps_timer   = time.time()

    print("[INFO] Running — press Q in the window to quit.\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # ── Inference ──────────────────────────────────
            keypoints = run_movenet(interpreter, frame)

            # ── Draw skeleton ──────────────────────────────
            visible = draw_skeleton(frame, keypoints)

            # ── FPS ────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count   = 0
                fps_timer   = time.time()

            # ── HUD overlay ────────────────────────────────
            draw_hud(frame, f"FPS: {fps_display:.1f}", (10, 25), (0, 255, 255))

            if visible > 0:
                status = f"HUMAN DETECTED  |  Visible Points: {visible}/17"
                draw_hud(frame, status, (10, 55), COLOR_DETECTED)
            else:
                draw_hud(frame, "No human detected", (10, 55), COLOR_NONE)

            draw_hud(
                frame,
                "MoveNet 17-pt Skeleton  |  Q = Quit",
                (10, FRAME_HEIGHT - 10),
                (180, 180, 180),
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        print("[INFO] Cleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    main()
