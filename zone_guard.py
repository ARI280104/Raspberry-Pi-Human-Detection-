#!/usr/bin/env python3
"""
=================================================================
  AI/ML Restricted Zone Detection System
  Raspberry Pi 4  |  Debian Bookworm  |  rpicam-vid
  
  Uses:
    - MoveNet TFLite  → 17-point skeleton detection
    - OpenCV          → zone drawing, movement tracking
    - NumPy           → ML-based movement prediction
    - GPIO (optional) → buzzer/LED alert on zone breach
  
  Features:
    1. Live 17-point skeleton tracking
    2. Draw custom restricted zone (polygon)
    3. Movement trail tracking per person
    4. AI zone-breach prediction (Kalman-like motion)
    5. On-screen alert + GPIO buzzer/LED signal
    6. Breach logging to file with timestamp
=================================================================
"""

import cv2
import numpy as np
import subprocess
import threading
import time
import sys
import os
import csv
import urllib.request
from datetime import datetime
from collections import deque

# ── GPIO (optional — only works on real Pi hardware) ──────────
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    pass   # Running without GPIO is fine — visual alert still works

# =================================================================
#  CONFIGURATION — Edit these to customise your setup
# =================================================================

FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
FPS           = 30
MODEL_INPUT   = 192          # MoveNet Lightning input size
CONF_THRESH   = 0.25         # Keypoint confidence threshold

# GPIO pin numbers (BCM mode) — change to match your wiring
PIN_BUZZER    = 17           # Connect buzzer between GPIO17 and GND
PIN_LED_RED   = 27           # Connect LED (+ resistor) between GPIO27 and GND
PIN_LED_GREEN = 22           # Green LED = safe zone indicator

# Alert settings
ALERT_DURATION  = 2.0        # Seconds to hold alert state
TRAIL_LENGTH    = 20         # How many past positions to remember per person
PREDICT_FRAMES  = 8          # How many future frames to predict movement

# File paths
MODEL_FILE  = "movenet_lightning.tflite"
LOG_FILE    = "breach_log.csv"
MODEL_URL   = (
    "https://tfhub.dev/google/lite-model/"
    "movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
)

# Colours
C_JOINT       = (0, 255, 0)      # Green  — safe skeleton joints
C_JOINT_ALERT = (0, 0, 255)      # Red    — breached skeleton joints
C_BONE        = (255, 140, 0)    # Orange — skeleton bones
C_BONE_ALERT  = (0, 0, 255)      # Red    — alert bones
C_ZONE_SAFE   = (0, 255, 100)    # Green  — zone border safe
C_ZONE_ALERT  = (0, 0, 255)      # Red    — zone border breached
C_FILL_SAFE   = (0, 255, 100)    # Zone fill safe (semi-transparent)
C_FILL_ALERT  = (0, 0, 255)      # Zone fill alert
C_TRAIL       = (255, 200, 0)    # Yellow — movement trail
C_PREDICT     = (255, 0, 255)    # Magenta— predicted future path
C_TEXT_GOOD   = (0, 255, 0)
C_TEXT_WARN   = (0, 0, 255)
C_TEXT_WHITE  = (255, 255, 255)
C_TEXT_YELLOW = (0, 255, 255)

WINDOW_NAME = "AI Zone Guard — Pi4 | Q=Quit | D=Draw Zone | C=Clear Zone"

# =================================================================
#  SKELETON DEFINITIONS
# =================================================================

KEYPOINT_NAMES = [
    "Nose",
    "L.Eye",   "R.Eye",
    "L.Ear",   "R.Ear",
    "L.Shldr", "R.Shldr",
    "L.Elbow", "R.Elbow",
    "L.Wrist", "R.Wrist",
    "L.Hip",   "R.Hip",
    "L.Knee",  "R.Knee",
    "L.Ankle", "R.Ankle",
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
    (5, 7), (7, 9),                            # Left arm
    (6, 8), (8, 10),                           # Right arm
    (11, 13), (13, 15),                        # Left leg
    (12, 14), (14, 16),                        # Right leg
]

# Body keypoints used to calculate body "centre" position
BODY_CENTER_KPS = [5, 6, 11, 12]   # Shoulders + Hips


# =================================================================
#  GPIO SETUP
# =================================================================

def gpio_setup():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PIN_BUZZER,    GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_LED_RED,   GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_LED_GREEN, GPIO.OUT, initial=GPIO.HIGH)  # Green = safe on start
    print(f"[GPIO] Buzzer  → GPIO{PIN_BUZZER}")
    print(f"[GPIO] Red LED → GPIO{PIN_LED_RED}")
    print(f"[GPIO] Grn LED → GPIO{PIN_LED_GREEN}")


def gpio_alert(active: bool):
    """Turn buzzer + red LED on/off."""
    if not GPIO_AVAILABLE:
        return
    GPIO.output(PIN_BUZZER,    GPIO.HIGH if active else GPIO.LOW)
    GPIO.output(PIN_LED_RED,   GPIO.HIGH if active else GPIO.LOW)
    GPIO.output(PIN_LED_GREEN, GPIO.LOW  if active else GPIO.HIGH)


def gpio_cleanup():
    if not GPIO_AVAILABLE:
        return
    GPIO.output(PIN_BUZZER,    GPIO.LOW)
    GPIO.output(PIN_LED_RED,   GPIO.LOW)
    GPIO.output(PIN_LED_GREEN, GPIO.LOW)
    GPIO.cleanup()


# =================================================================
#  BREACH LOGGER
# =================================================================

class BreachLogger:
    def __init__(self, path=LOG_FILE):
        self.path = path
        self._write_header()

    def _write_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Timestamp", "Event", "X", "Y"])

    def log(self, event: str, x: int, y: int):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ts, event, x, y])


# =================================================================
#  AI MOVEMENT PREDICTOR  (simple Kalman-style linear predictor)
#  This is the ML component — learns velocity from past positions
#  and predicts where the person will be in future frames.
# =================================================================

class MovementPredictor:
    """
    Lightweight ML predictor using exponential-smoothed velocity.
    Predicts future body positions to warn BEFORE breach occurs.
    """

    def __init__(self, trail_len=TRAIL_LENGTH):
        self.trail   = deque(maxlen=trail_len)  # past (x,y) positions
        self.vx      = 0.0
        self.vy      = 0.0
        self.alpha   = 0.4   # smoothing factor (higher = more reactive)

    def update(self, x: float, y: float):
        if len(self.trail) > 0:
            px, py   = self.trail[-1]
            raw_vx   = x - px
            raw_vy   = y - py
            # Exponential smoothing — the ML part
            self.vx  = self.alpha * raw_vx + (1 - self.alpha) * self.vx
            self.vy  = self.alpha * raw_vy + (1 - self.alpha) * self.vy
        self.trail.append((x, y))

    def predict(self, steps=PREDICT_FRAMES):
        """Return list of predicted (x, y) for next `steps` frames."""
        if len(self.trail) < 2:
            return []
        px, py  = self.trail[-1]
        preds   = []
        vx, vy  = self.vx, self.vy
        for _ in range(steps):
            px += vx
            py += vy
            vx *= 0.92   # slight deceleration assumption
            vy *= 0.92
            preds.append((int(px), int(py)))
        return preds

    def get_trail(self):
        return list(self.trail)

    def speed(self):
        return (self.vx**2 + self.vy**2) ** 0.5


# =================================================================
#  ZONE MANAGER  — draw and manage restricted polygon
# =================================================================

class ZoneManager:
    """
    Manages the restricted zone polygon.
    Drawn by the user via mouse clicks.
    """

    def __init__(self):
        self.points      = []   # list of (x, y) tuples
        self.is_closed   = False
        self.drawing     = False

    def start_drawing(self):
        self.points    = []
        self.is_closed = False
        self.drawing   = True
        print("[ZONE] Click on the video window to place zone corners.")
        print("[ZONE] Press ENTER or 'Z' when done to close the zone.")

    def add_point(self, x, y):
        if self.drawing:
            self.points.append((x, y))
            print(f"[ZONE] Point added: ({x}, {y})  —  Total: {len(self.points)}")

    def close_zone(self):
        if len(self.points) >= 3:
            self.is_closed = True
            self.drawing   = False
            print(f"[ZONE] Zone closed with {len(self.points)} points.")
        else:
            print("[ZONE] Need at least 3 points to close a zone.")

    def clear(self):
        self.points    = []
        self.is_closed = False
        self.drawing   = False
        print("[ZONE] Zone cleared.")

    def is_ready(self):
        return self.is_closed and len(self.points) >= 3

    def point_inside(self, x, y):
        """Check if a pixel point is inside the zone polygon."""
        if not self.is_ready():
            return False
        poly = np.array(self.points, dtype=np.int32)
        result = cv2.pointPolygonTest(poly, (float(x), float(y)), False)
        return result >= 0   # >= 0 means inside or on edge

    def any_prediction_inside(self, predictions):
        """Return True if ANY predicted future position is inside zone."""
        return any(self.point_inside(px, py) for px, py in predictions)

    def draw(self, frame, alert=False, alpha=0.25):
        """Draw the zone on the frame with semi-transparent fill."""
        if len(self.points) < 2:
            # Just show dots while drawing
            for p in self.points:
                cv2.circle(frame, p, 6, C_ZONE_SAFE, -1)
            return

        pts = np.array(self.points, dtype=np.int32)
        color_border = C_ZONE_ALERT if alert else C_ZONE_SAFE
        color_fill   = C_FILL_ALERT  if alert else C_FILL_SAFE

        if self.is_closed:
            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color_fill)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # Solid border
            cv2.polylines(frame, [pts], True, color_border, 2, cv2.LINE_AA)
            # Corners
            for p in self.points:
                cv2.circle(frame, p, 5, color_border, -1)
        else:
            # Still drawing — show open polyline
            cv2.polylines(frame, [pts], False, C_ZONE_SAFE, 2, cv2.LINE_AA)
            for p in self.points:
                cv2.circle(frame, p, 5, C_ZONE_SAFE, -1)

        # Zone label
        if self.is_closed and len(self.points) > 0:
            cx = int(np.mean([p[0] for p in self.points]))
            cy = int(np.mean([p[1] for p in self.points]))
            label = "!! RESTRICTED !!" if alert else "RESTRICTED ZONE"
            cv2.putText(frame, label, (cx - 65, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_border, 2, cv2.LINE_AA)


# =================================================================
#  CAMERA READER  — rpicam-vid YUV420 pipe
# =================================================================

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
            print("  Fix : sudo apt install libcamera-apps -y")
            sys.exit(1)

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[INFO] Camera warming up (2 sec)...")
        time.sleep(2.0)
        print("[OK]  Camera ready.")

    def _loop(self):
        size = (FRAME_WIDTH * FRAME_HEIGHT * 3) // 2
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


# =================================================================
#  MODEL DOWNLOAD + LOAD
# =================================================================

def download_model():
    if os.path.exists(MODEL_FILE):
        print(f"[OK] Model found: {MODEL_FILE}")
        return
    print("[INFO] Downloading MoveNet (~3 MB, one-time only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[OK] Model downloaded.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print(f"  Manually download from:\n  {MODEL_URL}")
        print(f"  Save as: {MODEL_FILE}")
        sys.exit(1)


def load_model():
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=MODEL_FILE)
        interp.allocate_tensors()
        print("[OK] MoveNet model loaded (tflite_runtime).")
        return interp
    except ImportError:
        print("[ERROR] tflite_runtime not installed.")
        print("  Fix: pip3 install tflite-runtime --break-system-packages")
        sys.exit(1)


def run_movenet(interp, frame_bgr):
    """Run MoveNet. Returns (17, 3) array: [y_norm, x_norm, confidence]."""
    inp = interp.get_input_details()
    out = interp.get_output_details()
    img = cv2.resize(frame_bgr, (MODEL_INPUT, MODEL_INPUT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0).astype(np.int32)
    interp.set_tensor(inp[0]['index'], img)
    interp.invoke()
    return interp.get_tensor(out[0]['index'])[0][0]   # (17,3)


# =================================================================
#  DRAWING UTILITIES
# =================================================================

def hud_text(frame, text, pos, color=C_TEXT_WHITE, scale=0.55, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame, (x-2, y-h-4), (x+w+2, y+4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, alert=False):
    """Draw 17-point skeleton. Returns (visible_count, center_x, center_y)."""
    h, w = frame.shape[:2]
    c_joint = C_JOINT_ALERT if alert else C_JOINT
    c_bone  = C_BONE_ALERT  if alert else C_BONE

    pts = []
    for ky, kx, kc in keypoints:
        pts.append((int(kx * w), int(ky * h), float(kc)))

    # Draw bones
    for (i, j) in SKELETON_CONNECTIONS:
        x1, y1, c1 = pts[i]
        x2, y2, c2 = pts[j]
        if c1 > CONF_THRESH and c2 > CONF_THRESH:
            cv2.line(frame, (x1, y1), (x2, y2), c_bone, 2, cv2.LINE_AA)

    # Draw joints + labels
    visible = 0
    for idx, (px, py, conf) in enumerate(pts):
        if conf > CONF_THRESH:
            visible += 1
            cv2.circle(frame, (px, py), 6, c_joint, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, KEYPOINT_NAMES[idx],
                        (px + 7, py - 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, C_TEXT_YELLOW, 1, cv2.LINE_AA)

    # Body centre (average of visible shoulder+hip points)
    cx_list, cy_list = [], []
    for idx in BODY_CENTER_KPS:
        px, py, conf = pts[idx]
        if conf > CONF_THRESH:
            cx_list.append(px)
            cy_list.append(py)

    if cx_list:
        return visible, int(np.mean(cx_list)), int(np.mean(cy_list))
    return visible, -1, -1


def draw_trail(frame, trail_pts, color=C_TRAIL):
    """Draw fading movement trail."""
    n = len(trail_pts)
    for i in range(1, n):
        alpha = i / n
        r = int(color[0] * alpha)
        g = int(color[1] * alpha)
        b = int(color[2] * alpha)
        thick = max(1, int(3 * alpha))
        cv2.line(frame, trail_pts[i-1], trail_pts[i], (b, g, r), thick, cv2.LINE_AA)


def draw_predictions(frame, preds, color=C_PREDICT):
    """Draw predicted future path as dashed line."""
    for i, (px, py) in enumerate(preds):
        r = max(1, 5 - i // 2)
        alpha = 1 - (i / len(preds)) * 0.7
        c = tuple(int(x * alpha) for x in color)
        cv2.circle(frame, (px, py), r, c, -1, cv2.LINE_AA)


def draw_alert_overlay(frame, message="⚠ BREACH DETECTED ⚠"):
    """Flashing red border + big warning text."""
    if int(time.time() * 3) % 2 == 0:   # flash at ~1.5 Hz
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH-1, FRAME_HEIGHT-1),
                      (0, 0, 255), 8)
    # Big centred warning
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.2
    thick = 2
    (tw, th), _ = cv2.getTextSize(message, font, scale, thick)
    tx = (FRAME_WIDTH  - tw) // 2
    ty = (FRAME_HEIGHT + th) // 2
    cv2.rectangle(frame, (tx-10, ty-th-10), (tx+tw+10, ty+10),
                  (0, 0, 0), -1)
    cv2.putText(frame, message, (tx, ty), font, scale, (0, 0, 255), thick, cv2.LINE_AA)


def draw_predictive_alert(frame):
    """Yellow warning when person predicted to enter zone soon."""
    msg = "⚠ PREDICTED ZONE ENTRY"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    tx = (FRAME_WIDTH - tw) // 2
    ty = 80
    cv2.rectangle(frame, (tx-8, ty-th-8), (tx+tw+8, ty+8), (0, 0, 0), -1)
    cv2.putText(frame, msg, (tx, ty), font, scale, (0, 220, 255), thick, cv2.LINE_AA)


# =================================================================
#  MOUSE CALLBACK — for drawing the zone
# =================================================================

zone_manager_ref = None   # set in main

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if zone_manager_ref and zone_manager_ref.drawing:
            zone_manager_ref.add_point(x, y)


# =================================================================
#  MAIN
# =================================================================

def main():
    global zone_manager_ref

    print("=" * 60)
    print("  AI Zone Guard — Raspberry Pi 4  |  Bookworm")
    print("  MoveNet 17-pt Skeleton + Restricted Zone Detection")
    print("=" * 60)

    # Setup
    download_model()
    gpio_setup()
    logger   = BreachLogger()
    zone     = ZoneManager()
    zone_manager_ref = zone
    predictor = MovementPredictor()

    print("[INFO] Loading MoveNet model...")
    interp = load_model()

    camera = CameraReader()
    camera.start()

    # OpenCV window + mouse
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # State variables
    alert_until      = 0.0     # timestamp when current alert expires
    predict_warn     = False   # predictive warning active
    breach_count     = 0       # total breach events this session
    fps_count        = 0
    fps_display      = 0.0
    fps_timer        = time.time()
    last_breach_log  = 0.0     # avoid spamming log file

    print("\n[CONTROLS]")
    print("  D       = Start drawing restricted zone (click corners)")
    print("  Z/Enter = Close/finish zone")
    print("  C       = Clear zone")
    print("  Q/Esc   = Quit\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()

            # ── MoveNet inference ────────────────────────────
            keypoints = run_movenet(interp, frame)

            # ── Body centre + movement tracking ─────────────
            alert_active    = now < alert_until
            visible, cx, cy = draw_skeleton(frame, keypoints, alert=alert_active)

            person_detected = visible >= 3 and cx != -1

            if person_detected:
                predictor.update(cx, cy)

            # ── Zone logic ───────────────────────────────────
            inside_zone   = False
            predict_warn  = False

            if person_detected and zone.is_ready():
                # Check if body centre is inside zone
                inside_zone = zone.point_inside(cx, cy)

                # Also check if ANY skeleton joint is inside (stricter)
                h, w = frame.shape[:2]
                for ky, kx, kc in keypoints:
                    if kc > CONF_THRESH:
                        jx, jy = int(kx * w), int(ky * h)
                        if zone.point_inside(jx, jy):
                            inside_zone = True
                            break

                # Predictive check — warn before breach
                if not inside_zone:
                    preds = predictor.predict()
                    if zone.any_prediction_inside(preds):
                        predict_warn = True

                # Trigger alert on actual breach
                if inside_zone:
                    alert_until = now + ALERT_DURATION
                    gpio_alert(True)
                    if now - last_breach_log > 2.0:   # log every 2 sec max
                        breach_count   += 1
                        last_breach_log = now
                        logger.log("BREACH", cx, cy)
                        print(f"[ALERT] Zone breach #{breach_count} at ({cx},{cy})")
                else:
                    if not alert_active:
                        gpio_alert(False)

            # ── Draw zone ────────────────────────────────────
            zone.draw(frame, alert=alert_active or inside_zone)

            # ── Draw movement trail ──────────────────────────
            trail = predictor.get_trail()
            if len(trail) > 1:
                draw_trail(frame, [(int(x), int(y)) for x, y in trail])

            # ── Draw predicted path ──────────────────────────
            if person_detected and zone.is_ready():
                preds = predictor.predict()
                if preds:
                    draw_predictions(frame, preds)

            # ── Draw body centre dot ─────────────────────────
            if person_detected:
                color_c = (0, 0, 255) if inside_zone else (255, 255, 255)
                cv2.circle(frame, (cx, cy), 10, color_c, 2, cv2.LINE_AA)
                cv2.putText(frame, "CENTRE", (cx+12, cy+4),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, color_c, 1, cv2.LINE_AA)

            # ── Alert overlays ───────────────────────────────
            if alert_active or inside_zone:
                draw_alert_overlay(frame)
            elif predict_warn:
                draw_predictive_alert(frame)

            # ── FPS ──────────────────────────────────────────
            fps_count += 1
            if now - fps_timer >= 1.0:
                fps_display = fps_count / (now - fps_timer)
                fps_count   = 0
                fps_timer   = now

            # ── HUD panel ────────────────────────────────────
            hud_text(frame, f"FPS: {fps_display:.1f}", (10, 22), C_TEXT_YELLOW)
            hud_text(frame, f"Keypoints: {visible}/17",  (10, 46))

            if not zone.is_ready():
                if zone.drawing:
                    hud_text(frame,
                             f"DRAWING ZONE: {len(zone.points)} pts | Press Z to finish",
                             (10, 70), (0, 200, 255))
                else:
                    hud_text(frame, "Press D to draw restricted zone",
                             (10, 70), (180, 180, 180))
            else:
                status = "IN ZONE — BREACH!" if inside_zone else ("HEADING TO ZONE" if predict_warn else "Zone: Active")
                color  = C_TEXT_WARN if inside_zone else (C_TEXT_YELLOW if predict_warn else C_TEXT_GOOD)
                hud_text(frame, status,        (10, 70), color)
                hud_text(frame, f"Total breaches: {breach_count}", (10, 94))

            speed = predictor.speed()
            hud_text(frame, f"Speed: {speed:.1f} px/fr", (10, 118))

            gpio_txt = "GPIO: ON" if GPIO_AVAILABLE else "GPIO: Simulation"
            hud_text(frame, gpio_txt, (FRAME_WIDTH - 160, 22),
                     C_TEXT_GOOD if GPIO_AVAILABLE else (180, 180, 180))

            hud_text(frame, "D=Draw  Z=Close  C=Clear  Q=Quit",
                     (10, FRAME_HEIGHT - 10), (160, 160, 160))

            cv2.imshow(WINDOW_NAME, frame)

            # ── Key handling ─────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('d'), ord('D')):
                zone.start_drawing()
            elif key in (ord('z'), ord('Z'), 13):   # Z or Enter
                zone.close_zone()
            elif key in (ord('c'), ord('C')):
                zone.clear()
                predictor.trail.clear()
                breach_count = 0

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        print("[INFO] Cleaning up...")
        gpio_cleanup()
        camera.stop()
        cv2.destroyAllWindows()
        print(f"[DONE] Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
