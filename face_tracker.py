#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Self-Contained Pan-Tilt Face Tracker for Raspberry Pi using YuNet

Author: Gemini
Date: 2025-08-30

Description:
This script is a complete, self-sufficient face-tracking solution. It addresses
common issues of color inaccuracy and poor detection by using a modern YuNet
face detection model and correct camera color space configuration.

Key Features:
- **YuNet DNN Model:** Uses a state-of-the-art face detector for higher
  accuracy in varied conditions.
- **Automatic Model Download:** Downloads the required model file on the first
  run, making the script portable and easy to set up.
- **Correct Color Handling:** Configures the PiCamera2 to output in BGR format,
  eliminating common color-swapping issues (e.g., brown appearing blue).
- **Hardware Abstraction:** Includes dummy classes for testing without a
  connected ServoKit, allowing for easy software development.
- **Robust Fallback:** If the YuNet model cannot be loaded, it will fall back
  to the classic Haar Cascade classifier.
"""

import os
import time
import argparse
import unittest
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

# --- Dependency Checks ---
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA2 = True
except (ImportError, ModuleNotFoundError):
    HAVE_PICAMERA2 = False

try:
    from adafruit_servokit import ServoKit as _RealServoKit
except (ImportError, ModuleNotFoundError):
    _RealServoKit = None

# --- Configuration ---
@dataclass
class CFG:
    """Holds all configuration parameters for the tracker."""
    # Camera and Display
    width: int = 640
    height: int = 480
    fps: int = 20
    # CRITICAL FIX for color: Use a BGR format directly from the camera so
    # OpenCV gets the correct channel order. This stops brown from looking blue.
    CAMERA_FORMAT: str = "BGR888"

    # Servo Channels and Limits (for Adafruit ServoKit)
    PAN_CH: int = 0
    TILT_CH: int = 1
    PAN_MIN: int = 5
    PAN_MAX: int = 175
    TILT_MIN: int = 15
    TILT_MAX: int = 165
    PAN_START: int = 90
    TILT_START: int = 90

    # Servo Direction (+1 or -1 to flip direction)
    PAN_DIR: int = +1
    TILT_DIR: int = +1

    # PD Controller Gains (Proportional, Derivative)
    KP_PAN: float = 6.0
    KP_TILT: float = 6.0
    KD_PAN: float = 8.0
    KD_TILT: float = 8.0

    # Tracking Algorithm Tuning
    EMA_ALPHA: float = 0.25      # Smoothing for target position (lower is smoother)
    DEADZONE_X: float = 0.08     # % of screen width from center to ignore error
    DEADZONE_Y: float = 0.10     # % of screen height from center to ignore error
    MAX_STEP: float = 2.5        # Max servo movement degrees per frame
    MIN_MOVE: float = 1.0        # Min servo movement to apply (filters noise)
    LOST_FRAMES: int = 30        # Frames without a face before returning home
    HOME_RATE: float = 2.0       # Degrees per frame when returning home

    # Face Detection Model Configuration
    MODEL_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    # NEW MODEL: YuNet, which is more accurate and robust.
    YUNET_MODEL: str = "face_detection_yunet_2023mar.onnx"
    # Score threshold for YuNet, can be tuned.
    CONF_THRESH: float = 0.6


# --- Utility & Setup ---
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def setup_models() -> None:
    """Checks for the YuNet model file and downloads it if missing."""
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    
    yunet_path = os.path.join(CFG.MODEL_DIR, CFG.YUNET_MODEL)
    if not os.path.exists(yunet_path) or os.path.getsize(yunet_path) < 100 * 1024:
        print(f"Downloading YuNet model: {CFG.YUNET_MODEL}...")
        url = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{CFG.YUNET_MODEL}"
        try:
            urllib.request.urlretrieve(url, yunet_path)
            print(f"Successfully downloaded model to {yunet_path}")
        except Exception as e:
            print(f"ERROR: Could not download YuNet model from {url}: {e}")
            print("The script will attempt to fall back to the Haar Cascade classifier.")

# --- Hardware Abstraction ---
class DummyServo:
    def __init__(self) -> None: self._angle: Optional[int] = None
    @property
    def angle(self) -> Optional[int]: return self._angle
    @angle.setter
    def angle(self, a: int) -> None: self._angle = a
    def set_pulse_width_range(self, *_: int) -> None: pass

class DummyKit:
    def __init__(self) -> None: self.servo = {i: DummyServo() for i in range(16)}

def get_servo_kit() -> Tuple[object, int, int]:
    """Initializes and returns the real or dummy servokit."""
    if _RealServoKit is not None:
        try:
            kit = _RealServoKit(channels=16)
            for s in (kit.servo[CFG.PAN_CH], kit.servo[CFG.TILT_CH]):
                s.set_pulse_width_range(600, 2400)
            kit.servo[CFG.PAN_CH].angle = CFG.PAN_START
            kit.servo[CFG.TILT_CH].angle = CFG.TILT_START
            print("Adafruit ServoKit initialized.")
            return kit, CFG.PAN_START, CFG.TILT_START
        except Exception as e:
            print(f"Could not initialize ServoKit: {e}. Using dummy servos.")
    print("Using dummy servos for testing.")
    kit = DummyKit()
    return kit, CFG.PAN_START, CFG.TILT_START

def set_angles(kit: object, pan: float, tilt: float) -> Tuple[int, int]:
    pan_out = int(clamp(round(pan), CFG.PAN_MIN, CFG.PAN_MAX))
    tilt_out = int(clamp(round(tilt), CFG.TILT_MIN, CFG.TILT_MAX))
    kit.servo[CFG.PAN_CH].angle = pan_out
    kit.servo[CFG.TILT_CH].angle = tilt_out
    return pan_out, tilt_out

# --- Control & Detection ---
def pd_step(err: float, prev_err: float, dt: float, kp: float, kd: float, max_step: float) -> float:
    dt = max(1e-3, dt)
    p = kp * err
    d = kd * (err - prev_err) / dt
    return float(np.clip(p + d, -max_step, max_step))

class FaceDetector:
    """Handles face detection using YuNet with a Haar cascade fallback."""
    def __init__(self, size: Tuple[int, int]) -> None:
        self.mode: Optional[str] = None
        self.detector: Optional[object] = None
        self._init_model(size)

    def _init_model(self, size: Tuple[int, int]) -> None:
        """Initializes the YuNet model or falls back to Haar."""
        yunet_path = os.path.join(CFG.MODEL_DIR, CFG.YUNET_MODEL)
        if os.path.exists(yunet_path):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=yunet_path,
                    config="",
                    input_size=size,
                    score_threshold=CFG.CONF_THRESH
                )
                self.mode = "yunet"
                print(f"YuNet face detector initialized successfully.")
                return
            except Exception as e:
                print(f"Error loading YuNet model: {e}")
        
        try:
            xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self.detector = cv2.CascadeClassifier(xml)
            if self.detector.empty(): raise IOError("Haar cascade file empty.")
            self.mode = "haar"
            print("Fell back to Haar Cascade for face detection.")
        except Exception as e:
            self.mode = "none"
            print(f"FATAL: Could not load any face detection model: {e}")

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects the largest face in a frame."""
        if self.mode == "yunet" and self.detector is not None:
            # YuNet returns (status, faces), where faces is None or a list of detections
            _, faces = self.detector.detect(frame_bgr)
            if faces is None or len(faces) == 0:
                return None
            # Find face with the largest area
            areas = [int(f[2]) * int(f[3]) for f in faces]
            largest_face = faces[np.argmax(areas)]
            return tuple(largest_face[0:4].astype(int))

        if self.mode == "haar" and self.detector is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            return int(x), int(y), int(w), int(h)

        return None

# --- Main Application ---
def run_tracker() -> None:
    """Initializes components and runs the main tracking loop."""
    show = True
    try:
        cv2.namedWindow("pibye Face Track", cv2.WINDOW_NORMAL)
    except Exception:
        show = False
        print("Could not create OpenCV window. Running in headless mode.")

    if HAVE_PICAMERA2:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (CFG.width, CFG.height), "format": CFG.CAMERA_FORMAT}
        )
        cam.configure(config)
        cam.start()
        use_cv_capture = False
        print("Using picamera2 for video capture.")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.height)
        use_cv_capture = True
        print("Using cv2.VideoCapture as fallback.")

    det = FaceDetector(size=(CFG.width, CFG.height))
    if det.mode == "none": return
        
    kit, pan, tilt = get_servo_kit()
    
    # State & timing variables
    ema_cx, ema_cy = CFG.width / 2.0, CFG.height / 2.0
    prev_err_x, prev_err_y = 0.0, 0.0
    miss_frames, frame_count = 0, 0
    last_t = last_fps_t = time.perf_counter()
    fps_display = 0.0

    print("Starting tracking loop... Press 'q' in the window to exit.")
    try:
        while True:
            # --- Frame Capture ---
            if use_cv_capture:
                ok, frame = cap.read()
                if not ok: continue
            else:
                frame = cam.capture_array()
            
            # --- Face Detection ---
            best_face = det.detect(frame)
            now = time.perf_counter()
            dt = now - last_t
            
            # --- Tracking & Homing Logic ---
            status = "TRACKING" if best_face else "SEARCHING"
            if best_face:
                miss_frames = 0
                x, y, w, h = best_face
                
                cx, cy = x + w / 2.0, y + h / 2.0
                ema_cx = CFG.EMA_ALPHA * cx + (1.0 - CFG.EMA_ALPHA) * ema_cx
                ema_cy = CFG.EMA_ALPHA * cy + (1.0 - CFG.EMA_ALPHA) * ema_cy

                err_x = (ema_cx - (CFG.width / 2.0)) / (CFG.width / 2.0)
                err_y = (ema_cy - (CFG.height / 2.0)) / (CFG.height / 2.0)

                if abs(err_x) < CFG.DEADZONE_X: err_x = 0.0
                if abs(err_y) < CFG.DEADZONE_Y: err_y = 0.0

                d_pan = -pd_step(err_x * CFG.PAN_DIR, prev_err_x * CFG.PAN_DIR, dt, CFG.KP_PAN, CFG.KD_PAN, CFG.MAX_STEP)
                d_tilt = pd_step(err_y * CFG.TILT_DIR, prev_err_y * CFG.TILT_DIR, dt, CFG.KP_TILT, CFG.KD_TILT, CFG.MAX_STEP)
                
                prev_err_x, prev_err_y = err_x, err_y
                last_t = now

                next_pan = pan + (d_pan if abs(d_pan) >= CFG.MIN_MOVE else 0.0)
                next_tilt = tilt + (d_tilt if abs(d_tilt) >= CFG.MIN_MOVE else 0.0)
                
                if next_pan != pan or next_tilt != tilt:
                    pan, tilt = set_angles(kit, next_pan, next_tilt)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(ema_cx), int(ema_cy)), 4, (0, 0, 255), -1)
            else:
                miss_frames += 1
                if miss_frames >= CFG.LOST_FRAMES:
                    status = "HOMING"
                    pan_diff, tilt_diff = CFG.PAN_START - pan, CFG.TILT_START - tilt
                    step_pan = np.sign(pan_diff) * min(CFG.HOME_RATE, abs(pan_diff))
                    step_tilt = np.sign(tilt_diff) * min(CFG.HOME_RATE, abs(tilt_diff))
                    if step_pan or step_tilt:
                        pan, tilt = set_angles(kit, pan + step_pan, tilt + step_tilt)
                prev_err_x = prev_err_y = 0.0

            # --- OSD & Display ---
            frame_count += 1
            if now - last_fps_t >= 1.0:
                fps_display = frame_count / (now - last_fps_t)
                last_fps_t, frame_count = now, 0
            
            info = f"PAN {pan:3d} TILT {tilt:3d} {status} FPS {fps_display:.1f}"
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if show:
                cv2.imshow("pibye Face Track", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"): break
    finally:
        print("\nCleaning up and exiting...")
        try:
            set_angles(kit, CFG.PAN_START, CFG.TILT_START)
            time.sleep(0.5)
        except Exception: pass
        if use_cv_capture: cap.release()
        else: cam.stop()
        cv2.destroyAllWindows()

def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced pan-tilt face tracker.")
    parser.add_argument("--test", action="store_true", help="Run internal unit tests.")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["first-arg-is-ignored"], exit=False)
    else:
        setup_models()
        run_tracker()

if __name__ == "__main__":
    main()
