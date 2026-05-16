import cv2
import pytesseract
import numpy as np
import time
import os
import logging
import math
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # File path for development, integer device index for capture card (e.g. 0)
    video_source: str | int = "sample.mp4"

    # ROI excluding the SN watermark (x, y, w, h)
    roi: tuple = (68, 22, 146, 29)

    # Stage 1: minimum fraction of red pixels to bother running OCR
    red_threshold: float = 0.35

    # Stage 2: substrings that confirm an MTL goal graphic
    # "ANADIE" is the stable core of CANADIENS even when edges are cropped
    goal_strings: tuple = ("ANADIE", "CANADIEN", "GOAL")

    # Frames per second to sample (2 is plenty, keeps CPU low)
    sample_fps: int = 15

    # Number of consecutive positive detections before firing
    confirmation_frames: int = 2

    # Seconds to ignore further triggers after a goal fires
    goal_cooldown_seconds: int = 10


# ---------------------------------------------------------------------------
# Stage 1 — cheap red detection
# ---------------------------------------------------------------------------

def is_mostly_red(roi_frame: np.ndarray, threshold: float) -> bool:
    """
    Returns True if enough of the ROI is red.
    Red wraps around in HSV so we need two ranges.
    This is just a gate — not the trigger.
    """
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   120, 70), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_pixels = cv2.countNonZero(mask1 | mask2)
    total_pixels = roi_frame.shape[0] * roi_frame.shape[1]
    ratio = red_pixels / total_pixels
    return ratio > threshold


# ---------------------------------------------------------------------------
# Stage 2 — OCR confirmation
# ---------------------------------------------------------------------------

def preprocess_for_ocr(roi_frame: np.ndarray) -> np.ndarray:
    """
    Scale up and threshold for better Tesseract accuracy.
    White bold text on red background — invert so text is dark on light.
    """
    scaled = cv2.resize(roi_frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    # Invert: white text on red becomes dark text on light background
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
    return binary


def is_mtl_goal_graphic(roi_frame: np.ndarray, goal_strings: tuple) -> bool:
    """
    Run Tesseract and check for goal-related substrings.
    Catches fully visible and partially cropped/animated frames.
    """
    processed = preprocess_for_ocr(roi_frame)
    text = pytesseract.image_to_string(processed, config="--psm 7").upper().strip()
    if text:
        log.debug(f"OCR read: {text!r}")
    return any(s in text for s in goal_strings)


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run(config: Config, goal_callback=None):
    """
    Main loop. Calls goal_callback() when an MTL goal graphic is detected.
    Defaults to a print if no callback provided.
    """
    if goal_callback is None:
        goal_callback = lambda timestamp_seconds, timestamp_str : print(f"GOAL! 🚨 time: {timestamp_str}")

    cap = cv2.VideoCapture(config.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_source!r}")

    # Total frames (0 for live feeds) and source fps for timestamp calculation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    source_fps = math.ceil(cap.get(cv2.CAP_PROP_FPS) or 30)
    frame_interval = max(1, int(source_fps / config.sample_fps))
    log.info(f"Source FPS: {source_fps:.1f} — sampling every {frame_interval} frames (~{config.sample_fps}fps)")

    x, y, w, h = config.roi
    frame_num = 0
    goals_detected = 0
    last_goal_time = 0.0
    consecutive_hits = 0
    ocr_calls = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of video or feed lost.")
                break

            frame_num += 1
            if frame_num % frame_interval != 0:
                continue

            roi_frame = frame[y:y + h, x:x + w]

            # --- Stage 1: red gate (cheap, runs every sampled frame) ---
            if not is_mostly_red(roi_frame, config.red_threshold):
                consecutive_hits = 0
                continue

            # --- Stage 2: OCR (only runs when ROI is red) ---
            ocr_calls += 1
            if not is_mtl_goal_graphic(roi_frame, config.goal_strings):
                consecutive_hits = 0
                continue

            consecutive_hits += 1
            log.debug(f"Hit streak: {consecutive_hits}/{config.confirmation_frames}")

            if consecutive_hits < config.confirmation_frames:
                continue

            # --- Confirmed goal ---
            now = time.time()
            if (now - last_goal_time) < config.goal_cooldown_seconds:
                remaining = config.goal_cooldown_seconds - (now - last_goal_time)
                log.debug(f"Goal detected but in cooldown ({remaining:.0f}s remaining)")
                continue

            goals_detected += 1
            last_goal_time = now
            consecutive_hits = 0
            log.info(f"*** GOAL #{goals_detected} CONFIRMED ***")
            # Compute timestamp (seconds) based on the current frame and source FPS.
            # Use frame_num (1-based count of frames read) and source_fps recorded earlier.
            try:
                timestamp_seconds = frame_num / (source_fps or 1)
            except Exception:
                timestamp_seconds = 0.0

            # Human-friendly HH:MM:SS using UTC (video-relative)
            timestamp_str = time.strftime("%H:%M:%S", time.gmtime(timestamp_seconds))

            # Save the full frame for inspection
            try:
                save_dir = "dev/detected"
                os.makedirs(save_dir, exist_ok=True)
                safe_ts = timestamp_str.replace(":", "-")
                filename = f"goal_{goals_detected:03d}_frame{frame_num}_{safe_ts}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, frame)
                log.info(f"Saved detected frame to {save_path}")
            except Exception as e:
                log.warning(f"Failed to save detected frame: {e}")

            # Call the callback with best-effort compatibility:
            # prefer (seconds, formatted), fall back to (seconds), then to no-arg.
            try:
                goal_callback(timestamp_seconds, timestamp_str)
            except TypeError:
                try:
                    goal_callback(timestamp_seconds)
                except TypeError:
                    goal_callback()

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        cap.release()
        log.info(
            f"Done. Frames processed: {frame_num} | "
            f"OCR calls: {ocr_calls} | "
            f"Goals detected: {goals_detected}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config(
        video_source="samples/mtl_ott_110326_30fps.mp4",
        roi=(68, 22, 146, 29),
        sample_fps=30,
    )
    run(config)