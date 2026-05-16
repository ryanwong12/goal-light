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

    # Stage 2a: substrings that match the CANADIENS wordmark
    # "ANADIE" is the stable core even when edges are cropped by the animation
    canadiens_strings: tuple = ("ANADIE", "CANADIEN", "CANADIENS")

    # Stage 2b: substrings that match the GOAL banner (must follow 2a)
    goal_strings: tuple = ("GOAL",)

    # Frames per second to sample (2 is plenty, keeps CPU low)
    sample_fps: int = 10

    # Seconds after seeing CANADIENS to still accept a GOAL frame.
    # The animation runs ~2-3s so 5s gives comfortable headroom.
    canadiens_window_seconds: float = 5.0

    # Seconds to ignore further triggers after a goal fires
    goal_cooldown_seconds: int = 60

    start_time_str: str = time.strftime('%Y%m%d_%H%M%S')


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


def ocr_matches(roi_frame: np.ndarray, substrings: tuple) -> bool:
    """
    Run Tesseract and check whether any of the given substrings appear.
    """
    processed = preprocess_for_ocr(roi_frame)
    text = pytesseract.image_to_string(processed, config="--psm 7").upper().strip()
    if text:
        log.debug(f"OCR read: {text!r}")
    return any(s in text for s in substrings)


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def fire_goal(cap, frame, frame_num, source_fps, goals_detected, goal_callback, start_time_str=Config.start_time_str):
    """
    Compute timestamp, save frame, and invoke callback.
    Extracted so it can be called from one place regardless of which
    path through the state machine confirmed the goal.
    """
    try:
        timestamp_seconds = frame_num / (source_fps or 1)
    except Exception:
        timestamp_seconds = 0.0

    timestamp_str = time.strftime("%H:%M:%S", time.gmtime(timestamp_seconds))

    try:
        save_dir = f"dev/detected/{start_time_str}"
        os.makedirs(save_dir, exist_ok=True)
        safe_ts = timestamp_str.replace(":", "-")
        filename = f"goal_{goals_detected:03d}_frame{frame_num}_{safe_ts}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, frame)
        log.info(f"Saved detected frame to {save_path}")
    except Exception as e:
        log.warning(f"Failed to save detected frame: {e}")

    try:
        goal_callback(timestamp_seconds, timestamp_str)
    except TypeError:
        try:
            goal_callback(timestamp_seconds)
        except TypeError:
            goal_callback()


def run(config: Config, goal_callback=None):
    """
    Main loop. Calls goal_callback() when an MTL goal graphic is detected.

    Detection requires a two-step sequence within a time window:
      1. ROI turns red AND OCR sees the CANADIENS wordmark.
      2. ROI is still red AND OCR sees the GOAL banner.
    Both steps must occur within `canadiens_window_seconds` of each other.

    Special case: if CANADIENS and GOAL are both readable in the same frame
    (common mid-animation), step 2 is satisfied immediately.
    """
    if goal_callback is None:
        goal_callback = lambda ts, ts_str: print(f"GOAL! 🚨 time: {ts_str}")

    cap = cv2.VideoCapture(config.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_source!r}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if not cap.get(cv2.CAP_PROP_FPS):
        log.warning("Could not determine source FPS, defaulting to 30")
    else:
        log.info(
            f"Source video {config.video_source!r} has {total_frames} frames "
            f"at {cap.get(cv2.CAP_PROP_FPS):.2f} FPS"
        )

    source_fps = math.ceil(cap.get(cv2.CAP_PROP_FPS) or 30)
    frame_interval = max(1, int(source_fps / config.sample_fps))
    log.info(f"Source FPS: {source_fps} — sampling every {frame_interval} frames (~{config.sample_fps}fps)")

    x, y, w, h = config.roi
    frame_num = 0
    goals_detected = 0
    last_goal_time = 0.0
    ocr_calls = 0

    # State machine:
    #   None             → watching for CANADIENS wordmark
    #   "canadiens_seen" → saw wordmark, now watching for GOAL banner
    sequence_state = None
    canadiens_seen_at = 0.0

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
            now = time.time()

            # ------------------------------------------------------------------
            # Stage 1: cheap red gate — skip OCR entirely if not red
            # ------------------------------------------------------------------
            if not is_mostly_red(roi_frame, config.red_threshold):
                # Expire a stale canadiens_seen state when the red clears
                if sequence_state == "canadiens_seen":
                    elapsed = now - canadiens_seen_at
                    if elapsed > config.canadiens_window_seconds:
                        log.debug("CANADIENS window expired (non-red frame) — resetting.")
                        sequence_state = None
                continue

            # ------------------------------------------------------------------
            # Stage 2: OCR — frame is red, check what it says
            # ------------------------------------------------------------------
            ocr_calls += 1

            # Run OCR once and check both strings to avoid a double call
            # on the same frame (handles mid-animation overlap).
            sees_canadiens = ocr_matches(roi_frame, config.canadiens_strings)
            sees_goal      = ocr_matches(roi_frame, config.goal_strings)

            # ------------------------------------------------------------------
            # State: waiting for CANADIENS wordmark
            # ------------------------------------------------------------------
            if sequence_state is None:
                if not sees_canadiens:
                    # Red but not CANADIENS — ignore (other team's graphic, etc.)
                    continue

                # CANADIENS confirmed.
                sequence_state = "canadiens_seen"
                canadiens_seen_at = now
                log.debug("CANADIENS wordmark detected — waiting for GOAL banner.")

                # Fall through: check if GOAL is also visible in this same frame.
                # (The `sees_goal` branch below handles it.)

            # ------------------------------------------------------------------
            # State: CANADIENS seen, waiting for GOAL banner
            # ------------------------------------------------------------------
            if sequence_state == "canadiens_seen":
                # Check window expiry first
                if now - canadiens_seen_at > config.canadiens_window_seconds:
                    log.debug("CANADIENS window expired — resetting.")
                    sequence_state = None
                    # Don't continue — re-evaluate this frame from scratch
                    # in case it's a fresh CANADIENS hit.
                    if sees_canadiens:
                        sequence_state = "canadiens_seen"
                        canadiens_seen_at = now
                        log.debug("Re-armed CANADIENS on the same frame.")
                    else:
                        continue

                if not sees_goal:
                    continue  # Still waiting for the GOAL banner

                # ---- GOAL CONFIRMED (CANADIENS → GOAL sequence) ----

                # Cooldown check
                if (now - last_goal_time) < config.goal_cooldown_seconds:
                    remaining = config.goal_cooldown_seconds - (now - last_goal_time)
                    log.debug(f"Goal sequence detected but in cooldown ({remaining:.0f}s remaining)")
                    sequence_state = None
                    continue

                # Fire!
                sequence_state = None
                goals_detected += 1
                last_goal_time = now
                log.info(f"*** GOAL #{goals_detected} CONFIRMED (CANADIENS → GOAL sequence) ***")
                fire_goal(cap, frame, frame_num, source_fps, goals_detected, goal_callback)

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