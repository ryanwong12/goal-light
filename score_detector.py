import cv2
import pytesseract
import numpy as np
import time
import os
import logging
from dataclasses import dataclass, field

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

    # --- Logo detection (stage 1) ---
    # Path to the cropped logo template image.
    # Generate with: python extract_logo_template.py
    logo_template_path: str = "dev/habs_logo_template.png"

    # Match score threshold (0-1). Lower = more permissive.
    logo_match_threshold: float = 0.72

    # Scale range to try for multi-scale matching [min, max] and how many steps.
    logo_scale_range: tuple = (0.8, 1.3)
    logo_scale_steps: int = 10

    # Seconds after seeing the logo to still accept a CANADIENS frame.
    logo_window_seconds: float = 8.0

    # --- CANADIENS detection (stage 2) ---
    # Red pixel fraction needed to bother running OCR.
    red_threshold: float = 0.35

    # Substrings that match the wordmark (catches partial/animated frames).
    canadiens_strings: tuple = ("ANADIE", "CANADIEN")

    # Seconds after seeing CANADIENS to still accept a GOAL frame.
    canadiens_window_seconds: float = 5.0

    # --- GOAL detection (stage 3) ---
    goal_strings: tuple = ("GOAL",)

    # --- General ---
    # Frames per second to sample from the source.
    sample_fps: int = 5

    # Seconds to ignore further triggers after a goal fires.
    goal_cooldown_seconds: int = 60


# ---------------------------------------------------------------------------
# Stage 1 — logo template matching
# ---------------------------------------------------------------------------

def load_template(path: str) -> np.ndarray | None:
    """Load the logo template. Returns None if file not found."""
    tmpl = cv2.imread(path)
    if tmpl is None:
        log.warning(
            f"Logo template not found at {path!r}. "
            "Run extract_logo_template.py to generate it. "
            "Logo detection (stage 1) will be skipped."
        )
    else:
        log.info(f"Loaded logo template from {path!r} — shape {tmpl.shape}")
    return tmpl


def logo_detected(
    roi_frame: np.ndarray,
    template: np.ndarray,
    threshold: float,
    scale_range: tuple,
    scale_steps: int,
) -> bool:
    """
    Multi-scale template match against the ROI.
    Tries the template at several sizes to handle the slow zoom animation.
    Returns True if any scale exceeds the match threshold.
    """
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    th, tw = gray_tmpl.shape

    best_score = 0.0
    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
        sw, sh = int(tw * scale), int(th * scale)
        if sh > gray_roi.shape[0] or sw > gray_roi.shape[1]:
            continue
        if sh < 4 or sw < 4:
            continue
        scaled_tmpl = cv2.resize(gray_tmpl, (sw, sh))
        result = cv2.matchTemplate(gray_roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(result.max())
        if score > best_score:
            best_score = score
        if score > threshold:
            log.debug(f"Logo matched at scale {scale:.2f}, score {score:.3f}")
            return True

    log.debug(f"Logo best score: {best_score:.3f} (threshold {threshold})")
    return False


# ---------------------------------------------------------------------------
# Stage 2 — red gate + OCR
# ---------------------------------------------------------------------------

def is_mostly_red(roi_frame: np.ndarray, threshold: float) -> bool:
    """
    Returns True if enough of the ROI is red.
    Red wraps around in HSV so we need two ranges.
    Used as a cheap gate before running OCR.
    """
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   120, 70), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_pixels = cv2.countNonZero(mask1 | mask2)
    total_pixels = roi_frame.shape[0] * roi_frame.shape[1]
    return (red_pixels / total_pixels) > threshold


def preprocess_for_ocr(roi_frame: np.ndarray) -> np.ndarray:
    """
    Scale up and binarize for Tesseract.
    White bold text on red — invert so text is dark on light.
    """
    scaled = cv2.resize(roi_frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
    return binary


def ocr_matches(roi_frame: np.ndarray, substrings: tuple) -> bool:
    """Run Tesseract and check whether any of the given substrings appear."""
    processed = preprocess_for_ocr(roi_frame)
    text = pytesseract.image_to_string(processed, config="--psm 7").upper().strip()
    if text:
        log.debug(f"OCR read: {text!r}")
    return any(s in text for s in substrings)


# ---------------------------------------------------------------------------
# Sequence state machine
# ---------------------------------------------------------------------------
#
#   IDLE
#     └─ logo detected (or skipped if no template) ──► LOGO_SEEN
#
#   LOGO_SEEN
#     ├─ logo_window expires without red ─────────────► IDLE
#     └─ red + CANADIENS OCR match ───────────────────► CANADIENS_SEEN
#
#   CANADIENS_SEEN
#     ├─ canadiens_window expires ────────────────────► IDLE
#     └─ red + GOAL OCR match ────────────────────────► 🚨 GOAL FIRED → IDLE
#
# If no logo template is loaded, the machine starts at LOGO_SEEN immediately
# on every red frame, i.e. it falls back to the CANADIENS → GOAL sequence.
#

IDLE            = "idle"
LOGO_SEEN       = "logo_seen"
CANADIENS_SEEN  = "canadiens_seen"


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run(config: Config, goal_callback=None):
    """
    Main loop. Calls goal_callback(timestamp_seconds, timestamp_str) when
    an MTL goal sequence is detected. Defaults to a print if not provided.
    """
    if goal_callback is None:
        goal_callback = lambda ts, ts_str: log.info(f"GOAL! 🚨  video time: {ts_str}")

    template = load_template(config.logo_template_path)
    has_template = template is not None

    cap = cv2.VideoCapture(config.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_source!r}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(source_fps / config.sample_fps))
    log.info(f"Source FPS: {source_fps:.1f} — sampling every {frame_interval} frames (~{config.sample_fps}fps)")
    if not has_template:
        log.info("No logo template — running CANADIENS → GOAL sequence only.")

    x, y, w, h = config.roi
    frame_num = 0
    goals_detected = 0
    last_goal_time = 0.0
    ocr_calls = 0
    logo_calls = 0

    state = IDLE
    state_entered_at = 0.0   # wall-clock time of last state transition

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
            elapsed = now - state_entered_at

            # ------------------------------------------------------------------
            # IDLE — watch for the logo (or go straight to red if no template)
            # ------------------------------------------------------------------
            if state == IDLE:
                if has_template:
                    logo_calls += 1
                    if logo_detected(
                        roi_frame, template,
                        config.logo_match_threshold,
                        config.logo_scale_range,
                        config.logo_scale_steps,
                    ):
                        log.debug("Logo detected → waiting for CANADIENS.")
                        state = LOGO_SEEN
                        state_entered_at = now
                else:
                    # No template: treat every red frame as a potential start
                    if is_mostly_red(roi_frame, config.red_threshold):
                        state = LOGO_SEEN
                        state_entered_at = now
                continue

            # ------------------------------------------------------------------
            # LOGO_SEEN — watch for red + CANADIENS OCR
            # ------------------------------------------------------------------
            if state == LOGO_SEEN:
                if elapsed > config.logo_window_seconds:
                    log.debug("Logo window expired — resetting to IDLE.")
                    state = IDLE
                    continue

                if not is_mostly_red(roi_frame, config.red_threshold):
                    continue   # still in window, just not red yet

                ocr_calls += 1
                if ocr_matches(roi_frame, config.canadiens_strings):
                    log.debug("CANADIENS detected → waiting for GOAL.")
                    state = CANADIENS_SEEN
                    state_entered_at = now
                continue

            # ------------------------------------------------------------------
            # CANADIENS_SEEN — watch for red + GOAL OCR
            # ------------------------------------------------------------------
            if state == CANADIENS_SEEN:
                if elapsed > config.canadiens_window_seconds:
                    log.debug("CANADIENS window expired — resetting to IDLE.")
                    state = IDLE
                    continue

                if not is_mostly_red(roi_frame, config.red_threshold):
                    continue

                ocr_calls += 1
                if not ocr_matches(roi_frame, config.goal_strings):
                    continue

                # ── Sequence complete ──
                state = IDLE

                if (now - last_goal_time) < config.goal_cooldown_seconds:
                    remaining = config.goal_cooldown_seconds - (now - last_goal_time)
                    log.debug(f"Sequence matched but in cooldown ({remaining:.0f}s remaining).")
                    continue

                goals_detected += 1
                last_goal_time = now
                sequence = "logo → CANADIENS → GOAL" if has_template else "CANADIENS → GOAL"
                log.info(f"*** GOAL #{goals_detected} CONFIRMED ({sequence}) ***")

                # Timestamp relative to video file
                try:
                    timestamp_seconds = frame_num / source_fps
                except Exception:
                    timestamp_seconds = 0.0
                timestamp_str = time.strftime("%H:%M:%S", time.gmtime(timestamp_seconds))

                # Save detected frame for inspection
                try:
                    save_dir = "dev/detected"
                    os.makedirs(save_dir, exist_ok=True)
                    safe_ts = timestamp_str.replace(":", "-")
                    filename = f"goal_{goals_detected:03d}_frame{frame_num}_{safe_ts}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, frame)
                    log.info(f"Saved detected frame → {save_path}")
                except Exception as e:
                    log.warning(f"Could not save frame: {e}")

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
            f"Logo checks: {logo_calls} | "
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
        logo_template_path="dev/habs_logo_template.png",
    )
    run(config)