import cv2
import pytesseract
import numpy as np
import time
import os
import logging
import math
import argparse
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
    logo_template_path: str = "dev/habs_logo_template.png"
    logo_match_threshold: float = 0.72
    logo_scale_range: tuple = (0.8, 1.3)
    logo_scale_steps: int = 10
    logo_window_seconds: float = 8.0

    # --- CANADIENS detection (stage 2) ---
    red_threshold: float = 0.35
    canadiens_strings: tuple = ("ANADIE", "CANADIEN", "NADI", "CANADIENS")
    canadiens_window_seconds: float = 5.0

    # --- GOAL detection (stage 3) ---
    goal_strings: tuple = ("GOAL",)

    # --- General ---
    sample_fps: int = 5
    goal_cooldown_seconds: int = 60


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

IDLE           = "idle"
LOGO_SEEN      = "logo_seen"
CANADIENS_SEEN = "canadiens_seen"


@dataclass
class DetectorState:
    state: str = IDLE
    state_entered_at: float = field(default_factory=time.time)
    last_goal_time: float = 0.0
    goals_detected: int = 0
    ocr_calls: int = 0
    logo_calls: int = 0


@dataclass
class FrameResult:
    """Everything process_frame() observed and decided for one frame."""
    # What the detector saw
    red_ratio: float = 0.0
    is_red: bool = False
    ocr_text: str = ""
    logo_score: float = 0.0
    # What stage matched
    canadiens_matched: bool = False
    goal_matched: bool = False
    logo_matched: bool = False
    # State machine
    prev_state: str = IDLE
    new_state: str = IDLE
    # Outcome
    goal_fired: bool = False
    cooldown_blocked: bool = False
    window_expired: bool = False


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------

def load_template(path: str) -> np.ndarray | None:
    tmpl = cv2.imread(path)
    if tmpl is None:
        log.warning(
            f"Logo template not found at {path!r}. "
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
) -> tuple[bool, float]:
    """Returns (matched, best_score)."""
    gray_roi  = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    th, tw = gray_tmpl.shape

    best_score = 0.0
    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
        sw, sh = int(tw * scale), int(th * scale)
        if sh > gray_roi.shape[0] or sw > gray_roi.shape[1] or sh < 4 or sw < 4:
            continue
        scaled = cv2.resize(gray_tmpl, (sw, sh))
        result = cv2.matchTemplate(gray_roi, scaled, cv2.TM_CCOEFF_NORMED)
        score  = float(result.max())
        if score > best_score:
            best_score = score
        if score > threshold:
            log.debug(f"Logo matched at scale {scale:.2f}, score {score:.3f}")
            return True, best_score

    log.debug(f"Logo best score: {best_score:.3f} (threshold {threshold})")
    return False, best_score


def red_ratio(roi_frame: np.ndarray) -> float:
    hsv   = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   120, 70), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red   = cv2.countNonZero(mask1 | mask2)
    total = roi_frame.shape[0] * roi_frame.shape[1]
    return red / total


def preprocess_for_ocr(roi_frame: np.ndarray) -> np.ndarray:
    scaled   = cv2.resize(roi_frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
    return binary


def run_ocr(roi_frame: np.ndarray) -> tuple[str, np.ndarray]:
    """Returns (uppercase text, preprocessed image)."""
    processed = preprocess_for_ocr(roi_frame)
    text = pytesseract.image_to_string(processed, config="--psm 7").upper().strip()
    if text:
        log.debug(f"OCR read: {text!r}")
    return text, processed


def ocr_matches(text: str, substrings: tuple) -> bool:
    return any(s in text for s in substrings)


# ---------------------------------------------------------------------------
# process_frame — the single source of truth for detection logic
# ---------------------------------------------------------------------------

def process_frame(
    roi_frame: np.ndarray,
    state: DetectorState,
    config: Config,
    template: np.ndarray | None,
) -> FrameResult:
    """
    Run one frame through the full detection pipeline.
    Mutates `state` in place and returns a FrameResult describing everything
    that happened — used by both run() and the debugger.
    """
    result = FrameResult(prev_state=state.state, new_state=state.state)
    now    = time.time()
    elapsed = now - state.state_entered_at
    has_template = template is not None

    # ── measure red ──────────────────────────────────────────────────────────
    result.red_ratio = red_ratio(roi_frame)
    result.is_red    = result.red_ratio > config.red_threshold

    # ── IDLE ─────────────────────────────────────────────────────────────────
    if state.state == IDLE:
        if has_template:
            state.logo_calls += 1
            matched, score = logo_detected(
                roi_frame, template,
                config.logo_match_threshold,
                config.logo_scale_range,
                config.logo_scale_steps,
            )
            result.logo_score   = score
            result.logo_matched = matched
            if matched:
                state.state = LOGO_SEEN
                state.state_entered_at = now
        else:
            # No template — jump straight to LOGO_SEEN on red
            if result.is_red:
                state.state = LOGO_SEEN
                state.state_entered_at = now

        result.new_state = state.state
        return result

    # ── LOGO_SEEN ─────────────────────────────────────────────────────────────
    if state.state == LOGO_SEEN:
        if elapsed > config.logo_window_seconds:
            result.window_expired = True
            state.state = IDLE
            state.state_entered_at = now
            result.new_state = state.state
            return result

        if not result.is_red:
            result.new_state = state.state
            return result

        state.ocr_calls += 1
        result.ocr_text, _ = run_ocr(roi_frame)
        result.canadiens_matched = ocr_matches(result.ocr_text, config.canadiens_strings)

        if result.canadiens_matched:
            state.state = CANADIENS_SEEN
            state.state_entered_at = now

        result.new_state = state.state
        return result

    # ── CANADIENS_SEEN ───────────────────────────────────────────────────────
    if state.state == CANADIENS_SEEN:
        if elapsed > config.canadiens_window_seconds:
            result.window_expired = True
            state.state = IDLE
            state.state_entered_at = now
            result.new_state = state.state
            return result

        if not result.is_red:
            result.new_state = state.state
            return result

        state.ocr_calls += 1
        result.ocr_text, _ = run_ocr(roi_frame)
        result.goal_matched = ocr_matches(result.ocr_text, config.goal_strings)

        if result.goal_matched:
            state.state = IDLE
            state.state_entered_at = now
            if (now - state.last_goal_time) < config.goal_cooldown_seconds:
                result.cooldown_blocked = True
            else:
                result.goal_fired = True
                state.goals_detected += 1
                state.last_goal_time = now

        result.new_state = state.state
        return result

    return result  # unreachable but satisfies linters


# ---------------------------------------------------------------------------
# run() — video loop, uses process_frame()
# ---------------------------------------------------------------------------

def run(config: Config, goal_callback=None):
    """
    Main loop. Calls goal_callback(timestamp_seconds, timestamp_str) on goals.
    """
    if goal_callback is None:
        goal_callback = lambda ts, ts_str: log.info(f"GOAL! 🚨  video time: {ts_str}")

    template     = load_template(config.logo_template_path)
    has_template = template is not None

    cap = cv2.VideoCapture(config.video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_source!r}")

    source_fps     = math.ceil(cap.get(cv2.CAP_PROP_FPS) or 30)
    frame_interval = max(1, int(source_fps / config.sample_fps))
    log.info(f"Source FPS: {source_fps} — sampling every {frame_interval} frames (~{config.sample_fps}fps)")
    if not has_template:
        log.info("No logo template — running CANADIENS → GOAL sequence only.")

    x, y, w, h = config.roi
    state       = DetectorState()
    frame_num   = 0

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
            result    = process_frame(roi_frame, state, config, template)

            if result.goal_fired:
                sequence = "logo → CANADIENS → GOAL" if has_template else "CANADIENS → GOAL"
                log.info(f"*** GOAL #{state.goals_detected} CONFIRMED ({sequence}) ***")

                try:
                    timestamp_seconds = frame_num / source_fps
                except Exception:
                    timestamp_seconds = 0.0
                timestamp_str = time.strftime("%H:%M:%S", time.gmtime(timestamp_seconds))

                try:
                    save_dir  = "dev/detected"
                    os.makedirs(save_dir, exist_ok=True)
                    safe_ts   = timestamp_str.replace(":", "-")
                    filename  = f"goal_{state.goals_detected:03d}_frame{frame_num}_{safe_ts}.jpg"
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
            f"Done. Frames: {frame_num} | "
            f"Logo checks: {state.logo_calls} | "
            f"OCR calls: {state.ocr_calls} | "
            f"Goals: {state.goals_detected}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect MTL goals in hockey broadcasts.")
    parser.add_argument("-s", "--source", type=str, default="samples/mtl_ott_110326_30fps.mp4")
    parser.add_argument("-f", "--fps",    type=int, default=15)
    args = parser.parse_args()

    config = Config(
        video_source=args.source,
        roi=(68, 22, 146, 29),
        sample_fps=args.fps,
    )
    run(config)