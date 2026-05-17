"""
Debug tool for OCR-based detection (CANADIENS → GOAL sequence).

Samples frames from the video, runs the red gate, OCR, and the full
state machine — so you can see exactly whether and why a goal would fire.

Usage:
    python debug_ocr.py --source samples/mtl_ott_110326.mp4 --seek 600 --duration 30

    --seek      seconds into the video to start (put this just before a goal)
    --duration  how many seconds to sample (default 60)
    --fps       frames per second to sample (default 10)
    --threshold red pixel threshold (default 0.35, try 0.25 if missing red frames)
"""

import cv2
import pytesseract
import numpy as np
import argparse
import os
import time

ROI = (68, 22, 146, 29)  # x, y, w, h

CANADIENS_STRINGS = ("ANADIE", "CANADIEN", "NADI", "CANADIENS")
GOAL_STRINGS      = ("GOAL",)
CANADIENS_WINDOW  = 5.0   # seconds
COOLDOWN          = 60.0  # seconds

IDLE           = "idle"
CANADIENS_SEEN = "canadiens_seen"


def is_mostly_red(roi_frame, threshold):
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   120, 70), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_pixels = cv2.countNonZero(mask1 | mask2)
    total = roi_frame.shape[0] * roi_frame.shape[1]
    ratio = red_pixels / total
    return ratio, ratio > threshold


def preprocess_for_ocr(roi_frame):
    scaled = cv2.resize(roi_frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
    return binary


def run_ocr(roi_frame):
    processed = preprocess_for_ocr(roi_frame)
    text = pytesseract.image_to_string(processed, config="--psm 7").upper().strip()
    return text, processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--seek",      type=float, default=0)
    parser.add_argument("--duration",  type=float, default=60)
    parser.add_argument("--fps",       type=int,   default=10)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.source!r}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Video: {source_fps:.0f}fps, {total_frames/source_fps/60:.1f} min total")

    if args.seek > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.seek * 1000)
        print(f"Seeked to {args.seek:.0f}s")

    frame_interval = max(1, int(source_fps / args.fps))
    max_frames = int(args.duration * source_fps)

    out_dir = "dev/debug_ocr"
    os.makedirs(out_dir, exist_ok=True)

    x, y, w, h = ROI
    frame_num  = 0
    saved      = 0
    goals_fired = 0

    # State machine
    state            = IDLE
    state_entered_at = time.time()
    last_goal_time   = 0.0

    header = f"{'Frame':>7}  {'Time':>8}  {'Red%':>6}  {'OCR text':<22}  {'Stage':<16}  {'State'}"
    print(f"\n{header}")
    print("─" * 85)

    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_num += 1
        if frame_num % frame_interval != 0:
            continue

        roi_frame = frame[y:y+h, x:x+w]
        video_ts  = args.seek + frame_num / source_fps
        ts = f"{int(video_ts//3600):02d}:{int((video_ts%3600)//60):02d}:{int(video_ts%60):02d}"
        now = time.time()
        elapsed = now - state_entered_at

        red_ratio, is_red = is_mostly_red(roi_frame, args.threshold)

        # ── non-red frame ────────────────────────────────────────────────────
        if not is_red:
            note = ""
            if state == CANADIENS_SEEN and elapsed > CANADIENS_WINDOW:
                note = f"  ← window expired, reset"
                state = IDLE
            print(f"{frame_num:>7}  {ts:>8}  {red_ratio:>5.1%}  {'':22}  {'':16}  {state}{note}")
            continue

        # ── red frame — run OCR ──────────────────────────────────────────────
        ocr_text, processed = run_ocr(roi_frame)
        is_canadiens = any(s in ocr_text for s in CANADIENS_STRINGS)
        is_goal      = any(s in ocr_text for s in GOAL_STRINGS)

        stage_label = "✓ CANADIENS" if is_canadiens else ("✓ GOAL" if is_goal else "✗ no match")

        # ── advance state machine ────────────────────────────────────────────
        prev_state  = state
        goal_marker = ""

        if state == IDLE:
            if is_canadiens:
                state = CANADIENS_SEEN
                state_entered_at = now

        elif state == CANADIENS_SEEN:
            if elapsed > CANADIENS_WINDOW:
                state = IDLE
                stage_label += "  (window expired)"
            elif is_goal:
                if (now - last_goal_time) < COOLDOWN:
                    stage_label += "  (cooldown)"
                else:
                    goals_fired += 1
                    last_goal_time = now
                    state = IDLE
                    goal_marker = "  🚨 GOAL FIRED"

        state_str = f"{prev_state} → {state}" if state != prev_state else state

        print(f"{frame_num:>7}  {ts:>8}  {red_ratio:>5.1%}  {ocr_text!r:<22}  {stage_label:<16}  {state_str}{goal_marker}")

        # ── save crops ───────────────────────────────────────────────────────
        if saved < 200:
            safe_ts = ts.replace(":", "-")
            tag = "CANADIENS" if is_canadiens else ("GOAL" if is_goal else "nomatch")
            big_roi = cv2.resize(roi_frame, (w*4, h*4), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{out_dir}/f{frame_num:06d}_{safe_ts}_{tag}_raw.jpg", big_roi)
            cv2.imwrite(f"{out_dir}/f{frame_num:06d}_{safe_ts}_{tag}_ocr.jpg", processed)
            saved += 1

    cap.release()
    print(f"\n{'─'*85}")
    print(f"Goals fired: {goals_fired}")
    print(f"Saved {saved} ROI crops to {out_dir}/")
    print("  *_raw.jpg — what the camera sees (4x)")
    print("  *_ocr.jpg — what Tesseract sees (inverted, 3x)")


if __name__ == "__main__":
    main()