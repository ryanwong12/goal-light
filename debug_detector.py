"""
Debug tool for goal detection — imports and uses score_detector directly,
so it always reflects the exact same logic as the real detector.

Usage:
    python debug_ocr.py --source samples/mtl_ott_110326.mp4 --seek 600 --duration 30

    --seek      seconds into the video to start (just before a goal)
    --duration  seconds to sample (default 60)
    --fps       frames per second to sample (default 10 — higher than detector
                so you catch more animation frames)
    --threshold override the red threshold from Config (optional)
"""

import cv2
import numpy as np
import argparse
import os
import math
import time

from score_detector import (
    Config,
    DetectorState,
    FrameResult,
    process_frame,
    load_template,
    preprocess_for_ocr,
    red_ratio,
    IDLE, LOGO_SEEN, CANADIENS_SEEN,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--seek",      type=float, default=0)
    parser.add_argument("--duration",  type=float, default=60)
    parser.add_argument("--fps",       type=int,   default=10)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override red_threshold from Config")
    args = parser.parse_args()

    config = Config(
        video_source=args.source,
        roi=(68, 22, 146, 29),
        sample_fps=args.fps,
    )
    if args.threshold is not None:
        config.red_threshold = args.threshold

    template = load_template(config.logo_template_path)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.source!r}")
        return

    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Video: {source_fps:.0f}fps, {total_frames/source_fps/60:.1f} min total")

    if args.seek > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.seek * 1000)
        print(f"Seeked to {args.seek:.0f}s")

    frame_interval = max(1, int(source_fps / args.fps))
    max_frames     = int(args.duration * source_fps)

    out_dir = "dev/debug_ocr"
    os.makedirs(out_dir, exist_ok=True)

    x, y, w, h = config.roi
    state       = DetectorState()
    frame_num   = 0
    saved       = 0

    # Column headers
    print(f"\n{'Frame':>7}  {'Time':>8}  {'Red%':>6}  {'OCR text':<22}  {'Stage':<22}  {'State transition'}")
    print("─" * 100)

    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_num += 1
        if frame_num % frame_interval != 0:
            continue

        roi_frame  = frame[y:y+h, x:x+w]
        video_ts   = args.seek + frame_num / source_fps
        ts = f"{int(video_ts//3600):02d}:{int((video_ts%3600)//60):02d}:{int(video_ts%60):02d}"

        result = process_frame(roi_frame, state, config, template)

        # ── build readable stage label ───────────────────────────────────────
        if result.logo_matched:
            stage = "✓ LOGO"
        elif result.canadiens_matched:
            stage = "✓ CANADIENS"
        elif result.goal_matched:
            stage = "✓ GOAL"
        elif result.window_expired:
            stage = "⚠ window expired"
        elif result.cooldown_blocked:
            stage = "⚠ cooldown"
        elif result.is_red and result.ocr_text:
            stage = "✗ no match"
        elif result.is_red:
            stage = "(red, no OCR)"
        else:
            stage = ""

        state_change = (
            f"{result.prev_state} → {result.new_state}"
            if result.prev_state != result.new_state
            else result.new_state
        )

        goal_marker = "  🚨 GOAL FIRED" if result.goal_fired else ""

        print(
            f"{frame_num:>7}  {ts:>8}  {result.red_ratio:>5.1%}  "
            f"{result.ocr_text!r:<22}  {stage:<22}  {state_change}{goal_marker}"
        )

        # ── save ROI crops for red frames ────────────────────────────────────
        if result.is_red and saved < 200:
            safe_ts = ts.replace(":", "-")
            if result.canadiens_matched:
                tag = "CANADIENS"
            elif result.goal_matched:
                tag = "GOAL"
            elif result.logo_matched:
                tag = "LOGO"
            else:
                tag = "nomatch"

            # Raw ROI (4x for visibility)
            big_roi = cv2.resize(roi_frame, (w*4, h*4), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{out_dir}/f{frame_num:06d}_{safe_ts}_{tag}_raw.jpg", big_roi)

            # Preprocessed (what Tesseract sees)
            processed = preprocess_for_ocr(roi_frame)
            cv2.imwrite(f"{out_dir}/f{frame_num:06d}_{safe_ts}_{tag}_ocr.jpg", processed)
            saved += 1

    cap.release()
    print(f"\n{'─'*100}")
    print(f"Goals fired: {state.goals_detected}")
    print(f"OCR calls:   {state.ocr_calls}")
    print(f"Logo checks: {state.logo_calls}")
    print(f"Saved {saved} ROI crops to {out_dir}/")
    print("  *_raw.jpg — what the camera sees (4x)")
    print("  *_ocr.jpg — what Tesseract sees (inverted, 3x)")


if __name__ == "__main__":
    main()