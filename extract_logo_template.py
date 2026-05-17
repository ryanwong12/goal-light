"""
Run this once to crop the Habs logo from a saved frame and save it as the
template used by score_detector.py stage 1.

Usage:
    python extract_logo_template.py --frame dev/frames/frame_000300.jpg

It will open the frame, let you draw a rectangle around the logo, then save
the crop to dev/habs_logo_template.png.

On WSL (no display): pass --roi x,y,w,h to skip the interactive picker and
crop directly. Find the coordinates by opening the frame in Paint on Windows.

    python extract_logo_template.py --frame dev/frames/frame_000300.jpg --roi 30,2,55,25
"""

import cv2
import os
import argparse
import numpy as np


def extract_interactive(frame_path: str, out_path: str):
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {frame_path!r}")

    # Show just the ROI strip so the selection is easier
    roi_strip = img[22:51, 68:214]   # default ROI coords — adjust if needed
    print("Draw a rectangle around the logo, then press SPACE or ENTER.")
    rect = cv2.selectROI("Select logo", roi_strip, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = rect
    if w == 0 or h == 0:
        print("No selection made — exiting.")
        return

    crop = roi_strip[y:y+h, x:x+w]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, crop)
    print(f"Saved template ({w}x{h}px) → {out_path}")


def extract_from_roi(frame_path: str, out_path: str, roi: tuple):
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {frame_path!r}")
    x, y, w, h = roi
    crop = img[y:y+h, x:x+w]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, crop)
    print(f"Saved template ({w}x{h}px) → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="Path to a saved frame showing the logo clearly")
    parser.add_argument("--roi", default=None, help="x,y,w,h — skip interactive picker (use on WSL)")
    parser.add_argument("--out", default="dev/habs_logo_template.png")
    args = parser.parse_args()

    if args.roi:
        coords = tuple(int(v) for v in args.roi.split(","))
        extract_from_roi(args.frame, args.out, coords)
    else:
        extract_interactive(args.frame, args.out)