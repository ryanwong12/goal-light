# Goal Light — Project Setup & Usage

Detects Montreal Canadiens goals from a hockey broadcast video feed (file or live HDMI capture) and triggers a GPIO output (goal light). Uses a three-stage sequence detector: Habs logo → CANADIENS wordmark → GOAL banner.

---

## Prerequisites (new machine setup)

```bash
sudo apt update
sudo apt install python3-full python3-venv python3-pip tesseract-ocr
```

---

## First-Time Project Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### 2. Install dependencies

```bash
python -m pip install opencv-python pytesseract yt-dlp
```

(Optional) Save dependencies:

```bash
pip freeze > requirements.txt
```

---

## Daily Usage

```bash
source .venv/bin/activate   # activate on every new terminal
python score_detector.py    # run the detector
deactivate                  # when done
```

### VSCode Setup

`Ctrl + Shift + P` → `Python: Select Interpreter` → select `.venv/bin/python`

### Verify Installation

```python
import cv2
import pytesseract
import yt_dlp
print("all good")
```

---

## Getting Sample Video

Use `yt-dlp` to download a broadcast VOD for development. Always download from the same broadcaster (Sportsnet) and resolution you plan to use, since ROI coordinates are resolution-specific.

```bash
# List available formats
yt-dlp --list-formats "<youtube_url>"

# Download 720p60 with audio (recommended — use format IDs from list)
yt-dlp -f "298+140" "<youtube_url>" -o samples/game.mp4

# Download best available up to 720p (simpler, may be 30fps)
yt-dlp -f "best[height<=720]" "<youtube_url>" -o samples/game.mp4
```

> **Note:** 720p60 streams are video-only on YouTube. The `298+140` format merges video (298 = 720p60 h264) with audio (140 = 128k m4a). Requires `ffmpeg` to be installed (`winget install ffmpeg` on Windows).

---

## Project Structure

```
goal-light/
├── score_detector.py        # Main detector — import this, don't duplicate logic
├── debug_ocr.py             # OCR/state machine debugger (imports score_detector)
├── debug_detector.py        # Logo template match score debugger
├── extract_logo_template.py # Helper to crop a logo template from a saved frame
├── samples/                 # Downloaded VODs for development
├── dev/
│   ├── frames/              # Extracted still frames for calibration
│   ├── detected/            # Frames saved when a goal is confirmed
│   ├── debug_ocr/           # ROI crops saved by debug_ocr.py
│   └── template/            # Logo template image
└── requirements.txt
```

---

## Running the Detector

### File mode (development — sample video)

Cooldown is measured in **video time**, so the detector correctly handles all goals even when processing a file much faster than real time.

```bash
python score_detector.py -s samples/game.mp4 -f 15
```

| Flag              | Description                  | Default                            |
| ----------------- | ---------------------------- | ---------------------------------- |
| `-s` / `--source` | Video file path              | `samples/mtl_ott_110326_30fps.mp4` |
| `-f` / `--fps`    | Frames per second to sample  | `15`                               |
| `--live`          | Enable live mode (see below) | off                                |

### Live mode (HDMI capture card)

Cooldown switches to **wall-clock time** so it works correctly with a real-time feed.

```bash
python score_detector.py -s 0 --live
```

`-s 0` is the capture card device index (try `0`, `1`, `/dev/video0` etc. if it doesn't open).

---

## ROI Calibration

The ROI (Region of Interest) is the pixel rectangle around the score bug, excluding the SN watermark. It must be recalibrated if you change broadcaster, resolution, or source.

### 1. Extract a still frame

```bash
python -c "
import cv2
cap = cv2.VideoCapture('samples/game.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 5 * 60 * 1000)  # 5 minutes in
ret, frame = cap.read()
cv2.imwrite('dev/frames/calibration_frame.jpg', frame)
print(frame.shape)
cap.release()
"
```

### 2. Find coordinates

Open `dev/frames/calibration_frame.jpg` in **Paint on Windows** (accessible at `\\wsl$\Ubuntu\...`). Hover over the corners of the score bug area (excluding the SN logo on the left) and read the pixel coordinates from the status bar.

### 3. Update Config

In `score_detector.py`, set:

```python
roi: tuple = (x, y, w, h)  # x, y = top-left corner; w, h = width, height
```

w = right_x − left_x, h = bottom_y − top_y.

Current calibrated value for Sportsnet 720p: `(68, 22, 146, 29)`

---

## Logo Template (optional stage 1 detection)

The logo template enables a third detection stage (logo → CANADIENS → GOAL) for additional false-positive protection. Without it, the detector falls back to CANADIENS → GOAL automatically.

> **Important:** the template must be cropped from the **same video source and resolution** you are detecting against. A template from a fullscreen VOD won't match a 720p broadcast ROI.

### 1. Extract a frame showing the logo cleanly

Seek to just before a goal (the logo appears small and centered before it zooms):

```bash
python -c "
import cv2
cap = cv2.VideoCapture('samples/game.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 70 * 1000)  # adjust to just before a goal
ret, frame = cap.read()
cv2.imwrite('dev/frames/logo_frame.jpg', frame)
cap.release()
"
```

### 2. Find the logo coordinates in Paint

Open `dev/frames/logo_frame.jpg` in Paint and hover over the corners of just the logo (not the full ROI strip). Note the pixel coordinates.

### 3. Crop the template

```bash
python extract_logo_template.py \
  --frame dev/frames/logo_frame.jpg \
  --roi x,y,w,h \
  --out dev/template/habs_logo_template.png
```

`--roi` coordinates are relative to whichever image you pass as `--frame`. Open `dev/template/habs_logo_template.png` to confirm it looks like a clean, isolated logo crop.

### 4. Point the config at it

```python
logo_template_path: str = "dev/template/habs_logo_template.png"
```

> **Tip:** the template should be the logo at its **smallest** natural size in the animation. The detector tries multiple scales (±30%) so it can match as the logo zooms in.

---

## Debuggers

Both debuggers import directly from `score_detector.py` — they use the exact same detection logic, not a copy. Any change to `score_detector.py` is automatically reflected when you run the debuggers.

---

### `debug_ocr.py` — OCR and state machine debugger

The primary debugging tool. Shows the full state machine running frame by frame: red pixel ratio, OCR text, which stage matched, state transitions, and whether a goal would fire.

```bash
python debug_ocr.py \
  --source samples/game.mp4 \
  --seek 600 \
  --duration 30 \
  --fps 10
```

| Flag          | Description                          | Default      |
| ------------- | ------------------------------------ | ------------ |
| `--source`    | Video file path                      | required     |
| `--seek`      | Start time in seconds                | `0`          |
| `--duration`  | How many seconds to sample           | `60`         |
| `--fps`       | Frames per second to sample          | `10`         |
| `--threshold` | Override `red_threshold` from Config | Config value |

**Output columns:**

| Column             | Meaning                                  |
| ------------------ | ---------------------------------------- |
| `Frame`            | Frame number in the file                 |
| `Time`             | Video timestamp (HH:MM:SS)               |
| `Red%`             | Fraction of ROI pixels that are red      |
| `OCR text`         | What Tesseract read (only on red frames) |
| `Stage`            | Which stage matched, or why it didn't    |
| `State transition` | State machine before → after             |
| `🚨 GOAL FIRED`    | This frame would trigger the goal light  |

Also saves ROI crops to `dev/debug_ocr/` for every red frame:

- `*_raw.jpg` — what the camera sees (4× scaled)
- `*_ocr.jpg` — what Tesseract sees (inverted, 3× scaled)

**Common issues to look for:**

- **Red% never hits threshold** → lower `--threshold` (try `0.25`)
- **OCR text is garbled on red frames** → open `*_ocr.jpg` to inspect preprocessing
- **CANADIENS matches but GOAL never fires** → window too short, or GOAL frame is being missed; increase `canadiens_window_seconds` or `--fps`
- **Goal fires in debugger but not in score_detector** → was likely a cooldown issue (now fixed with video-time cooldown)

---

### `debug_detector.py` — logo template match score debugger

Use this when tuning logo detection. Prints the best template match score for every sampled frame and saves ROI crops so you can see what the template matcher is working with.

```bash
python debug_detector.py \
  --source samples/game.mp4 \
  --template dev/template/habs_logo_template.png \
  --seek 60 \
  --duration 30
```

| Flag         | Description                 | Default  |
| ------------ | --------------------------- | -------- |
| `--source`   | Video file path             | required |
| `--template` | Path to logo template image | required |
| `--seek`     | Start time in seconds       | `0`      |
| `--duration` | Seconds to sample           | `30`     |
| `--fps`      | Frames per second to sample | `5`      |

**Output columns:**

| Column       | Meaning                                             |
| ------------ | --------------------------------------------------- |
| `Frame`      | Frame number                                        |
| `Time`       | Video timestamp                                     |
| `Score`      | Best template match score (0–1)                     |
| `Scale`      | Scale at which the best score was found             |
| `Threshold?` | `*** YES ***` if score exceeds 0.72 (would trigger) |

Also saves to `dev/debug/`:

- Per-frame ROI crops (4× scaled) named with their match score
- `TEMPLATE.jpg` — your template at 4× scale for visual comparison

**What to look for:**

- **All scores near 0** → template is the wrong size (likely cropped from a different resolution), or coordinates are off
- **Scores around 0.5–0.65** → template is right but threshold too high; lower `logo_match_threshold` in Config
- **Scores spike above threshold on non-logo frames** → threshold too low; raise it or use a cleaner template crop

---

## Common Issues

### `externally-managed-environment` error

You're installing packages globally on Ubuntu 24.04+. Activate the venv first:

```bash
source .venv/bin/activate
```

### yt-dlp format not available

720p60 streams are video-only — use explicit format IDs:

```bash
yt-dlp -f "298+140" "<url>" -o samples/game.mp4
```

Run `yt-dlp --list-formats "<url>"` to find the right IDs for your video.

### ROI crops look black or wrong in debugger

The ROI coordinates are off. Re-run calibration — make sure `w` and `h` are width/height, not the bottom-right corner coordinates.

### All three goals not detected in file mode

Ensure you are **not** using `--live` flag when running against a sample file. Without `--live`, cooldown uses video time so all goals fire correctly regardless of processing speed.
