# Project Setup

This project is developed in **Ubuntu WSL** using a Python virtual environment (`venv`) to avoid Ubuntu Python package restrictions (`externally-managed-environment` / PEP 668).

## Prerequisites (new machine setup)

Install required system packages:

```bash
sudo apt update
sudo apt install python3-full python3-venv python3-pip tesseract-ocr
```

Verify Python is installed:

```bash
python3 -V
```

Expected output (or similar):

```bash
Python 3.x.x
```

---

## First-Time Project Setup

### 1. Create a virtual environment

From the project root:

```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Install dependencies

```bash
python -m pip install opencv-python pytesseract yt-dlp
```

(Optional) Save dependencies:

```bash
pip freeze > requirements.txt
```

---

## Daily Usage

Every time you open the repo in a new terminal:

### Activate the environment

```bash
source .venv/bin/activate
```

### Run the project

Example:

```bash
python main.py
```

### Exit the environment

```bash
deactivate
```

---

## VSCode Setup

After opening the repo in VSCode:

1. Press `Ctrl + Shift + P`
2. Search: `Python: Select Interpreter`
3. Select:

```bash
.venv/bin/python
```

This ensures VSCode uses the correct Python environment and installed packages.

---

## Verify Installation

Run Python:

```bash
python
```

Then test imports:

```python
import cv2
import pytesseract
import yt_dlp

print("all good")
```

If this prints `all good`, setup is working.

---

## Common Issue: `externally-managed-environment`

If you see:

```bash
error: externally-managed-environment
```

You are likely trying to install packages globally on Ubuntu 24.04+.

**Fix:** activate the virtual environment first:

```bash
source .venv/bin/activate
```

Then install packages again.

Avoid using:

```bash
--break-system-packages
```

unless you intentionally want system-wide installs.s

## Getting Sample Video

**yt-dlp** is your best friend here:

```bash
pip install yt-dlp
yt-dlp -f "best[height<=720]" "<youtube_url>" -o sample.mp4
```

You can use

```bash
yt-dlp --list-formats "<youtube_url>
```
