#!/usr/bin/env python3
"""
populate_calibration_images.py
==============================

README — COPY CHECKERBOARD PHOTOS INTO calibration_images/
-------------------------------------------------------------

HOW TO RUN
  cd /Users/home/GSU/Computer_Vision/Computer_Vision
  .venv/bin/python HW1/populate_calibration_images.py

  Add any extra photos until you have at least 20, then:
  .venv/bin/python HW1/calibrate_camera.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEST = SCRIPT_DIR / "calibration_images"

SOURCES = [
    Path("/Users/home/Downloads/IMG_5270.jpg"),
    Path("/Users/home/Downloads/IMG_5271.jpg"),
    Path("/Users/home/Downloads/IMG_5257.jpg"),
    Path("/Users/home/Downloads/IMG_5273.jpg"),
    Path("/Users/home/Downloads/IMG_5275.jpg"),
    Path("/Users/home/Downloads/IMG_5261.jpg"),
    Path("/Users/home/Downloads/compressed/IMG_5264.jpg"),
    Path("/Users/home/Downloads/compressed/IMG_5262.jpg"),
    Path("/Users/home/Downloads/compressed/IMG_5272.jpg"),
    Path("/Users/home/Downloads/compressed/IMG_5263.jpg"),
    Path("/Users/home/Downloads/compressed/IMG_5277.jpg"),
    Path("/Users/home/Downloads/compressed/IMG_5274.jpg"),
    Path("/Users/home/Downloads/IMG_5278.jpg"),
    Path("/Users/home/Downloads/IMG_5279.jpg"),
    Path("/Users/home/Downloads/IMG_5280.jpg"),
    Path("/Users/home/Downloads/IMG_5281.jpg"),
    Path("/Users/home/Downloads/IMG_5282.jpg"),
    Path("/Users/home/Downloads/IMG_5283.jpg"),
]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in SOURCES:
        if not src.is_file():
            print(f"Skip missing: {src}")
            continue
        dst = DEST / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    total = len(list(DEST.glob("*.*")))
    print(f"Copied {copied} new files. Total in {DEST}: {total}")
    if total < 20:
        print(f"Add {20 - total} more checkerboard photos, then run calibrate_camera.py")


if __name__ == "__main__":
    main()
