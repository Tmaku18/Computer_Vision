#!/usr/bin/env python3
"""
calibrate_video.py
==================

README — CALIBRATE THE PHONE'S VIDEO MODE
-----------------------------------------

WHAT THIS DOES
  Video mode crops and stabilizes the frame, so the photo calibration does
  not apply to a video. This script pulls sharp checkerboard frames out of a
  short clip and runs the same Zhang calibration as Module 2.

HOW TO SHOOT
  Settings → Camera → Record Video → 1080p at 30 fps. Turn off Enhanced
  Stabilization or Action mode if the phone shows it. Film the checkerboard
  for about 40 seconds, tilted and at several distances, filling different
  parts of the frame.

HOW TO RUN (from the repository root)
  .venv/bin/python HW5/calibrate_video.py --video checkerboard.mov --square-size 1 --unit in

OUTPUT
  HW5/video_calibration.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT / "HW1", ROOT / "HW5"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cv_core import UNIT_TO_METERS, calibrate, detect_checkerboard  # noqa: E402


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def collect_frames(path: Path, pattern: tuple[int, int], step: int, max_frames: int) -> list[tuple[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    found: list[tuple[float, int, np.ndarray]] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0 and detect_checkerboard(frame, pattern) is not None:
            found.append((sharpness(frame), index, frame))
        index += 1
    cap.release()
    found.sort(key=lambda item: item[0], reverse=True)
    # Keep the sharpest frames, but not two that are almost the same moment.
    chosen: list[tuple[str, np.ndarray]] = []
    used_index: list[int] = []
    for score, frame_index, frame in found:
        if any(abs(frame_index - prev) < step * 3 for prev in used_index):
            continue
        chosen.append((f"frame_{frame_index:05d}", frame))
        used_index.append(frame_index)
        if len(chosen) >= max_frames:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate video mode from a checkerboard clip.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument("--unit", default="in", choices=sorted(UNIT_TO_METERS))
    parser.add_argument("--step", type=int, default=5, help="Check every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=25)
    parser.add_argument("--min-images", type=int, default=12)
    args = parser.parse_args()
    pattern = (args.pattern_cols, args.pattern_rows)
    frames = collect_frames(args.video, pattern, args.step, args.max_frames)
    print(f"{len(frames)} sharp checkerboard frames")
    if len(frames) < args.min_images:
        raise SystemExit(f"Need at least {args.min_images} detections. Film the board longer, from more angles.")
    calib, views = calibrate(
        frames,
        pattern=pattern,
        square_size=args.square_size,
        square_units=args.unit,
        min_images=args.min_images,
    )
    out = ROOT / "HW5" / "video_calibration.json"
    payload = calib.to_json()
    payload["note"] = "Video mode. Do not reuse the still-photo K on these frames."
    out.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    kept = sum(1 for view in views if view.detected)
    print(f"RMS {calib.rms_px:.3f} px on {kept} frames, image {calib.image_width}x{calib.image_height}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
