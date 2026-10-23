#!/usr/bin/env python3
"""
stereo_calibrate.py
===================

README — STEREO CALIBRATION FROM A SIDEWAYS SLIDE
-------------------------------------------------

WHAT THIS DOES
  Reads pairs of checkerboard photos taken at the two stops on the rail.
  The phone's own intrinsics stay fixed at the Module 2 calibration.
  The new result is the rotation and the translation between the two stops.
  The length of that translation should match the distance you marked on the rail.

HOW TO RUN (from the repository root)
  .venv/bin/python HW5/stereo_calibrate.py --left HW5/calibration/left --right HW5/calibration/right --baseline 20 --unit cm

  Matching file names are one pair (left/board03.jpg with right/board03.jpg).
  Shoot the board, slide, shoot again, then move the board and repeat.
  15 to 20 pairs is enough. The board must be found in both photos.

OUTPUT
  HW5/stereo_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT / "HW1", ROOT / "HW5"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cv_core import UNIT_TO_METERS, detect_checkerboard, load_image  # noqa: E402
from stereo_core import calibrate_stereo, checkerboard_object, load_monocular, scale_intrinsics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the slide between two phone positions.")
    parser.add_argument("--left", type=Path, default=Path("HW5/calibration/left"))
    parser.add_argument("--right", type=Path, default=Path("HW5/calibration/right"))
    parser.add_argument("--baseline", type=float, default=None, help="Marked slide length, for comparison only.")
    parser.add_argument("--unit", default="cm", choices=sorted(UNIT_TO_METERS))
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=1.0, help="Checkerboard square, in --unit.")
    parser.add_argument("--min-pairs", type=int, default=8)
    args = parser.parse_args()

    names = sorted(p.name for p in args.left.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    if not names:
        raise SystemExit(f"No images in {args.left}")
    pattern = (args.pattern_cols, args.pattern_rows)
    square_m = args.square_size * UNIT_TO_METERS[args.unit]
    obj = checkerboard_object(pattern, square_m)
    mono = load_monocular()
    object_points, img1, img2 = [], [], []
    image_size = None
    for name in names:
        right_path = args.right / name
        if not right_path.is_file():
            print(f"skip {name}: no matching right photo")
            continue
        try:
            left = load_image(args.left / name)
            right = load_image(right_path)
        except ValueError:
            print(f"skip {name}: could not read")
            continue
        if left.shape[:2] != right.shape[:2]:
            print(f"skip {name}: the two photos differ in size")
            continue
        height, width = left.shape[:2]
        if image_size is None:
            image_size = (width, height)
        if (width, height) != image_size:
            print(f"skip {name}: size {width}x{height} differs from {image_size[0]}x{image_size[1]}")
            continue
        c1 = detect_checkerboard(left, pattern)
        c2 = detect_checkerboard(right, pattern)
        if c1 is None or c2 is None:
            print(f"skip {name}: checkerboard not found in both")
            continue
        object_points.append(obj)
        img1.append(c1.reshape(-1, 1, 2).astype(np.float32))
        img2.append(c2.reshape(-1, 1, 2).astype(np.float32))
        print(f"pair {name}")
    if image_size is None or len(object_points) < args.min_pairs:
        raise SystemExit(f"Need at least {args.min_pairs} pairs; found {len(object_points)}.")
    k = scale_intrinsics(mono["k"], mono["image_size"], image_size)
    result = calibrate_stereo(object_points, img1, img2, k, mono["dist"], image_size)
    result["square_size_m"] = square_m
    result["pattern"] = list(pattern)
    result["num_pairs"] = len(object_points)
    if args.baseline is not None:
        marked = args.baseline * UNIT_TO_METERS[args.unit]
        result["marked_baseline_m"] = marked
        result["baseline_error_m"] = result["baseline_m"] - marked
    out = ROOT / "HW5" / "stereo_calibration.json"

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    out.write_text(json.dumps({key: convert(val) for key, val in result.items()}, indent=2), encoding="utf-8")
    print(f"RMS {result['rms_px']:.3f} px")
    print(f"|T| = {result['baseline_m'] * 100:.2f} cm")
    if args.baseline is not None:
        print(f"marked slide {args.baseline} {args.unit}, difference {result['baseline_error_m'] * 100:.2f} cm")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
