#!/usr/bin/env python3
"""
calibrate_camera.py
===================

README — STEP 1: CAMERA CALIBRATION (OpenCV, 20+ SMARTPHONE PHOTOS)
--------------------------------------------------------------------

WHAT THIS DOES
  Detects the checkerboard in every photo in calibration_images/ (JPEG, PNG or
  iPhone HEIC), runs OpenCV's calibrateCamera, and writes:
    camera_calibration.json   intrinsics K, distortion (k1, k2), image size
    calibration_report.json   RMS reprojection error, per-photo errors, skipped photos

  The same calibration is used by measure_object_calibrated.py,
  validate_measurements.py and the web application.

SETUP
  1. Print the checkerboard, measure one square with a ruler.
  2. Take 20+ photos with the phone in LANDSCAPE (same orientation for all),
     varying tilt and distance, and copy them into HW1/calibration_images/.
  3. Set the inner-corner count and square size in calibration_config.json
     (default: 9 x 6 inner corners, 1 inch squares).

HOW TO RUN (from the repository root)
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python HW1/calibrate_camera.py

  Options:
  .venv/bin/python HW1/calibrate_camera.py --folder HW1/calibration_images --min-images 20
  .venv/bin/python HW1/calibrate_camera.py --pattern-cols 9 --pattern-rows 6 --square-size 1.0 --units in
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cv_core import calibrate, load_image  # noqa: E402

DEFAULT_FOLDER = SCRIPT_DIR / "calibration_images"
DEFAULT_CONFIG = SCRIPT_DIR / "calibration_config.json"
OUT_JSON = SCRIPT_DIR / "camera_calibration.json"
OUT_REPORT = SCRIPT_DIR / "calibration_report.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def load_config(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    inner = cfg.get("checkerboard_inner_corners", [9, 6])

    parser = argparse.ArgumentParser(description="OpenCV checkerboard calibration from a folder.")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--pattern-cols", type=int, default=int(inner[0]))
    parser.add_argument("--pattern-rows", type=int, default=int(inner[1]))
    parser.add_argument("--square-size", type=float, default=float(cfg.get("square_size", 1.0)))
    parser.add_argument("--units", default=cfg.get("square_size_units", "in"))
    parser.add_argument("--min-images", type=int, default=int(cfg.get("min_calibration_images", 20)))
    args = parser.parse_args()

    files = sorted(p for p in args.folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"No images in {args.folder}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} images in {args.folder}")

    def images():
        for path in files:
            print(f"  detecting {path.name}")
            yield path.name, load_image(path)

    try:
        calib, views = calibrate(
            images(),
            pattern=(args.pattern_cols, args.pattern_rows),
            square_size=args.square_size,
            square_units=args.units,
            min_images=args.min_images,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    OUT_JSON.write_text(json.dumps(calib.to_json(), indent=2) + "\n", encoding="utf-8")
    report = {
        "num_images_found": len(views),
        "num_images_used": calib.num_images,
        "overall_reproj_rms_px": calib.rms_px,
        "per_view_errors": [
            {"image": v.name, "used": v.detected, "reproj_rms_px": v.reproj_rms_px, "note": v.note}
            for v in views
        ],
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    k = calib.camera_matrix
    print(f"\nImages used: {calib.num_images}/{len(views)}   RMS reprojection error: {calib.rms_px:.3f} px")
    print(f"fx={k[0, 0]:.1f}  fy={k[1, 1]:.1f}  cx={k[0, 2]:.1f}  cy={k[1, 2]:.1f}")
    print(f"k1={calib.dist_coeffs[0]:.4f}  k2={calib.dist_coeffs[1]:.4f}")
    for v in views:
        if not v.detected:
            print(f"  skipped {v.name}: {v.note}")
    print(f"Wrote {OUT_JSON}\nWrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
