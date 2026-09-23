#!/usr/bin/env python3
"""
measure_object_calibrated.py
============================

README — MEASURE REAL 2D OBJECT SIZE USING YOUR CALIBRATION
-------------------------------------------------------------

WHAT THIS DOES
  1. Loads intrinsics from camera_calibration.json (calibrate_camera.py or MATLAB export).
  2. Removes lens distortion from the clicked pixels, then applies the planar homography
       λ [u, v, 1]^T = H [X, Y, 1]^T
  3. Prints edge lengths in the units of the reference (inches by default).

BEFORE YOU RUN
  1. Run Step 1 once: .venv/bin/python HW1/calibrate_camera.py
  2. Pick reference + object pixels on your photo with pick_measurement_points.py,
     or edit HW1/example_measurements.json (distorted pixels from the original photo).

HOW TO RUN (from the repository root)
  .venv/bin/pip install -r requirements.txt
  .venv/bin/python HW1/pick_measurement_points.py --image /path/to/photo.jpg
  .venv/bin/python HW1/measure_object_calibrated.py --json HW1/measurement_points.json

  .venv/bin/python HW1/measure_object_calibrated.py --json HW1/example_measurements.json

  Synthetic test (no calibration file):
  .venv/bin/python HW1/measure_planar_object_2d.py --demo

  The web application (Module 2 > Step 2) runs the same code with a click-to-measure UI.

INPUT JSON FIELDS
  reference_world_xy      — N×2 world points on the plane (inches, etc.)
  reference_image_uv      — N×2 distorted pixel coordinates (u, v)
  object_image_uv         — M×2 distorted pixel coordinates for the object
  world_units             — optional; overridden by calibration file if present
  closed_polygon          — optional, default true
  calibration_file        — optional path to camera_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cv_core import undistort_points  # noqa: E402
from measure_planar_object_2d import measure_with_reference_correspondences  # noqa: E402

DEFAULT_CALIBRATION = SCRIPT_DIR / "camera_calibration.json"
DEFAULT_MEASUREMENTS = SCRIPT_DIR / "measurement_points.json"
if not DEFAULT_MEASUREMENTS.is_file():
    DEFAULT_MEASUREMENTS = SCRIPT_DIR / "example_measurements.json"


def load_calibration_json(path: Path) -> dict:
    """Load K, distortion coefficients, and world units from exported MATLAB JSON."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    k = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    return {
        "camera_matrix": k,
        "dist_coeffs": dist[:5],
        "image_size": tuple(data.get("image_size", [])),
        "world_units": data.get("world_units", "mm"),
    }


def undistort_image_points(
    points_uv: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Convert distorted pixel clicks to undistorted pixel coordinates (Newton inversion)."""
    return undistort_points(points_uv, camera_matrix, dist_coeffs)


def run_measurement(
    calibration_path: Path,
    measurements_path: Path,
) -> dict:
    cal = load_calibration_json(calibration_path)
    with measurements_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    ref_world = np.asarray(cfg["reference_world_xy"], dtype=np.float64)
    ref_dist = np.asarray(cfg["reference_image_uv"], dtype=np.float64)
    obj_dist = np.asarray(cfg["object_image_uv"], dtype=np.float64)
    closed = cfg.get("closed_polygon", True)
    units = cfg.get("world_units", cal["world_units"])

    k = cal["camera_matrix"]
    dist = cal["dist_coeffs"]

    ref_undist = undistort_image_points(ref_dist, k, dist)
    obj_undist = undistort_image_points(obj_dist, k, dist)

    result = measure_with_reference_correspondences(
        ref_world,
        ref_undist,
        obj_undist,
        world_units=units,
        closed=closed,
    )
    result["calibration_file"] = str(calibration_path)
    result["measurements_file"] = str(measurements_path)
    result["reference_image_uv_undistorted"] = ref_undist.tolist()
    result["object_image_uv_undistorted"] = obj_undist.tolist()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure planar object size using MATLAB-exported camera calibration."
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help=f"Path to camera_calibration.json (default: {DEFAULT_CALIBRATION.name})",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_MEASUREMENTS,
        help=f"Measurement point pairs JSON (default: {DEFAULT_MEASUREMENTS.name})",
    )
    args = parser.parse_args()

    if not args.calibration.is_file():
        print(
            f"Calibration not found: {args.calibration}\n"
            "Run Calibration1.m in MATLAB first to generate cameraParams.mat and "
            "camera_calibration.json.",
            file=sys.stderr,
        )
        return 1
    if not args.json.is_file():
        print(f"Measurements JSON not found: {args.json}", file=sys.stderr)
        return 1

    out = run_measurement(args.calibration, args.json)
    print(json.dumps(out, indent=2))

    edges = out.get("edge_lengths", [])
    if len(edges) >= 2:
        print(
            f"\nSummary: edges[0] x edges[1] = {edges[0]:.4f} x {edges[1]:.4f} "
            f"{out.get('world_units', '')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
