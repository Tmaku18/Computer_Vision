#!/usr/bin/env python3
"""
validate_measurements.py
========================

README — STEP 3: VALIDATE 20 MEASUREMENTS + ERROR STATISTICS
------------------------------------------------------------

WHAT THIS DOES
  Compares measured segment lengths against tape-measured ground truth for
  both measurement methods and reports MAE, RMSE, bias, standard deviation,
  95% confidence interval of the bias, mean absolute % error and max error.

    homography  plane reference (checkerboard or known rectangle) -> H^-1
    pinhole     fronto-parallel plane at the taped distance Z -> X = Z (u - cx) / fx

INPUT (either of)
  A) CSV exported from the web app (Module 2 > Step 3 > Export CSV). It already
     contains the estimates; this script recomputes the statistics:
       .venv/bin/python HW1/validate_measurements.py --csv validation_export.csv

  B) validation_config.json + validation_measurements.csv with raw pixels:
       validation_config.json:
         image_path            validation photo
         camera_distance_m     taped camera-to-plane distance (> 2 m)
         unit                  unit of ground truth and reference ("in", "cm", ...)
         reference             {"type": "checkerboard", "pattern": [9, 6], "square_size": 1.0}
                            or {"type": "rectangle", "width": 10, "height": 7,
                                "image_uv": [[u,v] x4 TL, TR, BR, BL]}
       validation_measurements.csv columns:
         measurement_id, ground_truth, u1, v1, u2, v2, description
       .venv/bin/python HW1/validate_measurements.py

  pick_validation_points.py fills the CSV interactively (2 clicks + typed length per row).

OUTPUT
  validation_report.json   per-measurement errors + statistics for each method
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cv_core import (  # noqa: E402
    CameraCalibration,
    convert_length,
    error_statistics,
    load_image,
    measure_homography,
    measure_pinhole,
    reference_from_checkerboard,
    reference_from_rectangle,
)

DEFAULT_CONFIG = SCRIPT_DIR / "validation_config.json"
DEFAULT_CSV = SCRIPT_DIR / "validation_measurements.csv"
DEFAULT_CALIB = SCRIPT_DIR / "camera_calibration.json"
OUT_REPORT = SCRIPT_DIR / "validation_report.json"


def _float(row: dict, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                return None
    return None


def rows_from_web_export(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        gt = _float(row, "ground_truth", "ground_truth_in")
        if not gt or gt <= 0:
            continue
        out.append({
            "measurement_id": row.get("measurement_id", ""),
            "image": row.get("image", ""),
            "description": row.get("description", ""),
            "ground_truth": gt,
            "homography": _float(row, "est_homography"),
            "pinhole": _float(row, "est_pinhole"),
        })
    return out


def rows_from_pixels(rows: list[dict], cfg: dict, calib: CameraCalibration) -> list[dict]:
    unit = cfg.get("unit", "in")
    image_path = Path(cfg["image_path"])
    if not image_path.is_absolute():
        image_path = SCRIPT_DIR.parent / image_path
    image = load_image(image_path)
    h_img, w_img = image.shape[:2]
    k, dist = calib.intrinsics_for(w_img, h_img)

    ref_cfg = cfg["reference"]
    if ref_cfg["type"] == "checkerboard":
        ref = reference_from_checkerboard(
            image, calib, tuple(ref_cfg.get("pattern", [9, 6])),
            float(ref_cfg.get("square_size", 1.0)), unit,
        )
    else:
        ref = reference_from_rectangle(
            np.asarray(ref_cfg["image_uv"]), float(ref_cfg["width"]), float(ref_cfg["height"]),
            calib, (w_img, h_img), unit,
        )
    print(
        f"Reference: {ref.kind}, reprojection {ref.reproj_rms_px:.2f} px, "
        f"PnP distance to plane {ref.distance_to_plane or float('nan'):.2f} {unit}"
    )

    depth_m = cfg.get("camera_distance_m")
    depth = convert_length(float(depth_m), "m", unit) if depth_m else None

    out = []
    for row in rows:
        gt = _float(row, "ground_truth", "ground_truth_in")
        pts = [_float(row, key) for key in ("u1", "v1", "u2", "v2")]
        if not gt or gt <= 0 or any(p is None for p in pts) or not any(pts):
            continue
        uv = np.array(pts, dtype=np.float64).reshape(2, 2)
        world_h = measure_homography(ref.homography, uv, k, dist)
        est_p = None
        if depth:
            world_p = measure_pinhole(uv, k, dist, depth)
            est_p = float(np.linalg.norm(world_p[1] - world_p[0]))
        out.append({
            "measurement_id": row.get("measurement_id", ""),
            "image": cfg["image_path"],
            "description": row.get("description", ""),
            "ground_truth": gt,
            "homography": float(np.linalg.norm(world_h[1] - world_h[0])),
            "pinhole": est_p,
        })
    return out


def build_report(measurements: list[dict], unit: str, distance_m) -> dict:
    report = {"unit": unit, "camera_distance_m": distance_m, "measurements": [], "statistics": {}}
    for m in measurements:
        entry = dict(m)
        for method in ("homography", "pinhole"):
            if m.get(method) is not None:
                entry[f"{method}_error"] = m[method] - m["ground_truth"]
        report["measurements"].append(entry)
    for method in ("homography", "pinhole"):
        pairs = [(m["ground_truth"], m[method]) for m in measurements if m.get(method) is not None]
        if pairs:
            gt, est = zip(*pairs)
            report["statistics"][method] = error_statistics(gt, est)
    return report


def print_statistics(report: dict) -> None:
    unit = report["unit"]
    for method, s in report["statistics"].items():
        print(f"\n{method.upper()}  (n = {s['n']})")
        print(f"  MAE            {s['mae']:.4f} {unit}")
        print(f"  RMSE           {s['rmse']:.4f} {unit}")
        print(f"  Bias (mean)    {s['bias']:+.4f} {unit}   95% CI {s['bias_ci95']}")
        print(f"  Std of error   {s['std']:.4f} {unit}")
        print(f"  Mean |% error| {s['mape_percent']:.2f} %   max {s['max_percent_error']:.2f} %")
        print(f"  Max |error|    {s['max_abs_error']:.4f} {unit}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate measurements vs ground truth.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--unit", default=None, help="Unit label for a web export (default: from file)")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if rows and "est_homography" in rows[0]:
        measurements = rows_from_web_export(rows)
        unit = args.unit or rows[0].get("unit") or "in"
        distance = rows[0].get("camera_distance_m")
    else:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        calib = CameraCalibration.load(args.calibration)
        try:
            measurements = rows_from_pixels(rows, cfg, calib)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        unit = cfg.get("unit", "in")
        distance = cfg.get("camera_distance_m")

    if not measurements:
        print("No valid rows (ground truth > 0 with pixel endpoints).", file=sys.stderr)
        return 1
    if len(measurements) < 20:
        print(f"Warning: {len(measurements)} valid rows; the assignment requires 20.", file=sys.stderr)

    report = build_report(measurements, unit, distance)
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_statistics(report)
    print(f"\nWrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
