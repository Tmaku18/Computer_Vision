#!/usr/bin/env python3
"""
measure_stereo.py
=================

README — OBJECT SIZE FROM A CALIBRATED STEREO PAIR
--------------------------------------------------

WHAT THIS DOES
  Takes the left and right photo from one slide along the rail, plus the
  corners (or rim points) you clicked. It finds the depth Z of each point
  from the disparity, then the width and height of a rectangle, the diameter
  of a circle, or every edge of another polygon.

HOW TO RUN (from the repository root)
  .venv/bin/python HW5/measure_stereo.py --shape rectangle --baseline 20 --unit cm
  .venv/bin/python HW5/measure_stereo.py --left L.jpg --right R.jpg --clicks clicks.json --shape circle --baseline 20 --unit cm

CLICKS FILE
  {"left": [[u, v], ...], "right": [[u, v], ...]}
  Omit "right" and the script matches each left click along its row.
  Order for a rectangle: around the outline. For a circle: 5 or more rim points.

OUTPUT
  Prints the sizes and writes HW5/results/measurement.json
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

from cv_core import UNIT_TO_METERS, load_image  # noqa: E402
from stereo_core import (  # noqa: E402
    load_demo,
    load_monocular,
    match_points,
    measure_points,
    result_to_json,
    scale_intrinsics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure an object from a stereo pair.")
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--clicks", type=Path)
    parser.add_argument("--shape", default="rectangle", choices=["rectangle", "circle", "polygon"])
    parser.add_argument("--baseline", type=float, default=20.0)
    parser.add_argument("--unit", default="cm", choices=sorted(UNIT_TO_METERS))
    args = parser.parse_args()
    baseline = args.baseline * UNIT_TO_METERS[args.unit]
    if args.left and args.right:
        try:
            left = load_image(args.left)
            right = load_image(args.right)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        clicks = json.loads(args.clicks.read_text(encoding="utf-8")) if args.clicks else None
        if clicks is None:
            raise SystemExit("Pass --clicks for your own photos.")
        mono = load_monocular()
        height, width = left.shape[:2]
        k = scale_intrinsics(mono["k"], mono["image_size"], (width, height))
        left_xy = np.asarray(clicks["left"], dtype=np.float64)
        if "right" in clicks:
            right_xy = np.asarray(clicks["right"], dtype=np.float64)
        else:
            right_xy, _found = match_points(left, right, left_xy)
    else:
        left, right, meta = load_demo()
        k = np.asarray(meta["k"], dtype=np.float64)
        left_xy = np.asarray(meta["shapes"][args.shape], dtype=np.float64)
        right_xy = left_xy.copy()
        right_xy[:, 0] -= meta["disparity_px"]
        baseline = float(meta["baseline_m"])
        print("Using the built-in sample pair.")
    result = measure_points(left_xy, right_xy, k, baseline, args.shape)
    result["unit"] = args.unit
    text = json.dumps(result_to_json(result), indent=2)
    out = ROOT / "HW5" / "results" / "measurement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.shape == "rectangle":
        print(f"width {result['width_m']:.4f} m, height {result['height_m']:.4f} m")
    elif args.shape == "circle":
        print(f"diameter {result['diameter_m']:.4f} m")
    else:
        print("edges m:", ", ".join(f"{e:.4f}" for e in result["edges_m"]))
    print(f"mean Z {result['mean_z_m']:.3f} m")
    print(f"disparity vs triangulation, max gap {result['triangulation_gap_m']:.3e} m")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
