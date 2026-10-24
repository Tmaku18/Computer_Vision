#!/usr/bin/env python3
"""
structure_from_motion.py
========================

README — FOUR-VIEW STRUCTURE FROM MOTION OF A PLANAR OBJECT
-----------------------------------------------------------

WHAT THIS DOES
  Undistorts four photographs with the Module 2 calibration, matches SIFT
  features, and combines them with boundary points you clicked in every view.
  It recovers the four camera poses, triangulates the boundary, scales the
  reconstruction with one tape-measured length, and fits the boundary to a plane.

HOW TO RUN (from the repository root)
  .venv/bin/python HW4/structure_from_motion.py \\
      --images v1.jpg v2.jpg v3.jpg v4.jpg \\
      --clicks sfm_points.json \\
      --known-length 11 --unit in

sfm_points.json
  {"boundary": [ [[u, v], ...], [[u, v], ...], [[u, v], ...], [[u, v], ...] ]}
  The same corners and edge points, in the same order, in all four photos.
  Optional "measured": {"height": 8.5, "unit": "in"} is reported next to the
  reconstructed edges but is not used to set the scale.

OUTPUT (HW4/results/sfm/)
  sfm.json   sfm_points.json   sfm_workout.txt
  sfm_boundary.png   sfm_cameras.png   view1.jpg … view4.jpg
  Optional camera_positions in the clicks file are copied into sfm.json
  for the report (tape-measured centers, object's frame).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for folder in (SCRIPT_DIR, ROOT / "HW1"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import cv_core  # noqa: E402
from sfm_core import format_worked_example, reconstruct, result_to_json  # noqa: E402

DEFAULT_CALIBRATION = ROOT / "HW1" / "camera_calibration.json"


def _draw(result: dict, out: Path) -> None:
    polygon = np.asarray(result["polygon"]["uv"], dtype=np.float64)
    closed = np.vstack([polygon, polygon[:1]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(closed[:, 0], closed[:, 1], "-o")
    ax.set_aspect("equal")
    ax.set_title("Boundary on the fitted plane")
    ax.set_xlabel("plane u (m)")
    ax.set_ylabel("plane v (m)")
    fig.tight_layout()
    fig.savefig(out / "sfm_boundary.png", dpi=140)
    plt.close(fig)

    centers = np.array([cam["center"] for cam in result["cameras"]], dtype=np.float64)
    pts = np.asarray(result["boundary_points"], dtype=np.float64)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="crimson", label="boundary")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="navy", s=40, label="cameras")
    for i, c in enumerate(centers):
        ax.text(c[0], c[1], c[2], f"  {i + 1}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sfm_cameras.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-view planar structure from motion.")
    parser.add_argument("--images", nargs=4, required=True, metavar="IMG")
    parser.add_argument("--clicks", required=True, help="JSON file of boundary clicks.")
    parser.add_argument("--known-length", type=float, required=True, help="Tape measurement of one edge.")
    parser.add_argument("--unit", default="in")
    parser.add_argument("--edge", nargs=2, type=int, default=(0, 1), metavar=("I", "J"),
                        help="Which boundary points the taped length joins. Default: the first edge.")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--out", default=str(SCRIPT_DIR / "results" / "sfm"))
    args = parser.parse_args()

    images = [cv_core.load_image(path) for path in args.images]
    payload = json.loads(Path(args.clicks).read_text(encoding="utf-8"))
    clicks = payload["boundary"]
    measured = payload.get("measured")
    result = reconstruct(
        images, clicks, args.calibration, args.known_length, args.unit,
        edge=(args.edge[0], args.edge[1]), measured_edges=measured,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    body = result_to_json(result)
    text = format_worked_example(result)
    body["workout"] = text
    body["clicks"] = clicks
    if payload.get("camera_positions"):
        body["camera_positions"] = payload["camera_positions"]
    if payload.get("note"):
        body["note"] = payload["note"]
    measured = payload.get("measured") if isinstance(payload.get("measured"), dict) else None
    if measured and measured.get("length") and measured.get("endpoints"):
        i, j = measured["endpoints"]
        edges = body["polygon_unit"]["edge_lengths"]
        n = len(edges)
        if (i + 1) % n == j:
            index = i
        elif (j + 1) % n == i:
            index = j
        else:
            index = None
        if index is not None:
            tape = float(measured["length"])
            body["check"] = {
                "tape": tape,
                "reconstructed": edges[index],
                "error": edges[index] - tape,
                "unit": measured.get("unit", args.unit),
            }
    (out / "sfm.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    (out / "sfm_points.json").write_text(Path(args.clicks).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "sfm_workout.txt").write_text(text, encoding="utf-8")
    for index, image in enumerate(images, start=1):
        cv2.imwrite(str(out / f"view{index}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _draw(result, out)
    print(text)
    print(f"\nWrote {out / 'sfm.json'}")


if __name__ == "__main__":
    main()
