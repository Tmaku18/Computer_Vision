#!/usr/bin/env python3
"""
room_dimensions.py
==================

README — ROOM LENGTH, WIDTH, AND HEIGHT FROM A SLIDE VIDEO
----------------------------------------------------------

WHAT THIS DOES
  Reads one clip in which the phone pauses, slides along the rail, and pauses
  again. It averages each pause, recovers the pose, scales it by the rail
  length, and builds a disparity map. If you pass the four corners of the
  wall, it reports that wall's width and height.

  Face the long wall for the room length. Face the short wall for the width.
  Both clips measure the height. Run this script once per wall.

HOW TO RUN (from the repository root)
  .venv/bin/python HW5/room_dimensions.py --video long_wall.mov --baseline 20 --unit cm --name length
  .venv/bin/python HW5/room_dimensions.py --video short_wall.mov --baseline 20 --unit cm --name width --clicks corners.json

CLICKS
  {"left": [[u, v], [u, v], [u, v], [u, v]], "right": [[u, v], ...]}
  Order: bottom-left, bottom-right, top-right, top-left, on the averaged left frame.
  Omit "right" to match each click along its row.

  Use HW5/video_calibration.json if you have run calibrate_video.py. Otherwise
  the script assumes a 65° horizontal field of view, which is only a rough K.

OUTPUT
  HW5/results/room_<name>/left.png right.png disparity.png room.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT / "HW1", ROOT / "HW5"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cv_core import UNIT_TO_METERS  # noqa: E402
from room_core import (  # noqa: E402
    colorize_disparity,
    default_video_k,
    k_for_frames,
    measure_room_frames,
    read_frames,
    stereo_pair_from_clip,
)
from stereo_core import load_monocular  # noqa: E402


def match_row_safe(left, right, points):
    from stereo_core import match_points

    return match_points(left, right, points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a wall from a pause-slide-pause video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--unit", default="cm", choices=sorted(UNIT_TO_METERS))
    parser.add_argument("--name", default="wall")
    parser.add_argument("--calibration", type=Path, default=Path("HW5/video_calibration.json"))
    parser.add_argument("--clicks", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=960)
    args = parser.parse_args()
    frames, fps = read_frames(args.video, max_side=args.max_side)
    pair = stereo_pair_from_clip(frames)
    left, right = pair["left"], pair["right"]
    if args.calibration.is_file():
        mono = load_monocular(args.calibration)
        k = k_for_frames(mono["k"], mono["image_size"], left)
        k_source = str(args.calibration)
    else:
        k = default_video_k(left)
        k_source = "assumed 65 degree horizontal field of view"
        print("No video calibration found. Run calibrate_video.py for a real K.")
    baseline = args.baseline * UNIT_TO_METERS[args.unit]
    clicks_l = clicks_r = None
    if args.clicks:
        payload = json.loads(args.clicks.read_text(encoding="utf-8"))
        clicks_l = np.asarray(payload["left"], dtype=np.float64)
        if "right" in payload:
            clicks_r = np.asarray(payload["right"], dtype=np.float64)
        else:
            clicks_r, _found = match_row_safe(left, right, clicks_l)
    measured = measure_room_frames(left, right, k, baseline, clicks_l, clicks_r)
    folder = ROOT / "HW5" / "results" / f"room_{args.name}"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "left.png"), left)
    cv2.imwrite(str(folder / "right.png"), right)
    cv2.imwrite(str(folder / "disparity.png"), colorize_disparity(measured["disparity"]))
    wall = measured.get("wall")
    body = {
        "name": args.name,
        "fps": fps,
        "baseline_m": baseline,
        "k_source": k_source,
        "k": k.tolist(),
        "R": measured["R"].tolist(),
        "t": measured["t"].tolist(),
        "inliers": measured["inliers"],
        "matches": measured["matches"],
        "median_depth_m": measured["median_depth_m"],
        "left_span": list(pair["left_span"]),
        "right_span": list(pair["right_span"]),
        "wall": None if wall is None else {
            "width_m": wall["width_m"],
            "height_m": wall["height_m"],
            "edges_m": wall["edges_m"].tolist(),
            "xyz": wall["xyz"].tolist(),
        },
    }
    (folder / "room.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"pose |t| = {np.linalg.norm(measured['t']):.3f} m, inliers {measured['inliers']}")
    if wall:
        print(f"wall width {wall['width_m']:.3f} m, height {wall['height_m']:.3f} m")
    if measured["median_depth_m"] is not None:
        print(f"median depth {measured['median_depth_m']:.2f} m")
    print(f"Wrote {folder}")


if __name__ == "__main__":
    main()
