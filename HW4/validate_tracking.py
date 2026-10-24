#!/usr/bin/env python3
"""
validate_tracking.py
====================

README — CHECK THE TRACKING EQUATIONS ON TWO REAL FRAMES
---------------------------------------------------------

WHAT THIS DOES
  Takes two consecutive frames. It predicts where corner points move, using
  the from-scratch Lucas–Kanade tracker and OpenCV's pyramidal tracker, and
  cross-checks both with sub-pixel template matching. If you pass clicked
  ground truth for frame 2 (``--clicks``), it prints the pixel errors.
  One point is written out in full: the 5×5 window, the gradients, AᵀA, Aᵀb,
  the Gauss–Newton step, and the bilinear weights.

HOW TO RUN (from the repository root)
  .venv/bin/python HW4/validate_tracking.py --frame1 a.jpg --frame2 b.jpg
  .venv/bin/python HW4/validate_tracking.py --video clip.mov --time 3.5 --clicks clicks.json --save

clicks.json
  {"frame1": [[x, y], ...], "frame2": [[x, y], ...]}
  frame2 clicks are the true locations, in the same order. Omit the file to
  let the script pick corners itself and compare the three predictors only.

OUTPUT
  HW4/results/tracking_<name>.txt and .json
  With --save, also tracking_<name>_frame1.jpg and tracking_<name>_frame2.jpg
  (frames scaled to --width, and clicks scaled by the same factor).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for folder in (SCRIPT_DIR, ROOT / "HW1"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import cv_core  # noqa: E402
from flow_core import evaluate_pair, good_features, gray_f64, resize_width  # noqa: E402


def _scale_frame(frame: np.ndarray, points: np.ndarray | None, max_width: int):
    width = frame.shape[1]
    resized = resize_width(frame, max_width)
    scale = resized.shape[1] / width
    if points is not None and scale != 1.0:
        points = np.asarray(points, dtype=np.float64) * scale
    return resized, points


def frames_from_video(path: Path, time_sec: float) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    index = int(round(time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok1, frame1 = cap.read()
    ok2, frame2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        raise SystemExit("Need two frames at that time.")
    return frame1, frame2


def _jsonable(result: dict) -> dict:
    example = result["example"]

    def arr(value):
        if value is None:
            return None
        return np.asarray(value).tolist()

    return {
        "points": arr(result["points"]),
        "scratch": arr(result["scratch"]),
        "opencv": arr(result["opencv"]),
        "template": arr(result["template"]),
        "ground_truth": arr(result["ground_truth"]),
        "scratch_error": arr(result["scratch_error"]),
        "opencv_error": arr(result["opencv_error"]),
        "template_error": arr(result["template_error"]),
        "eigenvalues": arr(result["eigenvalues"]),
        "example_index": result["example_index"],
        "example_text": result["example_text"],
        "ATA": np.asarray(example["ATA"]).tolist(),
        "ATb": np.asarray(example["ATb"]).tolist(),
        "step_uv": np.asarray(example["step_uv"]).tolist(),
        "eigenvalues_example": np.asarray(example["eigenvalues"]).tolist(),
    }


def _stats(errors) -> str:
    if errors is None:
        return "no clicked ground truth"
    e = np.asarray(errors, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return "no finite errors"
    return f"n={e.size}  mean={e.mean():.3f}px  median={np.median(e):.3f}px  max={e.max():.3f}px"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Lucas–Kanade on two frames.")
    parser.add_argument("--frame1")
    parser.add_argument("--frame2")
    parser.add_argument("--video")
    parser.add_argument("--time", type=float, default=0.0, help="Time of the first frame, in seconds.")
    parser.add_argument("--clicks", help="JSON with frame1 and frame2 point lists.")
    parser.add_argument("--points", type=int, default=10, help="How many corners to pick when --clicks is omitted.")
    parser.add_argument("--out", default="")
    parser.add_argument("--save", action="store_true", help="Write the two frames as JPEGs next to the JSON.")
    parser.add_argument("--width", type=int, default=960, help="With --save, scale frames to this width.")
    parser.add_argument("--label", default="", help="Short name stored in the JSON for the report.")
    parser.add_argument("--note", default="", help="Note stored in the JSON (for example, that the clip is synthetic).")
    args = parser.parse_args()

    if args.video:
        frame1, frame2 = frames_from_video(Path(args.video), args.time)
        stem = Path(args.video).stem
    elif args.frame1 and args.frame2:
        frame1 = cv_core.load_image(args.frame1)
        frame2 = cv_core.load_image(args.frame2)
        stem = Path(args.frame1).stem
    else:
        raise SystemExit("Pass --video or both --frame1 and --frame2.")

    ground_truth = None
    points = None
    if args.clicks:
        payload = json.loads(Path(args.clicks).read_text(encoding="utf-8"))
        points = np.asarray(payload["frame1"], dtype=np.float64)
        ground_truth = np.asarray(payload["frame2"], dtype=np.float64)
    if args.save:
        frame1, points = _scale_frame(frame1, points, args.width)
        frame2, ground_truth = _scale_frame(frame2, ground_truth, args.width)
    if points is None:
        points = good_features(gray_f64(frame1).astype(np.uint8), max_corners=args.points)

    result = evaluate_pair(frame1, frame2, points, ground_truth)
    lines = [
        "From-scratch Lucas–Kanade vs clicked frame 2: " + _stats(result["scratch_error"]),
        "OpenCV pyramidal LK vs clicked frame 2:      " + _stats(result["opencv_error"]),
        "Template match vs clicked frame 2:           " + _stats(result["template_error"]),
        "",
        "Worked example (first Gauss–Newton step, 5×5 window):",
        result["example_text"],
    ]
    text = "\n".join(lines)
    print(text)
    out = Path(args.out) if args.out else SCRIPT_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"tracking_{stem}.txt").write_text(text, encoding="utf-8")
    payload = _jsonable(result)
    payload["summary"] = "\n".join(lines[:3])
    if args.label:
        payload["label"] = args.label
    if args.note:
        payload["note"] = args.note
    (out / f"tracking_{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.save:
        cv2.imwrite(str(out / f"tracking_{stem}_frame1.jpg"), frame1, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(out / f"tracking_{stem}_frame2.jpg"), frame2, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\nWrote {out / f'tracking_{stem}.txt'}")


if __name__ == "__main__":
    main()
