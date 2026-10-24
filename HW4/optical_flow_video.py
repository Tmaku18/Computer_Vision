#!/usr/bin/env python3
"""
optical_flow_video.py
=====================

README — DENSE OPTICAL FLOW FOR A CLIP
---------------------------------------

WHAT THIS DOES
  Takes a video, keeps ``--duration`` seconds starting at ``--start``, and
  writes browser-playable H.264 videos of Farnebäck flow (color wheel and
  arrows) plus sparse Lucas–Kanade trails. Motion statistics go to stats.json:
  mean speed, direction histogram, moving-object mask, ego-motion, focus of
  expansion, and a parallax cue.

HOW TO RUN (from the repository root)
  .venv/bin/python HW4/optical_flow_video.py --video clip.mov --start 0 --duration 30
  .venv/bin/python HW4/optical_flow_video.py --video clip.mov --start 12.5 --duration 30 --width 960

OUTPUT (HW4/results/flow_<name>/)
  original.mp4  flow_hsv.mp4  flow_arrows.mp4  tracks.mp4  side_by_side.mp4
  moving_mask.png  stats.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flow_core import process_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense optical flow for a clip of a video.")
    parser.add_argument("--video", required=True, help="Path to a video file.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--duration", type=float, default=30.0, help="Length of the clip in seconds.")
    parser.add_argument("--width", type=int, default=960, help="Frames are scaled to this width.")
    parser.add_argument("--out", default="", help="Output directory. Default: HW4/results/flow_<video name>.")
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else SCRIPT_DIR / "results" / f"flow_{video.stem}"

    def progress(fraction, message):
        print(f"\r{message:40s} {fraction * 100:5.1f}%", end="", flush=True)

    summary = process_video(video, out, args.start, args.duration, args.width, progress=progress)
    print()
    print(summary["interpretation"])
    print(f"Wrote {summary['frames']} frames to {out}")


if __name__ == "__main__":
    main()
