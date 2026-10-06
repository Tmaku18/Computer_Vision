#!/usr/bin/env python3
"""
segment_thermal.py
==================

README — PART 2: OUTLINE A PERSON IN A THERMAL IMAGE (CLASSICAL OPENCV)
------------------------------------------------------------------------

People are warmer than most of the scene, so they are bright. The pipeline
boosts local contrast (CLAHE), blurs a little, thresholds with Otsu, and
keeps connected components that are large enough and person-shaped.
--invert is for palettes that draw hot objects as dark.
A box from prompts.json selects which person, the same box SAM2 sees.

HOW TO RUN (from the repository root)
  .venv/bin/python HW3/segment_thermal.py --prompts HW3/prompts.json
  .venv/bin/python HW3/segment_thermal.py --image thermal.png --box 200,30,160,300
  .venv/bin/python HW3/segment_thermal.py --image thermal.png --invert --min-area 400

  --out   directory for the mask, the outline, and metrics (default HW3/results)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

import segment_core


def _parse_box(text: str):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--box must be x,y,width,height")
    return [float(p) for p in parts]


def _write(out: Path, stem: str, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{stem}_mask.png"), result["mask"])
    cv2.imwrite(str(out / f"{stem}_outline.png"), result["overlay"])
    print(f"wrote {out / f'{stem}_mask.png'}")
    print(f"wrote {out / f'{stem}_outline.png'}")
    if result["comparison"] is not None:
        cv2.imwrite(str(out / f"{stem}_compare.png"), result["comparison"])
    if result["metrics"] is not None:
        path = out / f"{stem}_metrics.json"
        path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        metrics = result["metrics"]
        print(
            f"  IoU {metrics['iou']:.3f}  Dice {metrics['dice']:.3f}  "
            f"boundary F {metrics['boundary_f']:.3f}  "
            f"Hausdorff {metrics['hausdorff_px']}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classical person outline on a thermal image.")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--box", help="x,y,width,height. Optional; without it every person-shaped blob is kept.")
    parser.add_argument("--invert", action="store_true", help="Hot objects are dark in this palette.")
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--max-aspect", type=float, default=4.5)
    parser.add_argument("--split", action="store_true", help="Try to split people who are touching.")
    parser.add_argument("--out", type=Path, default=Path("HW3/results"))
    args = parser.parse_args(argv)
    root = segment_core.repo_root()

    if args.prompts:
        prompts = [p for p in segment_core.load_prompts(args.prompts) if p.get("kind") == "thermal"]
        if not prompts:
            raise SystemExit(f"No thermal prompts in {args.prompts}")
        for prompt in prompts:
            prompt["method"] = "thermal"
            result = segment_core.run_prompt(prompt, root)
            _write(args.out, Path(prompt["image"]).stem, result)
        return

    if args.image is None:
        raise SystemExit("Pass --prompts or --image.")
    prompt = {
        "image": str(args.image),
        "kind": "thermal",
        "method": "thermal",
        "box": None if args.box is None else _parse_box(args.box),
        "invert": args.invert,
        "max_aspect": args.max_aspect,
        "split": args.split,
    }
    if args.min_area is not None:
        prompt["min_area"] = args.min_area
    result = segment_core.run_prompt(prompt, root)
    _write(args.out, args.image.stem, result)


if __name__ == "__main__":
    main(sys.argv[1:])
