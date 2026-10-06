#!/usr/bin/env python3
"""
segment_rgb.py
==============

README — PART 1: OUTLINE A PERSON IN AN RGB PHOTO (CLASSICAL OPENCV)
---------------------------------------------------------------------

Draws the person's boundary with GrabCut (the default), marker-based
watershed, or background differencing. No neural network is loaded.
If a SAM2 mask already exists for the photo, the script also prints
IoU, Dice, boundary F, and Hausdorff distance.

The box is the same one SAM2 uses. It comes from --box or from prompts.json,
which the web app downloads after you drag the rectangle.

HOW TO RUN (from the repository root)
  .venv/bin/python HW3/segment_rgb.py --prompts HW3/prompts.json
  .venv/bin/python HW3/segment_rgb.py --image photo.jpg --box 120,40,200,360
  .venv/bin/python HW3/segment_rgb.py --image photo.jpg --background empty.jpg --method background --box 120,40,200,360

  --method grabcut | watershed | background
  --out    directory for the mask, the outline, and metrics (default HW3/results)
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
        print(f"wrote {out / f'{stem}_compare.png'}")
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
    parser = argparse.ArgumentParser(description="Classical person outline on an RGB photo.")
    parser.add_argument("--prompts", type=Path, help="prompts.json from the web app.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--box", help="x,y,width,height in pixels.")
    parser.add_argument("--background", type=Path, help="Empty frame, same camera position.")
    parser.add_argument("--method", default="grabcut", choices=("grabcut", "watershed", "background"))
    parser.add_argument("--out", type=Path, default=Path("HW3/results"))
    args = parser.parse_args(argv)
    root = segment_core.repo_root()

    if args.prompts:
        prompts = [p for p in segment_core.load_prompts(args.prompts) if p.get("kind", "rgb") == "rgb"]
        if not prompts:
            raise SystemExit(f"No RGB prompts in {args.prompts}")
        for prompt in prompts:
            if args.method and "method" not in prompt:
                prompt["method"] = args.method
            result = segment_core.run_prompt(prompt, root)
            _write(args.out, Path(prompt["image"]).stem, result)
        return

    if args.image is None or args.box is None:
        raise SystemExit("Pass --prompts, or both --image and --box.")
    prompt = {
        "image": str(args.image),
        "kind": "rgb",
        "method": args.method,
        "box": _parse_box(args.box),
    }
    if args.background:
        prompt["background"] = str(args.background)
    result = segment_core.run_prompt(prompt, root)
    _write(args.out, args.image.stem, result)


if __name__ == "__main__":
    main(sys.argv[1:])
