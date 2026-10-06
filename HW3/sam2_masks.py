#!/usr/bin/env python3
"""
sam2_masks.py
=============

README — SAM2 MASKS, RUN ONCE ON YOUR MAC (NOT ON THE WEB HOST)
----------------------------------------------------------------

Prompts Meta's SAM2 with the same box stored in prompts.json and writes
a mask PNG per image to HW3/sam2_masks/. The web app and compare_sam2.py
read those PNGs. SAM2 is a large network, so it stays out of the hosted
app and out of requirements.txt.

SETUP (Python 3.12 — PyTorch may not have a wheel for 3.14)
  python3.12 -m venv .venv-sam2
  .venv-sam2/bin/pip install -r HW3/requirements-sam2.txt
  mkdir -p HW3/checkpoints
  curl -L -o HW3/checkpoints/sam2.1_hiera_small.pt \\
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

HOW TO RUN
  .venv-sam2/bin/python HW3/sam2_masks.py
  .venv-sam2/bin/python HW3/sam2_masks.py --prompts HW3/prompts.json --checkpoint HW3/checkpoints/sam2.1_hiera_small.pt

The script uses Apple MPS when PyTorch can see it, otherwise CPU.
The model is sam2.1_hiera_small.

IF INSTALLATION FAILS
  Open Meta's SAM2 demo in a browser, segment the person with the same
  box you drew in the web app, save the mask image, and upload that mask
  on the RGB or Thermal page. The comparison math does not care which
  program wrote the PNG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import segment_core

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
PRETRAINED_ID = "facebook/sam2.1-hiera-small"


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_predictor(checkpoint: Path, device: str):
    """Prefer a local checkpoint. Fall back to Hugging Face via from_pretrained."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise SystemExit(
            "The sam2 package is not installed in this Python.\n"
            "Create .venv-sam2 with Python 3.12 and install HW3/requirements-sam2.txt.\n"
            "Fallback: use Meta's online SAM2 demo, save the mask, and upload it on the Module 4 page.\n"
            f"Import error: {exc}"
        ) from exc

    if checkpoint.is_file():
        model = build_sam2(MODEL_CFG, str(checkpoint), device=device)
        return SAM2ImagePredictor(model)
    print(f"No checkpoint at {checkpoint}. Trying {PRETRAINED_ID} (downloads the weights).")
    return SAM2ImagePredictor.from_pretrained(PRETRAINED_ID, device=device)


def _predict(predictor, image_bgr: np.ndarray, box) -> np.ndarray:
    x, y, w, h = (float(v) for v in box)
    xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, _scores, _logits = predictor.predict(box=xyxy, multimask_output=False)
    mask = np.asarray(masks[0])
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write SAM2 masks for every box in prompts.json.")
    parser.add_argument("--prompts", type=Path, default=Path("HW3/prompts.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("HW3/checkpoints/sam2.1_hiera_small.pt"))
    parser.add_argument("--out", type=Path, default=Path("HW3/sam2_masks"))
    args = parser.parse_args(argv)
    root = segment_core.repo_root()
    prompts_path = args.prompts if args.prompts.is_absolute() else root / args.prompts
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    out = args.out if args.out.is_absolute() else root / args.out
    prompts = segment_core.load_prompts(prompts_path)
    missing_box = [p["image"] for p in prompts if not p.get("box")]
    if missing_box:
        raise SystemExit("SAM2 needs a box. These prompts have none: " + ", ".join(missing_box))

    device = _device()
    print(f"SAM2 device: {device}")
    predictor = _build_predictor(checkpoint, device)
    out.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        image_path = Path(prompt["image"])
        if not image_path.is_absolute():
            image_path = root / image_path
        image = segment_core.read_bgr(image_path)
        mask = _predict(predictor, image, prompt["box"])
        dest = out / f"{image_path.stem}.png"
        if not cv2.imwrite(str(dest), mask):
            raise SystemExit(f"Could not write {dest}")
        print(f"wrote {dest}")
    print("Done. Next: .venv/bin/python HW3/compare_sam2.py")


if __name__ == "__main__":
    main(sys.argv[1:])
