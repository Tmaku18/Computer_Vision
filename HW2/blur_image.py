#!/usr/bin/env python3
"""
blur_image.py
=============

README — BLUR ONE IMAGE IN SPACE AND IN THE FOURIER DOMAIN
----------------------------------------------------------

WHAT THIS DOES
  Builds a blur kernel (box, Gaussian, disk, or motion) and applies it twice:
  once as a spatial convolution, once by multiplying Fourier transforms.
  The two results are written next to the input, with a difference map and
  the max absolute difference (about 1e-12 when the boundary conditions match).

HOW TO RUN (from the repository root)
  .venv/bin/python HW2/blur_image.py --image photo.jpg --kernel gaussian --size 21 --sigma 4
  .venv/bin/python HW2/blur_image.py --kernel motion --size 15 --angle 25 --mode circular
  .venv/bin/python HW2/blur_image.py            # built-in sharp test image

OUTPUT (HW2/results/)
  <name>_spatial.png  <name>_fft.png  <name>_difference.png  <name>_spectrum_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from filtering_core import (  # noqa: E402
    blur_pair,
    demo_image,
    difference_map,
    make_kernel,
    spectra,
    to_uint8,
)

OUT_DIR = SCRIPT_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Blur an image spatially and with the FFT, and compare.")
    parser.add_argument("--image", type=Path, default=None, help="Input image. Omit to use the built-in test image.")
    parser.add_argument("--kernel", choices=("box", "gaussian", "disk", "motion"), default="gaussian")
    parser.add_argument("--size", type=int, default=21, help="Odd kernel size.")
    parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian standard deviation, in pixels.")
    parser.add_argument("--angle", type=float, default=25.0, help="Motion-blur angle, degrees.")
    parser.add_argument("--mode", choices=("linear", "circular"), default="linear",
                        help="linear = zero padding (matches constant borders); circular = wrap-around.")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.image is None:
        image = demo_image()
        stem = "demo"
    else:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            sys.exit(f"Could not read {args.image}")
        image = image.astype("float64")
        stem = args.image.stem

    kernel = make_kernel(args.kernel, args.size, args.sigma, args.angle)
    result = blur_pair(image, kernel, mode=args.mode)
    spec = spectra(image, kernel, mode=args.mode)
    diff, diff_info = difference_map(result["spatial"], result["frequency"])

    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        "spatial": to_uint8(result["spatial"]),
        "fft": to_uint8(result["frequency"]),
        "difference": diff,
        "spectrum_image": spec["image"],
        "spectrum_kernel": spec["kernel"],
        "spectrum_product": spec["product"],
    }
    for suffix, picture in files.items():
        path = args.output / f"{stem}_{suffix}.png"
        cv2.imwrite(str(path), picture)
        print(f"wrote {path}")

    metrics = result["metrics"]
    print(f"mode {args.mode}, spatial border '{result['boundary']}', kernel {args.kernel} {kernel.shape}")
    print(f"max |spatial - fft| = {metrics['max_abs']:.3e}   RMSE = {metrics['rmse']:.3e}   PSNR = {metrics['psnr_db']:.1f} dB")
    print(f"time  spatial {result['timing_s']['spatial']:.3f} s   FFT {result['timing_s']['fft']:.3f} s")
    print(f"difference map gain {diff_info['gain']:.3g} (1 means the map is unamplified, i.e. essentially black)")


if __name__ == "__main__":
    main()
