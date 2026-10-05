#!/usr/bin/env python3
"""
verify_convolution_theorem.py
=============================

README — EXPERIMENT: CONVOLUTION IN SPACE EQUALS MULTIPLICATION IN FREQUENCY
----------------------------------------------------------------------------

WHAT THIS DOES
  For every kernel (box, Gaussian, disk, motion) and both DFT boundary
  conditions (zero-pad / linear, wrap / circular), blurs an image both ways
  and records max absolute difference, RMSE and PSNR. Also records the
  reflected-border blur, which is NOT predicted by the plain DFT, and times
  spatial vs FFT blur as the Gaussian grows.

HOW TO RUN (from the repository root)
  .venv/bin/python HW2/verify_convolution_theorem.py
  .venv/bin/python HW2/verify_convolution_theorem.py --images photo1.jpg photo2.jpg

  With no --images, the built-in test image is used. Extra images in
  HW2/images/ are included automatically.

OUTPUT
  HW2/results/report.json     every row of the experiment
  HW2/results/differences.png bar chart of max |error|
  HW2/results/timing.png      spatial vs FFT time against kernel size
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from filtering_core import demo_image, run_experiment  # noqa: E402

OUT_DIR = SCRIPT_DIR / "results"
IMAGE_DIR = SCRIPT_DIR / "images"


def load_images(paths: list[Path]) -> list[tuple[str, np.ndarray]]:
    found = [("demo", demo_image())]
    extra = list(paths)
    if IMAGE_DIR.is_dir():
        extra += sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in extra:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skipping unreadable {path}", file=sys.stderr)
            continue
        found.append((path.stem, image.astype(np.float64)))
    return found


def save_figures(report: dict) -> None:
    rows = [r for r in report["rows"] if r["image"] == report["images"][0]]
    labels = [f"{r['kernel']}\n{r['boundary']}" for r in rows]
    values = [max(r["max_abs"], 1e-18) for r in rows]
    colors = ["#1a7f4b" if r["expected_match"] else "#b3261e" for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=0)
    ax.set_ylabel("max |spatial − FFT|")
    ax.set_title("Green: theorem says these match. Red: reflected borders, which the DFT does not model.")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "differences.png", dpi=130)
    plt.close(fig)

    timing = report["timing"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sizes = [t["size"] for t in timing]
    ax.plot(sizes, [t["spatial_s"] for t in timing], "o-", label="spatial sum")
    ax.plot(sizes, [t["fft_s"] for t in timing], "s-", label="FFT multiply")
    ax.set_xlabel("Gaussian kernel size (pixels)")
    ax.set_ylabel("seconds")
    ax.set_title("Same blur, two algorithms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "timing.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the convolution theorem on several kernels.")
    parser.add_argument("--images", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    pieces = [run_experiment(image, name) for name, image in load_images(args.images)]
    report = {
        "images": [p["image"] for p in pieces],
        "rows": [row for p in pieces for row in p["rows"]],
        "timing": pieces[0]["timing"],
    }
    matching = [r for r in report["rows"] if r["expected_match"]]
    worst = max(matching, key=lambda r: r["max_abs"])
    report["worst_matching_max_abs"] = worst["max_abs"]
    report["passed"] = worst["max_abs"] < 1e-8
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_figures(report)

    print(f"{'kernel':<24} {'border':<10} {'match?':<8} {'max|e|':>10} {'RMSE':>10}")
    for row in report["rows"]:
        if row["image"] != report["images"][0]:
            continue
        print(f"{row['kernel']:<24} {row['boundary']:<10} {str(row['expected_match']):<8} "
              f"{row['max_abs']:10.2e} {row['rmse']:10.2e}")
    print(f"\nworst matching error {worst['max_abs']:.3e} on {worst['kernel']} ({worst['boundary']})")
    print("PASS" if report["passed"] else "FAIL")
    print(f"wrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
