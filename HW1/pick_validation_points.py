#!/usr/bin/env python3
"""
pick_validation_points.py
=========================

README — CLICK 20 VALIDATION SEGMENTS (2 POINTS EACH)
-----------------------------------------------------

WHAT THIS DOES
  Opens validation_config.json image_path. For each measurement you:
    1. Click start and end of a segment on the object (same plane as reference).
    2. Type ground-truth length in inches (from ruler/tape).
  Appends rows to validation_measurements.csv until you have 20 (or stop early).

HOW TO RUN
  cd /Users/home/GSU/Computer_Vision/Computer_Vision/HW1
  .venv/bin/python pick_validation_points.py
  .venv/bin/python pick_validation_points.py --count 20

THEN
  .venv/bin/python validate_measurements.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "validation_config.json"
DEFAULT_CSV = SCRIPT_DIR / "validation_measurements.csv"


def load_image_path(config_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    return Path(cfg["image_path"])


def pick_segment(image_path: Path) -> tuple[float, float, float, float] | None:
    import matplotlib.pyplot as plt

    img = plt.imread(str(image_path))
    clicks: list[tuple[float, float]] = []

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img)
    ax.set_title("Click segment START then END (2 clicks). Close window to cancel.")
    ax.set_axis_off()

    def onclick(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button != 1:
            return
        clicks.append((float(event.xdata), float(event.ydata)))
        ax.plot(event.xdata, event.ydata, "c+", markersize=12)
        if len(clicks) == 2:
            xs, ys = zip(*clicks)
            ax.plot(xs, ys, "c-", linewidth=2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    if len(clicks) != 2:
        return None
    return clicks[0][0], clicks[0][1], clicks[1][0], clicks[1][1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    image_path = load_image_path(args.config)
    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    fieldnames = [
        "measurement_id",
        "ground_truth_in",
        "u1",
        "v1",
        "u2",
        "v2",
        "description",
    ]
    rows: list[dict] = []
    if args.csv.is_file():
        with args.csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [r for r in rows if float(r.get("ground_truth_in", 0) or 0) > 0]

    start_id = len(rows) + 1
    print(f"Image: {image_path}")
    print(f"Existing valid rows: {len(rows)}. Target: {args.count}.")

    for mid in range(start_id, args.count + 1):
        print(f"\n--- Measurement {mid}/{args.count} ---")
        seg = pick_segment(image_path)
        if seg is None:
            print("Cancelled.")
            break
        gt_str = input("Ground truth length (inches, from ruler): ").strip()
        try:
            gt = float(gt_str)
        except ValueError:
            print("Invalid number, skipping.")
            continue
        desc = input("Short description (optional): ").strip()
        u1, v1, u2, v2 = seg
        rows.append(
            {
                "measurement_id": str(mid),
                "ground_truth_in": str(gt),
                "u1": str(u1),
                "v1": str(v1),
                "u2": str(u2),
                "v2": str(v2),
                "description": desc,
            }
        )

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.csv}")
    print("Run: .venv/bin/python validate_measurements.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
