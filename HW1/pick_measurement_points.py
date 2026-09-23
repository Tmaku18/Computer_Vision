#!/usr/bin/env python3
"""
pick_measurement_points.py
==========================

README — INTERACTIVELY PICK PIXELS FOR MEASUREMENT (Python)
-------------------------------------------------------------

WHAT THIS DOES
  1. Displays your photo (matplotlib).
  2. You click reference corners, then object corners.
  3. Writes measurement_points.json (and updates example_measurements.json).

BEFORE YOU RUN
  - Run Calibration1.m once (creates camera_calibration.json).
  - Edit DEFAULT_IMAGE and REFERENCE_WORLD_XY below if needed.

HOW TO RUN
  cd /Users/home/GSU/Computer_Vision/Computer_Vision/HW1
  .venv/bin/pip install matplotlib   # if not installed
  .venv/bin/python pick_measurement_points.py

  Optional:
  .venv/bin/python pick_measurement_points.py --image /path/to/photo.jpg

THEN MEASURE
  .venv/bin/python measure_object_calibrated.py --json measurement_points.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cv_core import load_image  # noqa: E402

DEFAULT_IMAGE = Path("/Users/home/Downloads/IMG_5270.jpg")
OUTPUT_JSON = SCRIPT_DIR / "measurement_points.json"
EXAMPLE_JSON = SCRIPT_DIR / "example_measurements.json"

# World (X, Y) for each reference corner you click — inches for HW1.
REFERENCE_WORLD_XY = np.array(
    [
        [0, 0],
        [10, 0],
        [10, 7],
        [0, 7],
    ],
    dtype=np.float64,
)


def pick_points_interactive(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib: pip install matplotlib") from exc

    img = plt.imread(str(image_path))
    num_ref = REFERENCE_WORLD_XY.shape[0]
    num_obj = 4

    ref_clicks: list[tuple[float, float]] = []
    obj_clicks: list[tuple[float, float]] = []
    phase = {"name": "reference"}

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(f"Click {num_ref} REFERENCE corners, then close the window.")

    ref_scatter = ax.plot([], [], "go-", linewidth=2, markersize=8)[0]
    obj_scatter = ax.plot([], [], "r.-", linewidth=2, markersize=10)[0]

    def refresh_title() -> None:
        if phase["name"] == "reference":
            ax.set_title(
                f"REFERENCE: click {len(ref_clicks)}/{num_ref} "
                f"(same order as reference_world_xy). Middle-click undo."
            )
        else:
            ax.set_title(
                f"OBJECT: click {len(obj_clicks)}/{num_obj}. "
                f"Middle-click undo. Close window when done."
            )
        fig.canvas.draw_idle()

    def onclick(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 2:
            if phase["name"] == "object" and obj_clicks:
                obj_clicks.pop()
            elif phase["name"] == "reference" and ref_clicks:
                ref_clicks.pop()
            elif phase["name"] == "object" and not obj_clicks and ref_clicks:
                ref_clicks.pop()
                phase["name"] = "reference"
        elif event.button == 1:
            if phase["name"] == "reference":
                ref_clicks.append((float(event.xdata), float(event.ydata)))
                if len(ref_clicks) >= num_ref:
                    phase["name"] = "object"
            else:
                obj_clicks.append((float(event.xdata), float(event.ydata)))

        if ref_clicks:
            xy = np.array(ref_clicks)
            ref_scatter.set_data(xy[:, 0], xy[:, 1])
        if obj_clicks:
            xy = np.array(obj_clicks)
            obj_scatter.set_data(xy[:, 0], xy[:, 1])
        refresh_title()

    fig.canvas.mpl_connect("button_press_event", onclick)
    refresh_title()
    plt.tight_layout()
    plt.show()

    if len(ref_clicks) != num_ref or len(obj_clicks) != num_obj:
        raise SystemExit(
            f"Need {num_ref} reference and {num_obj} object clicks; "
            f"got {len(ref_clicks)} and {len(obj_clicks)}."
        )

    return np.array(ref_clicks), np.array(obj_clicks)


def save_json(
    image_path: Path,
    reference_world: np.ndarray,
    reference_uv: np.ndarray,
    object_uv: np.ndarray,
    output_path: Path,
) -> dict:
    payload = {
        "image_path": str(image_path),
        "calibration_file": "camera_calibration.json",
        "world_units": "inches",
        "closed_polygon": True,
        "reference_world_xy": reference_world.tolist(),
        "reference_image_uv": reference_uv.tolist(),
        "object_image_uv": object_uv.tolist(),
        "notes": "Written by pick_measurement_points.py — distorted pixel coordinates.",
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick reference and object pixels on a photo.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Image file path")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
        help="Output JSON path (default: measurement_points.json)",
    )
    parser.add_argument(
        "--copy-example",
        action="store_true",
        help="Also overwrite example_measurements.json with the same data.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    print(f"Opening {args.image} — click in the image window.")
    ref_uv, obj_uv = pick_points_interactive(args.image)
    payload = save_json(args.image, REFERENCE_WORLD_XY, ref_uv, obj_uv, args.output)
    print(f"Saved {args.output}")

    if args.copy_example:
        save_json(args.image, REFERENCE_WORLD_XY, ref_uv, obj_uv, EXAMPLE_JSON)
        print(f"Updated {EXAMPLE_JSON}")

    print(json.dumps(payload, indent=2))
    print("\nNext:")
    print(f"  {SCRIPT_DIR / '.venv/bin/python'} measure_object_calibrated.py --json {args.output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
