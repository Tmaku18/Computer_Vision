"""Shared file storage for uploaded photos and calibrations.

Uploads live in DATA_DIR (default /tmp/cv-course) and disappear when that
directory is cleared, which happens on every fresh deploy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
for _folder in ("HW1", "HW2"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/cv-course"))
IMAGE_DIR = DATA_DIR / "images"
CALIB_DIR = DATA_DIR / "calibrations"
RESULT_DIR = DATA_DIR / "results"
for _folder in (IMAGE_DIR, CALIB_DIR, RESULT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_CALIBRATION_FILES = 80
THUMB_WIDTH = 480
DEFAULT_CALIBRATION_ID = "default"


def safe_id(value: str) -> str:
    if not value or not all(c.isalnum() or c in "-_" for c in value):
        raise HTTPException(400, "Invalid id.")
    return value


def image_meta(image_id: str) -> dict:
    path = IMAGE_DIR / f"{safe_id(image_id)}.json"
    if not path.is_file():
        raise HTTPException(404, "Image not found (it may have expired on server restart). Upload it again.")
    return json.loads(path.read_text(encoding="utf-8"))


def read_image(image_id: str) -> np.ndarray:
    image_meta(image_id)
    img = cv2.imread(str(IMAGE_DIR / f"{image_id}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise HTTPException(404, "Image file missing.")
    return img


def as_array(points, shape_msg: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise HTTPException(400, shape_msg)
    return arr


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a float or uint8 image (gray or BGR) as PNG."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f"Could not write {path}")
